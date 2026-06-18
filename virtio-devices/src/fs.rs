// Copyright 2026 The Cloud Hypervisor Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Barrier};
use std::{io, result};

use anyhow::anyhow;
use event_monitor::event;
use log::{error, info};
use seccompiler::SeccompAction;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use virtio_queue::{Queue, QueueT};
use virtiofsd::passthrough::PassthroughFs;
use virtiofsd::server::Server;
use vm_memory::bitmap::AtomicBitmap;
use vm_memory::{ByteValued, GuestAddressSpace, GuestMemoryAtomic};
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};
use vm_virtio::AccessPlatform;
use vmm_sys_util::eventfd::EventFd;

use super::{
    ActivateError, ActivateResult, EPOLL_HELPER_EVENT_LAST, EpollHelper, EpollHelperError,
    EpollHelperHandler, Error as DeviceError, VIRTIO_F_VERSION_1, VirtioCommon, VirtioDevice,
    VirtioDeviceType,
};
use crate::seccomp_filters::Thread;
use crate::{GuestMemoryMmap, VirtioInterrupt, VirtioInterruptType};

pub const VIRTIO_FS_TAG_LEN: usize = 36;

// Available event index for epoll helper.
const QUEUE_AVAIL_EVENT: u16 = EPOLL_HELPER_EVENT_LAST + 1;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Descriptor chain too short")]
    DescriptorChainTooShort,
    #[error("Invalid descriptor")]
    InvalidDescriptor,
    #[error("Failed adding used index")]
    QueueAddUsed(#[source] virtio_queue::Error),
}

#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct VirtioFsConfig {
    pub tag: [u8; VIRTIO_FS_TAG_LEN],
    pub num_request_queues: u32,
}

impl Default for VirtioFsConfig {
    fn default() -> Self {
        VirtioFsConfig {
            tag: [0; VIRTIO_FS_TAG_LEN],
            num_request_queues: 0,
        }
    }
}

// SAFETY: only a series of integers
unsafe impl ByteValued for VirtioFsConfig {}

struct FsEpollHandler {
    mem: GuestMemoryAtomic<GuestMemoryMmap>,
    queue: Queue,
    queue_evt: EventFd,
    interrupt_cb: Arc<dyn VirtioInterrupt>,
    queue_index: usize,
    kill_evt: EventFd,
    pause_evt: EventFd,
    server: Arc<Server<PassthroughFs, AtomicBitmap>>,
}

impl FsEpollHandler {
    fn process_queue(&mut self) -> result::Result<bool, Error> {
        let mut used_descs = false;
        let mem = self.mem.memory();

        while let Some(desc_chain) = self.queue.pop_descriptor_chain(&*mem) {
            let reader = virtiofsd::descriptor_utils::Reader::new(&*mem, desc_chain.clone())
                .map_err(|_| Error::InvalidDescriptor)?;
            let writer = virtiofsd::descriptor_utils::Writer::new(&*mem, desc_chain.clone())
                .map_err(|_| Error::InvalidDescriptor)?;

            let head_index = desc_chain.head_index();

            let len = self
                .server
                .handle_message::<()>(reader, writer, None)
                .map_err(|e| {
                    error!("FUSE server error: {e:?}");
                    Error::InvalidDescriptor
                })?;

            self.queue
                .add_used(&*mem, head_index, len as u32)
                .map_err(Error::QueueAddUsed)?;
            used_descs = true;
        }

        Ok(used_descs)
    }

    fn signal_used_queue(&self) -> result::Result<(), DeviceError> {
        self.interrupt_cb
            .trigger(VirtioInterruptType::Queue(self.queue_index as u16))
            .map_err(|e| {
                error!("Failed to signal used queue: {e:?}");
                DeviceError::FailedSignalingUsedQueue(e)
            })
    }

    fn run(
        &mut self,
        paused: &AtomicBool,
        paused_sync: &Barrier,
    ) -> result::Result<(), EpollHelperError> {
        let mut helper = EpollHelper::new(&self.kill_evt, &self.pause_evt)?;
        helper.add_event(self.queue_evt.as_raw_fd(), QUEUE_AVAIL_EVENT)?;
        helper.run(paused, paused_sync, self)?;

        Ok(())
    }
}

impl EpollHelperHandler for FsEpollHandler {
    fn handle_event(
        &mut self,
        _helper: &mut EpollHelper,
        event: &epoll::Event,
    ) -> result::Result<(), EpollHelperError> {
        let ev_type = event.data as u16;
        match ev_type {
            QUEUE_AVAIL_EVENT => {
                self.queue_evt.read().map_err(|e| {
                    EpollHelperError::HandleEvent(anyhow!("Failed to get queue event: {e:?}"))
                })?;
                let needs_notification = self.process_queue().map_err(|e| {
                    EpollHelperError::HandleEvent(anyhow!("Failed to process queue: {e:?}"))
                })?;
                if needs_notification {
                    self.signal_used_queue().map_err(|e| {
                        EpollHelperError::HandleEvent(anyhow!("Failed to signal used queue: {e:?}"))
                    })?;
                }
            }
            _ => {
                return Err(EpollHelperError::HandleEvent(anyhow!(
                    "Unexpected event: {ev_type}"
                )));
            }
        }
        Ok(())
    }
}

pub struct Fs {
    common: VirtioCommon,
    id: String,
    tag: String,
    server: Arc<Server<PassthroughFs, AtomicBitmap>>,
    seccomp_action: SeccompAction,
    exit_evt: EventFd,
}

#[derive(Serialize, Deserialize)]
pub struct FsState {
    pub avail_features: u64,
    pub acked_features: u64,
}

impl Fs {
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn new(
        id: String,
        tag: &str,
        shared_dir: PathBuf,
        xattr: bool,
        writeback: bool,
        num_queues: usize,
        queue_size: u16,
        seccomp_action: SeccompAction,
        exit_evt: EventFd,
        state: Option<FsState>,
    ) -> io::Result<Fs> {
        let (avail_features, acked_features, paused) = if let Some(state) = state {
            info!("Restoring virtio-fs {id}");
            (state.avail_features, state.acked_features, true)
        } else {
            let avail_features = 1u64 << VIRTIO_F_VERSION_1;
            (avail_features, 0, false)
        };
        if shared_dir.is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("shared_dir must not be a symlink: {}", shared_dir.display()),
            ));
        }

        let fs_cfg = virtiofsd::passthrough::Config {
            root_dir: shared_dir.to_str().unwrap().to_string(),
            xattr,
            writeback,
            ..Default::default()
        };
        let fs = PassthroughFs::new(fs_cfg).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to create passthrough fs: {e}"),
            )
        })?;
        let server = Arc::new(Server::new(fs));

        // There is 1 hiprio queue + num_request_queues
        let actual_queues_count = 1 + num_queues;

        Ok(Fs {
            common: VirtioCommon {
                device_type: VirtioDeviceType::Fs as u32,
                queue_sizes: vec![queue_size; actual_queues_count],
                paused_sync: Some(Arc::new(Barrier::new(actual_queues_count + 1))), // +1 for VMM thread
                avail_features,
                acked_features,
                min_queues: 2, // At least hiprio queue + 1 request queue
                paused: Arc::new(AtomicBool::new(paused)),
                ..Default::default()
            },
            id,
            tag: tag.to_string(),
            server,
            seccomp_action,
            exit_evt,
        })
    }

    fn state(&self) -> FsState {
        FsState {
            avail_features: self.common.avail_features,
            acked_features: self.common.acked_features,
        }
    }
}

impl VirtioDevice for Fs {
    fn device_type(&self) -> u32 {
        self.common.device_type
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &self.common.queue_sizes
    }

    fn features(&self) -> u64 {
        self.common.avail_features
    }

    fn ack_features(&mut self, value: u64) {
        self.common.ack_features(value);
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        let mut config = VirtioFsConfig::default();
        let tag_bytes = self.tag.as_bytes();
        let len = std::cmp::min(tag_bytes.len(), config.tag.len());
        config.tag[..len].copy_from_slice(&tag_bytes[..len]);
        config.num_request_queues = (self.common.queue_sizes.len() - 1) as u32;

        self.read_config_from_slice(config.as_slice(), offset, data);
    }

    fn activate(&mut self, context: crate::device::ActivationContext) -> ActivateResult {
        let crate::device::ActivationContext {
            mem,
            interrupt_cb,
            queues,
            device_status,
        } = context;
        self.common.activate(&queues, interrupt_cb.clone())?;

        let (kill_evt, pause_evt) = self.common.dup_eventfds()?;

        for (i, (_, queue, queue_evt)) in queues.into_iter().enumerate() {
            let kill_evt_dup = kill_evt.try_clone().map_err(|e| {
                error!("Failed to clone kill eventfd: {e:?}");
                ActivateError::BadActivate
            })?;
            let pause_evt_dup = pause_evt.try_clone().map_err(|e| {
                error!("Failed to clone pause eventfd: {e:?}");
                ActivateError::BadActivate
            })?;

            let mut handler = FsEpollHandler {
                mem: mem.clone(),
                queue,
                queue_evt,
                interrupt_cb: interrupt_cb.clone(),
                queue_index: i,
                kill_evt: kill_evt_dup,
                pause_evt: pause_evt_dup,
                server: self.server.clone(),
            };

            let paused = self.common.paused.clone();
            let paused_sync = self.common.paused_sync.clone();

            self.common.spawn_worker(
                &self.id,
                &self.seccomp_action,
                Thread::VirtioFs,
                &self.exit_evt,
                device_status.clone(),
                interrupt_cb.clone(),
                move || handler.run(&paused, paused_sync.as_ref().unwrap()),
            )?;
        }

        event!("virtio-device", "activated", "id", &self.id);
        Ok(())
    }

    fn reset(&mut self) {
        self.common.reset();
        event!("virtio-device", "reset", "id", &self.id);
    }

    fn set_access_platform(&mut self, access_platform: Arc<dyn AccessPlatform>) {
        self.common.set_access_platform(access_platform);
    }
}

impl Pausable for Fs {
    fn pause(&mut self) -> result::Result<(), MigratableError> {
        self.common.pause()
    }

    fn resume(&mut self) -> result::Result<(), MigratableError> {
        self.common.resume()
    }
}

impl Snapshottable for Fs {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn snapshot(&mut self) -> std::result::Result<Snapshot, MigratableError> {
        Snapshot::new_from_state(&self.state())
    }
}

impl Transportable for Fs {}
impl Migratable for Fs {}
