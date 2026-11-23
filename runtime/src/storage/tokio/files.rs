//! Bound resident descriptors independently of the number of open logical blobs.
use commonware_utils::sync::Mutex;
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock, Weak},
};

const CAPACITY: usize = 256;
type Identity = (u64, u64);

pub(super) struct Handle {
    path: PathBuf,
    identity: Identity,
    // An unlinked blob has no pathname to reopen. Retain its original inode
    // until the last logical handle and in-flight operation have gone away.
    unlinked: Mutex<Option<Arc<File>>>,
}

struct Cached {
    file: Arc<File>,
    used: u64,
}

#[derive(Default)]
struct State {
    handles: HashMap<PathBuf, Weak<Handle>>,
    files: HashMap<Identity, Cached>,
    clock: u64,
}

impl State {
    fn cache(&mut self, identity: Identity, file: Arc<File>) -> Arc<File> {
        self.clock = self.clock.wrapping_add(1);
        self.files.insert(
            identity,
            Cached {
                file: file.clone(),
                used: self.clock,
            },
        );
        while self.files.len() > CAPACITY {
            let oldest = *self
                .files
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .unwrap()
                .0;
            self.files.remove(&oldest);
        }
        file
    }

    fn file(&mut self, handle: &Handle) -> io::Result<Arc<File>> {
        if let Some(file) = handle.unlinked.lock().as_ref() {
            return Ok(file.clone());
        }
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self.files.get_mut(&handle.identity) {
            entry.used = self.clock;
            return Ok(entry.file.clone());
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&handle.path)?;
        let metadata = file.metadata()?;
        if (metadata.dev(), metadata.ino()) != handle.identity {
            return Err(io::Error::other("blob pathname refers to another inode"));
        }
        Ok(self.cache(handle.identity, Arc::new(file)))
    }
}

#[derive(Default)]
pub(super) struct Files {
    state: Mutex<State>,
}

impl Files {
    // Share across runtime instances too, so removal through another instance
    // preserves handles opened against the same storage directory.
    pub(super) fn shared() -> Arc<Self> {
        static FILES: OnceLock<Arc<Files>> = OnceLock::new();
        FILES.get_or_init(Arc::default).clone()
    }

    pub(super) fn register(&self, path: &Path, file: File) -> io::Result<Arc<Handle>> {
        let mut state = self.state.lock();
        let path = path.canonicalize()?;
        let metadata = file.metadata()?;
        let identity = (metadata.dev(), metadata.ino());
        let named = path.metadata()?;
        if (named.dev(), named.ino()) != identity {
            return Err(io::Error::other("blob pathname changed during open"));
        }
        let handle = state.handles.get(&path).and_then(Weak::upgrade);
        let handle = match handle {
            Some(handle) if handle.identity == identity => handle,
            _ => {
                let handle = Arc::new(Handle {
                    path: path.clone(),
                    identity,
                    unlinked: Mutex::new(None),
                });
                state.handles.insert(path, Arc::downgrade(&handle));
                handle
            }
        };
        // Dead logical handles need no registry entries. Do this periodically,
        // without scanning the registry on every hot read or write.
        if state.handles.len().is_multiple_of(CAPACITY) {
            state.handles.retain(|_, handle| handle.strong_count() > 0);
        }
        // Reuse a resident descriptor when open_versioned opened the same inode.
        if !state.files.contains_key(&identity) {
            state.cache(identity, Arc::new(file));
        }
        Ok(handle)
    }

    pub(super) fn file(&self, handle: &Handle) -> io::Result<Arc<File>> {
        self.state.lock().file(handle)
    }

    /// Run an unlink while excluding descriptor reopen, and preserve any live
    /// logical handles to the removed inodes. A recreated name gets a new handle.
    pub(super) fn remove(&self, path: &Path, partition: bool) -> io::Result<()> {
        let path = if partition {
            path.canonicalize()?
        } else {
            path.parent()
                .ok_or_else(|| io::Error::other("blob has no parent"))?
                .canonicalize()?
                .join(
                    path.file_name()
                        .ok_or_else(|| io::Error::other("blob has no name"))?,
                )
        };
        let mut state = self.state.lock();
        let handles = state
            .handles
            .iter()
            .filter(|(name, _)| {
                if partition {
                    name.starts_with(&path)
                } else {
                    *name == &path
                }
            })
            .filter_map(|(_, handle)| handle.upgrade())
            .collect::<Vec<_>>();
        let retained = handles
            .iter()
            .map(|handle| state.file(handle))
            .collect::<io::Result<Vec<_>>>()?;
        let result = if partition {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        for (handle, file) in handles.iter().zip(retained) {
            // A recursive removal may fail after unlinking some children.
            // Preserve those inodes even when the operation returns an error.
            let still_linked = result.is_err()
                && handle
                    .path
                    .metadata()
                    .is_ok_and(|metadata| (metadata.dev(), metadata.ino()) == handle.identity);
            if !still_linked {
                *handle.unlinked.lock() = Some(file);
                state.files.remove(&handle.identity);
                state.handles.remove(&handle.path);
            }
        }
        result
    }
}
