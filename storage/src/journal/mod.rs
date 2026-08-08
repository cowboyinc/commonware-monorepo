//! An append-only log for storing arbitrary data.
//!
//! Journals provide append-only logging for persisting arbitrary data with fast replay, historical
//! pruning, and rudimentary support for fetching individual items. A journal can be used on its own
//! to serve as a backing store for some in-memory data structure, or as a building block for a more
//! complex construction that prescribes some meaning to items in the log.

use thiserror::Error;

commonware_macros::stability_mod!(ALPHA, pub mod authenticated);
pub mod contiguous;
pub mod segmented;


/// Reusable zstd decompression, shared by the segmented journals.
///
/// `zstd::decode_all` constructs a fresh `DCtx` on every call. Profiling a
/// validator under load showed `ZSTD_createDCtx` at ~25% of total CPU against
/// ~10% for the actual decompression work -- context setup cost 2.5x the
/// decompression itself, because journal reads are many and individually small.
///
/// Holding one `Decompressor` per thread reuses its context across reads. The
/// decompressed bytes are identical either way; only scratch-memory management
/// changes, so this is safe for consensus.
#[cfg(feature = "zstd")]
pub(crate) mod decompress {
    use std::cell::RefCell;
    use zstd::bulk::Decompressor;

    thread_local! {
        static DECOMPRESSOR: RefCell<Option<Decompressor<'static>>> =
            const { RefCell::new(None) };
    }

    /// Decompress a single zstd frame, reusing this thread's context.
    ///
    /// Frames written by `zstd::bulk::compress` pledge their content size in the
    /// header, so the output buffer is sized exactly. Frames without a pledged
    /// size fall back to `zstd::decode_all`.
    pub(crate) fn decode_frame(src: &[u8]) -> std::io::Result<Vec<u8>> {
        // The pledged size sizes the output buffer, so cap what we will trust:
        // a corrupt or malformed header could otherwise request an enormous
        // allocation. Above the cap, fall back to streaming, which grows
        // incrementally instead of allocating up front.
        const MAX_PLEDGED_BYTES: u64 = 256 * 1024 * 1024;
        // `get_frame_content_size` reports the size of the FIRST frame only, and
        // `Decompressor::decompress` decodes exactly one frame -- but `decode_all`
        // (which this replaces) consumes EVERY concatenated frame in the input.
        // Taking the single-frame path on a multi-frame blob therefore returns a
        // silently TRUNCATED value: measured 39 bytes of payload returned as 19,
        // with no error. That corrupts state fast-sync (`RootUnproven`). Only take
        // the fast path when the first frame spans the whole input; otherwise fall
        // back to `decode_all`, which handles the general case.
        match zstd::zstd_safe::find_frame_compressed_size(src) {
            Ok(n) if n == src.len() => {}
            _ => return zstd::decode_all(src),
        }
        let pledged = zstd::zstd_safe::get_frame_content_size(src)
            .ok()
            .flatten()
            .filter(|n| *n <= MAX_PLEDGED_BYTES)
            .and_then(|n| usize::try_from(n).ok());
        let Some(capacity) = pledged else {
            return zstd::decode_all(src);
        };
        DECOMPRESSOR.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                *slot = Some(Decompressor::new()?);
            }
            slot.as_mut()
                .expect("initialized above")
                .decompress(src, capacity)
        })
    }

    #[cfg(test)]
    mod tests {
        use super::decode_frame;

        /// Regression: `get_frame_content_size` reports only the FIRST frame and
        /// `Decompressor::decompress` decodes only one, while the `decode_all` this
        /// replaced consumes every concatenated frame. The fast path therefore
        /// returned a silently TRUNCATED value (39 bytes of payload as 19), which
        /// corrupted state fast-sync with `RootUnproven`.
        #[test]
        fn matches_decode_all_on_concatenated_frames() {
            let a = zstd::bulk::compress(b"first-frame-payload", 3).unwrap();
            let b = zstd::bulk::compress(b"second-frame-payload", 3).unwrap();
            let mut joined = a.clone();
            joined.extend_from_slice(&b);

            assert_eq!(decode_frame(&a).unwrap(), zstd::decode_all(&a[..]).unwrap());

            let got = decode_frame(&joined).unwrap();
            let want = zstd::decode_all(&joined[..]).unwrap();
            assert_eq!(got, want, "truncated: {} bytes vs {}", got.len(), want.len());
            assert_eq!(want, b"first-frame-payloadsecond-frame-payload");
        }

        /// Empty and single-frame inputs must still round-trip.
        #[test]
        fn round_trips_single_frames() {
            for payload in [b"".as_slice(), b"x".as_slice(), &vec![7u8; 100_000]] {
                let c = zstd::bulk::compress(payload, 3).unwrap();
                assert_eq!(decode_frame(&c).unwrap(), payload);
            }
        }
    }
}

#[cfg(all(test, feature = "arbitrary"))]
mod conformance;

/// Errors that can occur when interacting with `Journal`.
#[derive(Debug, Error)]
pub enum Error {
    #[error("merkle error: {0}")]
    Merkle(anyhow::Error),
    #[error("journal error: {0}")]
    Journal(anyhow::Error),
    #[error("runtime error: {0}")]
    Runtime(#[from] commonware_runtime::Error),
    #[error("codec error: {0}")]
    Codec(#[from] commonware_codec::Error),
    #[error("metadata error: {0}")]
    Metadata(#[from] crate::metadata::Error),
    #[error("invalid blob name: {0}")]
    InvalidBlobName(String),
    #[error("invalid blob size: index={0} size={1}")]
    InvalidBlobSize(u64, u64),
    #[error("item too large: size={0}")]
    ItemTooLarge(usize),
    #[error("already pruned to section: {0}")]
    AlreadyPrunedToSection(u64),
    #[error("section out of range: {0}")]
    SectionOutOfRange(u64),
    #[error("usize too small")]
    UsizeTooSmall,
    #[error("offset overflow")]
    OffsetOverflow,
    #[error("unexpected size: expected={0} actual={1}")]
    UnexpectedSize(u32, u32),
    #[error("missing blob: {0}")]
    MissingBlob(u64),
    #[error("item out of range: {0}")]
    ItemOutOfRange(u64),
    #[error("item pruned: {0}")]
    ItemPruned(u64),
    #[error("invalid rewind: {0}")]
    InvalidRewind(u64),
    #[error("compression failed")]
    CompressionFailed,
    #[error("decompression failed")]
    DecompressionFailed,
    #[error("value too large (> u32::MAX)")]
    ValueTooLarge,
    #[error("corruption detected: {0}")]
    Corruption(String),
    /// The offsets journal and the data journal disagree about an item's layout. Either the
    /// on-disk state is corrupted, or the caller passed offsets that aren't byte-adjacent on disk.
    #[error("offset/data layout mismatch in section {section} at offset {offset}: offsets journal expected {expected_len}, data varint reports {actual_len}")]
    OffsetDataMismatch {
        section: u64,
        offset: u64,
        expected_len: usize,
        actual_len: usize,
    },
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("checksum mismatch: expected={0}, found={1}")]
    ChecksumMismatch(u32, u32),
    #[error("empty append")]
    EmptyAppend,
}
