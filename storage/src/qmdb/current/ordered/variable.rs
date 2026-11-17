//! An _ordered_ variant of a [crate::qmdb::current] authenticated database for variable-size values
//!
//! This variant maintains the lexicographic-next active key for each active key, enabling exclusion
//! proofs (proving a key is currently inactive). Use [crate::qmdb::current::unordered::variable] if
//! exclusion proofs are not needed.
//!
//! See [Db] for the main database type and [super::ExclusionProof] for proving key inactivity.

pub use super::db::KeyValueProof;
use crate::{
    index::ordered::Index,
    journal::contiguous::variable::Journal,
    merkle::{Graftable, Location},
    qmdb::{
        any::{ordered::variable::Operation, value::VariableEncoding, VariableValue},
        current::VariableConfig as Config,
        operation::Key,
        Error,
    },
    translator::Translator,
    Context,
};
use commonware_codec::{Codec, Read};
use commonware_cryptography::Hasher;
use commonware_parallel::Strategy;

pub type Db<F, E, K, V, H, T, const N: usize, S> = super::db::Db<
    F,
    E,
    Journal<E, Operation<F, K, V>>,
    K,
    VariableEncoding<V>,
    Index<T, Location<F>>,
    H,
    N,
    S,
>;

impl<
        F: Graftable,
        E: Context,
        K: Key,
        V: VariableValue,
        H: Hasher,
        T: Translator,
        const N: usize,
        S: Strategy,
    > Db<F, E, K, V, H, T, N, S>
where
    Operation<F, K, V>: Codec,
{
    /// Initializes a [Db] from the given `config`.
    /// The configured [`Strategy`] is used to parallelize merkleization.
    pub async fn init(
        context: E,
        config: Config<T, <Operation<F, K, V> as Read>::Cfg, S>,
    ) -> Result<Self, Error<F>> {
        crate::qmdb::current::init(context, config).await
    }

    /// Return the first active translated key at or after `T::transform(key)`, or `None` if no
    /// such key exists. This synchronous lookup uses only the in-memory snapshot, without reading
    /// values or the journal, and never wraps to the first key.
    ///
    /// Ordering is in the translated key space. Collision-compressing translators return a shared
    /// translated key, not an original database key. The returned bytes equal the original key
    /// bytes only when translation preserves those bytes. Input lengths follow the translator's
    /// contract.
    pub fn translated_key_at_or_after(&self, key: &[u8]) -> Option<T::Key> {
        self.any.snapshot.translated_key_at_or_after(key)
    }
}

pub mod partitioned {
    //! A variant of [super] that uses a partitioned index for the snapshot.

    use super::*;
    use crate::index::partitioned::ordered::Index;

    /// A partitioned variant of [super::Db].
    ///
    /// The const generic `P` specifies the number of prefix bytes used for partitioning:
    /// - `P = 1`: 256 partitions
    /// - `P = 2`: 65,536 partitions
    /// - `P = 3`: ~16 million partitions
    pub type Db<F, E, K, V, H, T, const P: usize, const N: usize, S> =
        crate::qmdb::current::ordered::db::Db<
            F,
            E,
            Journal<E, Operation<F, K, V>>,
            K,
            VariableEncoding<V>,
            Index<T, Location<F>, P>,
            H,
            N,
            S,
        >;

    impl<
            F: Graftable,
            E: Context,
            K: Key,
            V: VariableValue,
            H: Hasher,
            T: Translator,
            const P: usize,
            const N: usize,
            S: Strategy,
        > Db<F, E, K, V, H, T, P, N, S>
    where
        Operation<F, K, V>: Codec,
    {
        /// Initializes a [Db] from the given `config`.
        /// The configured [`Strategy`] is used to parallelize merkleization.
        pub async fn init(
            context: E,
            config: Config<T, <Operation<F, K, V> as Read>::Cfg, S>,
        ) -> Result<Self, Error<F>> {
            crate::qmdb::current::init(context, config).await
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        mmr,
        qmdb::current::{ordered::tests as shared, tests::variable_config},
        translator::OneCap,
    };
    use bytes::{Buf, BufMut};
    use commonware_codec::{EncodeSize, Read, Write};
    use commonware_cryptography::{sha256::Digest, Sha256};
    use commonware_macros::test_traced;
    use commonware_runtime::{deterministic, Metrics, Runner, Supervisor};
    use commonware_utils::sequence::FixedBytes;
    use std::{cell::Cell, collections::hash_map::DefaultHasher, fmt::Debug, hash::BuildHasher};

    thread_local! {
        static VALUE_CODEC: Cell<(usize, usize, usize)> = const { Cell::new((0, 0, 0)) };
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CountedValue(u64);

    impl EncodeSize for CountedValue {
        fn encode_size(&self) -> usize {
            VALUE_CODEC.with(|counts| {
                let (reads, writes, sizes) = counts.get();
                counts.set((reads, writes, sizes + 1));
            });
            self.0.encode_size()
        }
    }

    impl Write for CountedValue {
        fn write(&self, buf: &mut impl BufMut) {
            VALUE_CODEC.with(|counts| {
                let (reads, writes, sizes) = counts.get();
                counts.set((reads, writes + 1, sizes));
            });
            self.0.write(buf);
        }
    }

    impl Read for CountedValue {
        type Cfg = ();

        fn read_cfg(buf: &mut impl Buf, cfg: &()) -> Result<Self, commonware_codec::Error> {
            VALUE_CODEC.with(|counts| {
                let (reads, writes, sizes) = counts.get();
                counts.set((reads + 1, writes, sizes));
            });
            Ok(Self(u64::read_cfg(buf, cfg)?))
        }
    }

    #[derive(Clone, Default)]
    struct FullKey;

    impl BuildHasher for FullKey {
        type Hasher = DefaultHasher;

        fn build_hasher(&self) -> Self::Hasher {
            DefaultHasher::new()
        }
    }

    impl crate::translator::Translator for FullKey {
        type Key = [u8; 86];

        fn transform(&self, key: &[u8]) -> Self::Key {
            let mut out = [0; 86];
            let len = key.len().min(out.len());
            out[..len].copy_from_slice(&key[..len]);
            out
        }
    }

    type KeyOnlyDb<T> = super::Db<
        mmr::Family,
        deterministic::Context,
        FixedBytes<86>,
        CountedValue,
        Sha256,
        T,
        32,
        commonware_parallel::Sequential,
    >;

    fn key_only_lifecycle<T>() -> String
    where
        T: crate::translator::Translator + Default,
        T::Key: Debug,
    {
        deterministic::Runner::default().start(|context| async move {
            VALUE_CODEC.set((0, 0, 0));
            let mut db = KeyOnlyDb::<T>::init(
                context.child("db"),
                variable_config::<T>("key-only", &context),
            )
            .await
            .unwrap();
            let mut control = KeyOnlyDb::<T>::init(
                context.child("control"),
                variable_config::<T>("control", &context),
            )
            .await
            .unwrap();

            let a = FixedBytes::new([0x10; 86]);
            let mut b = [0x10; 86];
            b[85] += 1;
            let b = FixedBytes::new(b);
            let c = FixedBytes::new([0x30; 86]);
            let d = FixedBytes::new([0xfe; 86]);
            let max = FixedBytes::new([0xff; 86]);
            let translator = T::default();
            let after_b = if translator.transform(a.as_ref()) == translator.transform(b.as_ref()) {
                &a
            } else {
                &c
            };
            let rounds = [
                (vec![], [None, None, None, None, None]),
                (
                    vec![
                        (a.clone(), Some(1)),
                        (b.clone(), Some(2)),
                        (c.clone(), Some(3)),
                        (d.clone(), Some(4)),
                    ],
                    [Some(&a), Some(&b), Some(&c), Some(&d), None],
                ),
                (
                    vec![(a.clone(), Some(5))],
                    [Some(&a), Some(&b), Some(&c), Some(&d), None],
                ),
                (
                    vec![(a.clone(), None)],
                    [Some(&b), Some(&b), Some(&c), Some(&d), None],
                ),
                (
                    vec![(b.clone(), None)],
                    [Some(&c), Some(&c), Some(&c), Some(&d), None],
                ),
                (
                    vec![(a.clone(), Some(6))],
                    [Some(&a), Some(after_b), Some(&c), Some(&d), None],
                ),
                (
                    vec![(a.clone(), None), (c.clone(), None), (d.clone(), None)],
                    [None, None, None, None, None],
                ),
                (
                    vec![(max.clone(), Some(7))],
                    [Some(&max), Some(&max), Some(&max), Some(&max), Some(&max)],
                ),
                (vec![(max.clone(), None)], [None, None, None, None, None]),
            ];
            for (writes, expected) in rounds {
                if !writes.is_empty() {
                    for target in [&mut db, &mut control] {
                        let mut batch = target.new_batch();
                        for (key, value) in &writes {
                            batch = batch.write(key.clone(), value.map(CountedValue));
                        }
                        let batch = batch.merkleize(target, None).await.unwrap();
                        target.apply_batch(batch).await.unwrap();
                    }
                }

                // Check the applied snapshot, the committed snapshot, and the reopened snapshot.
                for phase in 0..3 {
                    assert_eq!(db.root(), control.root());
                    let root = db.root();
                    let keys = crate::index::Unordered::keys(&db.any.snapshot);
                    let metrics = context.encode();
                    let state = context.auditor().state();
                    let codec = VALUE_CODEC.get();
                    for _ in 0..100 {
                        assert_eq!(
                            db.translated_key_at_or_after(&[0; 86]),
                            expected[0].map(|key| translator.transform(key.as_ref()))
                        );
                        assert_eq!(
                            db.translated_key_at_or_after(c.as_ref()),
                            expected[2].map(|key| translator.transform(key.as_ref()))
                        );
                        for (query, expected) in [
                            a.as_ref(),
                            b.as_ref(),
                            &[0x20; 86],
                            d.as_ref(),
                            max.as_ref(),
                        ]
                        .into_iter()
                        .zip(expected)
                        {
                            assert_eq!(
                                db.translated_key_at_or_after(query),
                                expected.map(|key| translator.transform(key.as_ref()))
                            );
                        }
                    }
                    assert_eq!(VALUE_CODEC.get(), codec);
                    assert_eq!(context.auditor().state(), state);
                    assert_eq!(context.encode(), metrics);
                    assert_eq!(crate::index::Unordered::keys(&db.any.snapshot), keys);
                    assert_eq!(db.root(), root);
                    match phase {
                        0 => {
                            db.commit().await.unwrap();
                            control.commit().await.unwrap();
                        }
                        1 => {
                            drop(db);
                            db = KeyOnlyDb::<T>::init(
                                context.child("reopened"),
                                variable_config::<T>("key-only", &context),
                            )
                            .await
                            .unwrap();
                        }
                        _ => {}
                    }
                    assert_eq!(db.root(), root);
                }
            }
            let (reads, writes, sizes) = VALUE_CODEC.get();
            assert!(reads > 0 && writes > 0 && sizes > 0);
            db.destroy().await.unwrap();
            control.destroy().await.unwrap();
            context.auditor().state()
        })
    }

    #[test_traced]
    fn test_translated_key_at_or_after_lifecycle() {
        assert_eq!(
            key_only_lifecycle::<FullKey>(),
            key_only_lifecycle::<FullKey>()
        );
        assert_eq!(
            key_only_lifecycle::<OneCap>(),
            key_only_lifecycle::<OneCap>()
        );
    }

    #[test_traced]
    fn test_translated_key_at_or_after_input_lengths() {
        deterministic::Runner::default().start(|context| async move {
            let mut db = KeyOnlyDb::<FullKey>::init(
                context.child("db"),
                variable_config::<FullKey>("lengths", &context),
            )
            .await
            .unwrap();
            assert_eq!(db.translated_key_at_or_after(&[0; 86]), None);
            let mut key = [0; 86];
            key[0] = 0x10;
            let batch = db
                .new_batch()
                .write(FixedBytes::new(key), Some(CountedValue(1)))
                .write(FixedBytes::new([0xff; 86]), Some(CountedValue(2)))
                .merkleize(&db, None)
                .await
                .unwrap();
            db.apply_batch(batch).await.unwrap();
            assert_eq!(db.translated_key_at_or_after(&[]), Some(key));
            assert_eq!(db.translated_key_at_or_after(&[0x10]), Some(key));
            assert_eq!(db.translated_key_at_or_after(&[0x10; 85]), Some([0xff; 86]));
            let mut long = [0; 87];
            long[..86].copy_from_slice(&key);
            long[86] = 0xff;
            assert_eq!(db.translated_key_at_or_after(&long), Some(key));
            assert_eq!(db.translated_key_at_or_after(&[0xff; 87]), Some([0xff; 86]));
            db.destroy().await.unwrap();
        });
    }

    /// A type alias for the concrete [Db] type used in these unit tests.
    type CurrentTest = super::Db<
        mmr::Family,
        deterministic::Context,
        Digest,
        Digest,
        Sha256,
        OneCap,
        32,
        commonware_parallel::Sequential,
    >;

    #[allow(dead_code)]
    fn _assert_stream_range_is_send(db: &CurrentTest, start: Digest) {
        fn require_send<F: core::future::Future + Send>(_: F) {}
        require_send(async move {
            let stream = db.stream_range(start).await.unwrap();
            futures::pin_mut!(stream);
            let _ = futures::StreamExt::next(&mut stream).await;
        });
    }

    /// Return a [Db] database initialized with a variable config.
    async fn open_db(context: deterministic::Context, partition_prefix: String) -> CurrentTest {
        let cfg = variable_config::<OneCap>(&partition_prefix, &context);
        CurrentTest::init(context, cfg).await.unwrap()
    }

    #[test_traced("DEBUG")]
    pub fn test_current_db_verify_proof_over_bits_in_uncommitted_chunk() {
        shared::test_verify_proof_over_bits_in_uncommitted_chunk(open_db);
    }

    #[test_traced("DEBUG")]
    pub fn test_current_db_range_proofs() {
        shared::test_range_proofs(open_db);
    }

    #[test_traced("DEBUG")]
    pub fn test_current_db_key_value_proof() {
        shared::test_key_value_proof(open_db);
    }

    #[test_traced("WARN")]
    pub fn test_current_db_proving_repeated_updates() {
        shared::test_proving_repeated_updates(open_db);
    }

    #[test_traced("DEBUG")]
    pub fn test_current_db_exclusion_proofs() {
        shared::test_exclusion_proofs(open_db);
    }
}
