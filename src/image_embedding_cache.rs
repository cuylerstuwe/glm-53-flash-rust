//! Process-local immutable vision results. The owning process fixes weights,
//! precision, codecs and preprocessing settings; nothing survives its restart.
use sha2::{Digest, Sha256};
use std::{collections::VecDeque, sync::Arc};

pub(crate) const MAX_BYTES: usize = 256 << 20;
const MAX_ENTRIES: usize = 64;

pub(crate) fn key(encoded: &[u8], max_tokens: usize) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"alaska-image-embedding-v1");
    hash.update((max_tokens as u64).to_le_bytes());
    hash.update(encoded);
    hash.finalize().into()
}

pub(crate) struct Embedding {
    pub image_key: [u8; 32],
    pub tokens: usize,
    pub rows: Vec<u8>,
}

struct Entry {
    key: [u8; 32],
    value: Arc<Embedding>,
    bytes: usize,
}

pub(crate) struct Cache {
    entries: VecDeque<Entry>,
    bytes: usize,
    max_bytes: usize,
}

impl Cache {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            max_bytes,
        }
    }

    pub(crate) fn get(&mut self, key: &[u8; 32]) -> Option<Arc<Embedding>> {
        let index = self.entries.iter().position(|entry| &entry.key == key)?;
        let entry = self.entries.remove(index).unwrap();
        let result = Arc::clone(&entry.value);
        self.entries.push_back(entry);
        Some(result)
    }

    /// Only complete, validated encoder output reaches this method. Oversized
    /// results bypass storage without evicting useful entries or changing rows.
    pub(crate) fn insert(&mut self, key: [u8; 32], value: Embedding) -> Arc<Embedding> {
        let bytes = value.rows.capacity().saturating_add(256);
        let value = Arc::new(value);
        if bytes > self.max_bytes {
            return value;
        }
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            self.bytes -= self.entries.remove(index).unwrap().bytes;
        }
        while self.entries.len() >= MAX_ENTRIES || self.bytes > self.max_bytes - bytes {
            self.bytes -= self.entries.pop_front().unwrap().bytes;
        }
        self.bytes += bytes;
        self.entries.push_back(Entry {
            key,
            value: Arc::clone(&value),
            bytes,
        });
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn result(byte: u8, bytes: usize) -> Embedding {
        Embedding {
            image_key: [byte; 32],
            tokens: 16,
            rows: vec![byte; bytes],
        }
    }
    #[test]
    fn identity_includes_every_encoded_byte_and_the_token_budget() {
        assert_eq!(key(b"image", 16), key(b"image", 16));
        assert_ne!(key(b"image", 16), key(b"image", 64));
        assert_ne!(key(b"image", 16), key(b"imagf", 16));
    }
    #[test]
    fn immutable_rows_metadata_and_external_reader_survive_eviction() {
        let mut cache = Cache::new(512);
        let first = cache.insert(key(b"a", 16), result(7, 256));
        let hit = cache.get(&key(b"a", 16)).unwrap();
        assert!(Arc::ptr_eq(&first, &hit));
        cache.insert(key(b"b", 16), result(8, 256));
        assert!(cache.get(&key(b"a", 16)).is_none());
        assert_eq!(hit.image_key, [7; 32]);
        assert_eq!(hit.tokens, 16);
        assert_eq!(hit.rows, vec![7; 256]);
        assert!(cache.bytes <= 512);
    }
    #[test]
    fn disabled_oversized_and_new_process_caches_recompute() {
        let mut disabled = Cache::new(0);
        assert_eq!(disabled.insert([1; 32], result(1, 1)).rows, vec![1]);
        assert!(disabled.get(&[1; 32]).is_none());
        let mut small = Cache::new(512);
        small.insert([1; 32], result(1, 16));
        small.insert([2; 32], result(2, 1024));
        assert!(small.get(&[1; 32]).is_some());
        assert!(small.get(&[2; 32]).is_none());
        assert!(Cache::new(512).get(&[1; 32]).is_none());
    }
    #[test]
    fn count_and_memory_bounds_lru_and_replacement_remain_consistent() {
        let mut cache = Cache::new(1 << 20);
        for n in 0..64 {
            cache.insert([n; 32], result(n, 16));
        }
        assert!(cache.get(&[0; 32]).is_some());
        cache.insert([64; 32], result(64, 16));
        assert!(cache.get(&[1; 32]).is_none());
        assert!(cache.get(&[0; 32]).is_some());
        cache.insert([0; 32], result(99, 32));
        assert_eq!(cache.get(&[0; 32]).unwrap().rows, vec![99; 32]);
        assert_eq!(
            cache.bytes,
            cache.entries.iter().map(|entry| entry.bytes).sum::<usize>()
        );
        assert!(cache.entries.len() <= 64 && cache.bytes <= 1 << 20);
    }
}
