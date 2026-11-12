//! The server's caches: values kept within a byte budget, least recently
//! used first out. Ported from `server/lru.py`.

use std::collections::HashMap;
use std::hash::Hash;

/// Values by key, each of a size its owner gives, within `budget_bytes`
/// and, if given, `capacity` entries: keeping one drops the least recently
/// used until both hold, and a value larger than the whole budget is not
/// kept. Its owner holds a lock around every use, as [`memoized`] holds
/// the one it is given.
pub struct LruCache<K, V> {
    pub budget_bytes: usize,
    pub capacity: Option<usize>,
    pub bytes: usize,
    /// The entries dropped to make room.
    pub evictions: usize,
    /// The entries, least recently used first.
    order: Vec<K>,
    entries: HashMap<K, (V, usize)>,
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    pub fn new(budget_bytes: usize, capacity: Option<usize>) -> LruCache<K, V> {
        LruCache {
            budget_bytes,
            capacity,
            bytes: 0,
            evictions: 0,
            order: Vec::new(),
            entries: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The keys, the least recently used first.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.order.iter()
    }

    /// The value kept for `key`, now the most recently used; `None` when
    /// none is kept.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.entries.contains_key(key) {
            self.touch(key);
            self.entries.get(key).map(|(value, _)| value)
        } else {
            None
        }
    }

    fn touch(&mut self, key: &K) {
        if let Some(position) = self.order.iter().position(|kept| kept == key) {
            let moved = self.order.remove(position);
            self.order.push(moved);
        }
    }

    /// Keep `value`, of `size` bytes, for `key` in place of any value kept
    /// for it; `false`, keeping nothing, when it alone outgrows the budget.
    pub fn put(&mut self, key: K, value: V, size: usize) -> bool {
        if size > self.budget_bytes {
            return false;
        }
        self.pop(&key);
        self.order.push(key.clone());
        self.entries.insert(key, (value, size));
        self.bytes += size;
        while self.bytes > self.budget_bytes
            || self.capacity.is_some_and(|capacity| self.entries.len() > capacity)
        {
            let evicted = self.order.remove(0);
            if let Some((_, evicted_size)) = self.entries.remove(&evicted) {
                self.bytes -= evicted_size;
            }
            self.evictions += 1;
        }
        true
    }

    /// Drop the value kept for `key`: it, or `None` when none is kept.
    pub fn pop(&mut self, key: &K) -> Option<V> {
        if let Some((value, size)) = self.entries.remove(key) {
            if let Some(position) = self.order.iter().position(|kept| kept == key) {
                self.order.remove(position);
            }
            self.bytes -= size;
            Some(value)
        } else {
            None
        }
    }
}

/// The value `cache` keeps for `key`, else the one `compute` returns with
/// its size, which `cache` then keeps. A value the budget refuses still
/// answers the computed value, as the Python's does.
///
/// The Python's `memoized` also releases the lock while computing and
/// re-checks after; in Rust that pattern belongs to the call site, which
/// holds the cache's lock and decides what runs outside it.
pub fn memoized<K, V, F>(cache: &mut LruCache<K, V>, key: K, compute: F) -> Option<V>
where
    K: Eq + Hash + Clone,
    V: Clone,
    F: FnOnce() -> Option<(V, usize)>,
{
    if let Some(value) = cache.get(&key) {
        return Some(value.clone());
    }
    let (value, size) = compute()?;
    cache.put(key, value.clone(), size);
    Some(value)
}
