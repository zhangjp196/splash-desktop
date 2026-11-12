//! The LRU cache's tests, ported from `dev/tests/server/test_lru_cache.py`.

use splash_server::support::lru::LruCache;

#[test]
fn least_recently_used_values_leave_first_to_fit_the_budget() {
    let mut cache = LruCache::new(10, None);
    assert!(cache.put("a", 1, 4));
    assert!(cache.put("b", 2, 4));
    // Reading a value makes it the most recently used.
    assert_eq!(cache.get(&"a"), Some(&1));
    assert!(cache.put("c", 3, 4));
    let keys: Vec<&str> = cache.keys().copied().collect();
    assert_eq!(keys, ["a", "c"]);
    assert_eq!(cache.get(&"b"), None);
    assert_eq!((cache.len(), cache.bytes, cache.evictions), (2, 8, 1));
}

#[test]
fn capacity_bounds_the_entries() {
    let mut cache = LruCache::new(100, Some(2));
    for key in ["a", "b", "c"] {
        cache.put(key, key.to_uppercase(), 1);
    }
    let keys: Vec<&str> = cache.keys().copied().collect();
    assert_eq!(keys, ["b", "c"]);
    assert_eq!((cache.bytes, cache.evictions), (2, 1));
}

#[test]
fn a_value_larger_than_the_budget_is_not_kept() {
    let mut cache = LruCache::new(10, None);
    cache.put("a", 1, 6);
    for key in ["a", "b"] {
        assert!(!cache.put(key, 2, 11));
    }
    let keys: Vec<&str> = cache.keys().copied().collect();
    assert_eq!(keys, ["a"]);
    assert_eq!(cache.get(&"a"), Some(&1));
    assert_eq!(cache.bytes, 6);
    // A value as large as the whole budget is kept alone.
    assert!(cache.put("b", 2, 10));
    let keys: Vec<&str> = cache.keys().copied().collect();
    assert_eq!(keys, ["b"]);
    assert_eq!((cache.bytes, cache.evictions), (10, 1));
}

#[test]
fn a_value_replaces_the_one_kept_for_its_key() {
    let mut cache = LruCache::new(10, None);
    cache.put("a", 1, 6);
    cache.put("b", 2, 2);
    assert!(cache.put("a", 3, 8));
    let keys: Vec<&str> = cache.keys().copied().collect();
    assert_eq!(keys, ["b", "a"]);
    assert_eq!(cache.get(&"a"), Some(&3));
    assert_eq!((cache.bytes, cache.evictions), (10, 0));
    assert_eq!(cache.pop(&"a"), Some(3));
    assert_eq!(cache.pop(&"a"), None);
    let keys: Vec<&str> = cache.keys().copied().collect();
    assert_eq!(keys, ["b"]);
    assert_eq!(cache.bytes, 2);
}
