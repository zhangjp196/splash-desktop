#pragma once

// Upstream 1.7.2's device-scope fence for the cross-threadgroup split reduce.
// The fork's `split_reduce.h` uses the older fence that upstream replaced:
// only thread zero fenced, but Metal promises a store's visibility to other
// threads only through fences run by those threads. The ported relative
// attention kernel includes this header after `split_reduce.h`, so its
// `split_arrive_last` resolves here while shipped kernels keep the old one.

#include <metal_stdlib>

using namespace metal;

// True in every thread of the threadgroup that arrives last.
inline bool split_arrive_last(device atomic_uint *counter, uint splits,
                              uint thread_index, threadgroup uint *arrival) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  threadgroup_barrier(mem_flags::mem_none);
  if (thread_index == 0)
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (*arrival != splits - 1)
    return false;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  return true;
}

// Returns the tile's counter to zero; called by the reducing threadgroup.
inline void split_release(device atomic_uint *counter, uint thread_index) {
  if (thread_index == 0)
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
}
