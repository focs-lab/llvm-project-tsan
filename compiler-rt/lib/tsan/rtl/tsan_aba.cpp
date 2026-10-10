//===-- tsan_aba.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Out-of-line state for ABA detection: the allocation epoch counter, the
// per-thread load cache, and the side table holding identities for pool slots
// annotated with __tsan_aba_pool_alloc/__tsan_aba_pool_free.
//
//===----------------------------------------------------------------------===//
#include "tsan_rtl_aba.h"

#include "tsan_rtl.h"

namespace __tsan {

// Monotonic allocation identity; 0 is reserved for dead/untracked memory.
atomic_uint64_t g_aba_epoch = {1};

__attribute__((tls_model("initial-exec")))
THREADLOCAL AbaCache aba_cache;

__attribute__((tls_model("initial-exec")))
THREADLOCAL u64 aba_report_count;

// Pool slot side table. Open addressing with a fixed probe window; removal
// zeroes the slot rather than leaving a tombstone. Lookups must therefore scan
// the whole window and never stop early, or a removal could hide a live entry
// behind it -- the resulting miss returns epoch 0, which the CAS check reads as
// an identity change and reports falsely. Overflowing the window drops the
// insert instead, costing only a missed detection.

static const uptr kPoolSize = 1 << 16;  // 64Ki entries, 1MB, lazily faulted
static const uptr kPoolMask = kPoolSize - 1;
static const uptr kPoolProbe = 8;

struct PoolSlot {
  atomic_uintptr_t addr;  // 0 == empty
  atomic_uint64_t epoch;
};

static PoolSlot g_pool_map[kPoolSize];

// Stays 0 until the first annotation, so unannotated programs pay one relaxed
// load per lookup instead of walking the probe window.
static atomic_uint32_t g_pool_used;

static inline uptr PoolHash(uptr p) {
  // Fibonacci hashing; the high product bits mix in the low address bits that
  // distinguish adjacent slots.
  return (uptr)(((u64)p * 0x9E3779B97F4A7C15ull) >> 47) & kPoolMask;
}

static void PoolInsert(uptr p, u64 epoch) {
  uptr h = PoolHash(p);
  for (uptr i = 0; i < kPoolProbe; i++) {
    PoolSlot *s = &g_pool_map[(h + i) & kPoolMask];
    uptr cur = atomic_load(&s->addr, memory_order_acquire);
    if (cur != 0 && cur != p)
      continue;  // taken by another address
    // On a lost race cur holds the winner's address; if it claimed p too,
    // the slot is still ours to stamp.
    if (cur == 0 &&
        !atomic_compare_exchange_strong(&s->addr, &cur, p,
                                        memory_order_acq_rel) &&
        cur != p)
      continue;
    atomic_store(&s->epoch, epoch, memory_order_release);
    return;
  }
}

static void PoolRemove(uptr p) {
  uptr h = PoolHash(p);
  for (uptr i = 0; i < kPoolProbe; i++) {
    PoolSlot *s = &g_pool_map[(h + i) & kPoolMask];
    if (atomic_load(&s->addr, memory_order_acquire) != p)
      continue;
    atomic_store(&s->epoch, 0, memory_order_relaxed);
    atomic_store(&s->addr, 0, memory_order_release);
    return;
  }
}

static u64 PoolLookup(uptr p) {
  uptr h = PoolHash(p);
  for (uptr i = 0; i < kPoolProbe; i++) {
    PoolSlot *s = &g_pool_map[(h + i) & kPoolMask];
    if (atomic_load(&s->addr, memory_order_acquire) == p)
      return atomic_load(&s->epoch, memory_order_acquire);
  }
  return 0;
}

u64 GetPointerEpoch(uptr p) {
  // Every pointer-sized atomic value lands here, including integers and
  // tagged pointers. GetBlock computes a meta address from p without a range
  // check, so a value outside application memory would fault.
  if (!IsAppMem(p))
    return 0;
  if (MBlock *b = ctx->metamap.GetBlock(p))
    return b->alloc_epoch;
  if (atomic_load(&g_pool_used, memory_order_relaxed) == 0)
    return 0;
  return PoolLookup(p);
}

}  // namespace __tsan

using namespace __tsan;

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_aba_pool_alloc(void *addr) {
  atomic_store(&g_pool_used, 1, memory_order_relaxed);
  u64 epoch = atomic_fetch_add(&g_aba_epoch, 1, memory_order_relaxed);
  PoolInsert((uptr)addr, epoch);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_aba_pool_free(void *addr) { PoolRemove((uptr)addr); }
SANITIZER_INTERFACE_ATTRIBUTE
unsigned long long __tsan_aba_report_count(void) { return aba_report_count; }
}  // extern "C"
