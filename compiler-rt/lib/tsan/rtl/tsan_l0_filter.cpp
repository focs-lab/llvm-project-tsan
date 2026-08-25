//===-- tsan_l0_filter.cpp ------------------------------------*- C++ -*-===//
//
// Runtime-side hook for the L0 inline cache filter. Interceptors for
// synchronization primitives call this to invalidate any stale
// "skip TSan" entries cached by code the compiler instrumented.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_rtl.h"

using namespace __tsan;
using namespace __sanitizer;

extern "C" {
__attribute__((visibility("default"))) thread_local u64 TSanL0Filter[4096] = {
    0};

__attribute__((visibility("default"))) thread_local u64 TSanLocalEpoch = 1;
}

namespace {
constexpr u64 kMaxEpoch = 0x7FFF;  // 15-bit rollover
}

extern "C" void __tsan_l0_advance_epoch() {
  u64 next = TSanLocalEpoch + 1;
  if (__builtin_expect(next > kMaxEpoch, 0)) {
    internal_memset(TSanL0Filter, 0, sizeof(TSanL0Filter));
    TSanLocalEpoch = 1;
    return;
  }
  TSanLocalEpoch = next;
}

extern "C" void __tsan_l0_warmup_thread() {
  ScopedIgnoreInterceptors ignore;
  TSanL0Filter[0] = 0;
  TSanLocalEpoch = 1;
}
