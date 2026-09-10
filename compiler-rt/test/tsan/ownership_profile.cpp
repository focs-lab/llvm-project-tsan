// RUN: %clangxx_tsan -O1 -mllvm -tsan-instrument-memory-accesses=0 %s -o %t
// RUN: env TSAN_OPTIONS=ownership_profile=1:ownership_profile_allocations=0 %run %t 2>&1 | FileCheck %s

#include <stdint.h>

extern "C" void __tsan_write8(void *);
extern "C" void *__tsan_get_current_fiber();
extern "C" void *__tsan_create_fiber(unsigned);
extern "C" void __tsan_destroy_fiber(void *);
extern "C" void __tsan_switch_to_fiber(void *, unsigned);

alignas(8) uint64_t private_cell;
alignas(8) uint64_t shared_cell;

int main() {
  void *original = __tsan_get_current_fiber();
  void *fiber = __tsan_create_fiber(0);
  __tsan_write8(&private_cell);
  __tsan_write8(&private_cell);
  __tsan_write8(&shared_cell);
  __tsan_switch_to_fiber(fiber, 0);
  __tsan_write8(&shared_cell);
  __tsan_switch_to_fiber(original, 0);
  __tsan_write8(&shared_cell);
  __tsan_destroy_fiber(fiber);
}

// CHECK: TSAN_OWNERSHIP_PROFILE {"schema":2
// CHECK-SAME: "shared_cells":1
// CHECK-SAME: "shared_accesses":2
// CHECK-SAME: "transitions":1
// CHECK-SAME: "shared_hits":1
