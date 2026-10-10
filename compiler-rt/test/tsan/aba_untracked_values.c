// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
//
// TN: pointer-sized atomics holding NULL, a global address, or a stack
// address. None of these is a heap block, so the detector must treat them as
// untracked: no report, and no crash from looking up metadata for a global or
// a stack address.
#include <stdatomic.h>
#include <stdio.h>

static int g_obj;

int main(void) {
  int s_obj;
  _Atomic(int *) p;
  int *e;

  // Global object: g -> NULL -> g.
  atomic_store(&p, &g_obj);
  e = atomic_load(&p);
  atomic_store(&p, NULL);
  atomic_store(&p, &g_obj);
  atomic_compare_exchange_strong(&p, &e, &s_obj);

  // Stack object: s -> NULL -> s.
  e = atomic_load(&p);
  atomic_store(&p, NULL);
  atomic_store(&p, &s_obj);
  atomic_compare_exchange_strong(&p, &e, NULL);

  // NULL itself.
  e = atomic_load(&p);
  atomic_compare_exchange_strong(&p, &e, &g_obj);

  fprintf(stderr, "done\n");
  return 0;
}
// CHECK-NOT: ABA detected
// CHECK: done
