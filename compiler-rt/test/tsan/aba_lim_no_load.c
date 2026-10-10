// RUN: %clang_tsan -O1 -DLOAD=1 %s -o %t-control && %run %t-control 2>&1 | FileCheck %s --check-prefix=CONTROL
// RUN: %clang_tsan -O1 -DLOAD=0 %s -o %t-limit && %run %t-limit 2>&1 | FileCheck %s --check-prefix=LIMIT
//
// LIMITATION (documents current behaviour; flip the limit run when fixed).
// The CAS's expected value did not come from an atomic operation on this
// variable: main keeps the pointer it allocated and initialised the atomic
// with atomic_init, which is a plain store. The thread has no note for the
// atomic, so a real heap-reuse ABA is missed.
// The control run (LOAD=1) does an atomic_load first and must report, which
// shows the silence in the limit run comes from the missing note.
#include <assert.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static _Atomic(int *) ptr;
static int *A;
static sem_t to_t2, to_t1;

static void *t2(void *arg) {
  (void)arg;
  sem_wait(&to_t2);
  free(A);
  int *B = malloc(sizeof(int));  // reuses A's address with a fresh epoch
  assert(B == A);                // ABA precondition
  atomic_store(&ptr, B);
  sem_post(&to_t1);
  return NULL;
}

int main(void) {
  sem_init(&to_t2, 0, 0);
  sem_init(&to_t1, 0, 0);
  A = malloc(sizeof(int));
  atomic_init(&ptr, A);          // plain store: no note

  pthread_t th;
  pthread_create(&th, NULL, t2, NULL);

#if LOAD
  int *expected = atomic_load(&ptr);
#else
  int *expected = A;             // value from the allocation, not the atomic
#endif
  sem_post(&to_t2);
  sem_wait(&to_t1);

  int ok = atomic_compare_exchange_strong(&ptr, &expected, NULL);
  assert(ok);
  pthread_join(th, NULL);
  fprintf(stderr, "done\n");
  return 0;
}
// CONTROL: WARNING: ThreadSanitizer: ABA detected
// CONTROL: done
// LIMIT-NOT: ABA detected
// LIMIT: done
