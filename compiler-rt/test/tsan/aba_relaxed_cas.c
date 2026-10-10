// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
//
// TP: heap-reuse ABA with relaxed ordering on the load and on both sides of
// the CAS. Relaxed atomics take separate fast paths in the runtime (OpLoad
// without acquire, OpCAS with relaxed/relaxed), so the ABA hooks there need
// their own coverage. Same interleaving as aba_heap_reuse.c.
#include <assert.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static _Atomic(int *) ptr;
static sem_t to_t2, to_t1;

static void *t2(void *arg) {
  (void)arg;
  sem_wait(&to_t2);
  int *A = atomic_load(&ptr);
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
  atomic_store(&ptr, malloc(sizeof(int)));

  pthread_t th;
  pthread_create(&th, NULL, t2, NULL);

  int *expected = atomic_load_explicit(&ptr, memory_order_relaxed);
  sem_post(&to_t2);
  sem_wait(&to_t1);              // T2 frees A and stores B at its address

  int ok = atomic_compare_exchange_strong_explicit(
      &ptr, &expected, malloc(sizeof(int)), memory_order_relaxed,
      memory_order_relaxed);
  assert(ok);
  pthread_join(th, NULL);
  fprintf(stderr, "done\n");
  return 0;
}
// CHECK: WARNING: ThreadSanitizer: ABA detected
// CHECK: done
