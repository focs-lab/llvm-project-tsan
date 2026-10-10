// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
//
// The per-thread ABA report counter goes up by one on the thread whose CAS
// reported, and stays 0 on the other thread. Same interleaving as
// aba_heap_reuse.c.
#include <assert.h>
#include <pthread.h>
#include <sanitizer/tsan_interface.h>
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

  sem_wait(&to_t2);              // wait until main's CAS has reported
  fprintf(stderr, "t2=%llu\n", __tsan_aba_report_count());
  return NULL;
}

int main(void) {
  sem_init(&to_t2, 0, 0);
  sem_init(&to_t1, 0, 0);
  atomic_store(&ptr, malloc(sizeof(int)));

  pthread_t th;
  pthread_create(&th, NULL, t2, NULL);

  int *expected = atomic_load(&ptr);
  sem_post(&to_t2);
  sem_wait(&to_t1);              // T2 frees A and stores B at its address

  fprintf(stderr, "before=%llu\n", __tsan_aba_report_count());
  atomic_compare_exchange_strong(&ptr, &expected, malloc(sizeof(int)));
  fprintf(stderr, "after=%llu\n", __tsan_aba_report_count());

  sem_post(&to_t2);
  pthread_join(th, NULL);
  return 0;
}
// CHECK: before=0
// CHECK: WARNING: ThreadSanitizer: ABA detected
// CHECK: after=1
// CHECK: t2=0