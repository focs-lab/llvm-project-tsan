// RUN: %clang_tsan -O1 -DRELOAD=0 %s -o %t-control && %run %t-control 2>&1 | FileCheck %s --check-prefix=CONTROL
// RUN: %clang_tsan -O1 -DRELOAD=1 %s -o %t-limit && %run %t-limit 2>&1 | FileCheck %s --check-prefix=LIMIT
//
// LIMITATION (documents current behaviour; flip the limit run when fixed).
// The per-thread note keeps only the latest load of each atomic. Here main
// loads A, another thread frees A and publishes B at the same address, and
// main reloads the atomic before CASing with the value from its first load.
// The reload overwrites the note with B's epoch, so the ABA is missed.
// The control run (RELOAD=0) skips the reload and must report.
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

  int *expected = atomic_load(&ptr);
  sem_post(&to_t2);
  sem_wait(&to_t1);

#if RELOAD
  int *again = atomic_load(&ptr); // same address, new object: note overwritten
  (void)again;
#endif
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
