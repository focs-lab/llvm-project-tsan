// The read-lock twin of deadlock_detector_exact_prelock_destroy.cpp: the same
// destroy-vs-lock race through pthread_rwlock_rdlock, whose pre-lock is
// MutexPreReadLock. Main holds r for writing through the real
// pthread_rwlock_wrlock (dlsym RTLD_NEXT, not seen by TSan); the other
// thread's pthread_rwlock_rdlock blocks in the real call after its pre-lock;
// main destroys r and releases it, and the other thread's post-lock reads r.
// Without "seen", r has no sync object and the flag must not skip the hook
// (skip count 0); with "seen", main first takes and drops r through TSan, the
// flag skips the hook, and the destroy still reports.
// RUN: %clangxx_tsan -O1 %s -o %t -ldl
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=0:print_deadlock_detector_stats=1 %deflake %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,NOSKIP
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=1:print_deadlock_detector_stats=1 %deflake %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,NOSKIP
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=0:print_deadlock_detector_stats=1 %deflake %run %t seen 2>&1 | FileCheck %s --check-prefixes=CHECK,NOSKIP
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=1:print_deadlock_detector_stats=1 %deflake %run %t seen 2>&1 | FileCheck %s --check-prefixes=CHECK,SKIP
// REQUIRES: linux
#include "test.h"
#include <dlfcn.h>

static pthread_rwlock_t r = PTHREAD_RWLOCK_INITIALIZER;
static int (*real_wrlock)(pthread_rwlock_t *);
static int (*real_unlock)(pthread_rwlock_t *);

static void *locker(void *arg) {
  barrier_wait(&barrier);
  pthread_rwlock_rdlock(&r); // blocks in the real call
  pthread_rwlock_unlock(&r);
  return NULL;
}

int main(int argc, char **argv) {
  real_wrlock =
      (int (*)(pthread_rwlock_t *))dlsym(RTLD_NEXT, "pthread_rwlock_wrlock");
  real_unlock =
      (int (*)(pthread_rwlock_t *))dlsym(RTLD_NEXT, "pthread_rwlock_unlock");
  if (!real_wrlock || !real_unlock)
    return 1;
  barrier_init(&barrier, 2);
  if (argc > 1) { // "seen"
    pthread_rwlock_rdlock(&r);
    pthread_rwlock_unlock(&r);
  }
  real_wrlock(&r);
  pthread_t t;
  pthread_create(&t, NULL, locker, NULL);
  barrier_wait(&barrier);
  sleep(1); // the other thread is blocked in the real lock by now
  pthread_rwlock_destroy(&r);
  real_unlock(&r);
  pthread_join(t, NULL);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: WARNING: ThreadSanitizer:
// CHECK: pthread_rwlock_destroy
// CHECK: DONE
// NOSKIP: pre-lock hooks skipped=0{{$}}
// SKIP: pre-lock hooks skipped={{[1-9][0-9]*$}}
