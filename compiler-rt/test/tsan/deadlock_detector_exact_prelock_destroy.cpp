// deadlock_detector_exact_prelock=1 skips the lock-order detector's pre-lock
// hook only when the mutex's sync object already exists. The
// hook's lookup creates it on a mutex TSan has not seen, and that is what lets
// a pthread_mutex_destroy between this lock's pre-lock and post-lock make its
// imitated free-write and report the destroy-vs-lock race.
// Here main holds m through the real pthread_mutex_lock (dlsym RTLD_NEXT, not
// seen by TSan), so m, a static mutex, has no sync object. The other thread's
// pthread_mutex_lock blocks in the real call after its pre-lock; main then
// destroys m (EBUSY) and releases it, and the other thread's post-lock reads
// m: the race with the destroy is reported, with the flag as without it.
// With "seen", main first locks and unlocks m through TSan, so m has a sync
// object and the flag does skip the other thread's pre-lock hook (the skip
// count shows it); the destroy still finds the sync object and reports.
// RUN: %clangxx_tsan -O1 %s -o %t -ldl
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=0:print_deadlock_detector_stats=1 %deflake %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,NOSKIP
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=1:print_deadlock_detector_stats=1 %deflake %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,NOSKIP
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=0:print_deadlock_detector_stats=1 %deflake %run %t seen 2>&1 | FileCheck %s --check-prefixes=CHECK,NOSKIP
// RUN: %env_tsan_opts=deadlock_detector_exact_prelock=1:print_deadlock_detector_stats=1 %deflake %run %t seen 2>&1 | FileCheck %s --check-prefixes=CHECK,SKIP
// REQUIRES: linux
#include "test.h"
#include <dlfcn.h>
#include <errno.h>

static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static int (*real_lock)(pthread_mutex_t *);
static int (*real_unlock)(pthread_mutex_t *);

static void *locker(void *arg) {
  barrier_wait(&barrier);
  pthread_mutex_lock(&m); // first lock TSan sees; blocks in the real call
  pthread_mutex_unlock(&m);
  return NULL;
}

int main(int argc, char **argv) {
  real_lock =
      (int (*)(pthread_mutex_t *))dlsym(RTLD_NEXT, "pthread_mutex_lock");
  real_unlock =
      (int (*)(pthread_mutex_t *))dlsym(RTLD_NEXT, "pthread_mutex_unlock");
  if (!real_lock || !real_unlock)
    return 1;
  barrier_init(&barrier, 2);
  if (argc > 1) { // "seen"
    pthread_mutex_lock(&m);
    pthread_mutex_unlock(&m);
  }
  real_lock(&m);
  pthread_t t;
  pthread_create(&t, NULL, locker, NULL);
  barrier_wait(&barrier);
  sleep(1); // the other thread is blocked in the real lock by now
  int rc = pthread_mutex_destroy(&m);
  fprintf(stderr, "destroy rc=%s\n",
          rc == EBUSY ? "EBUSY"
          : rc == 0   ? "0"
                      : "other");
  real_unlock(&m);
  pthread_join(t, NULL);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: WARNING: ThreadSanitizer:
// CHECK: pthread_mutex_destroy
// CHECK: DONE
// NOSKIP: pre-lock hooks skipped=0{{$}}
// SKIP: pre-lock hooks skipped={{[1-9][0-9]*$}}
