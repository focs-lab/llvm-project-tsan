// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
//
// TP: heap-reuse ABA in C++, with std::atomic<Node *> and nodes from
// new/delete instead of malloc/free. operator new/delete go through their own
// interceptors, so this checks they stamp and clear the allocation epoch too.
#include <atomic>
#include <cassert>
#include <cstdio>
#include <pthread.h>
#include <semaphore.h>

struct Node {
  Node *next;
  int value;
};

static std::atomic<Node *> head;
static sem_t to_t2, to_t1;

static void *t2(void *) {
  sem_wait(&to_t2);
  Node *A = head.load();
  delete A;
  Node *B = new Node{nullptr, 2};  // reuses A's address with a fresh epoch
  assert(B == A);                  // ABA precondition
  head.store(B);
  sem_post(&to_t1);
  return nullptr;
}

int main() {
  sem_init(&to_t2, 0, 0);
  sem_init(&to_t1, 0, 0);
  head.store(new Node{nullptr, 1});

  pthread_t th;
  pthread_create(&th, nullptr, t2, nullptr);

  Node *expected = head.load();
  sem_post(&to_t2);
  sem_wait(&to_t1);                // T2 deletes A and stores B at its address

  bool ok = head.compare_exchange_strong(expected, nullptr);
  assert(ok);
  pthread_join(th, nullptr);
  fprintf(stderr, "done\n");
  return 0;
}
// CHECK: WARNING: ThreadSanitizer: ABA detected
// CHECK: done
