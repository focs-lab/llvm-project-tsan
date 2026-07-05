#include <cstdio>

// noinline prevents the compiler from inlining these into main and then
// proving the address never escapes (which would let mem2reg promote the
// memory to a register and delete the loop entirely).

// Hot loop over the SAME address repeatedly — after first touch,
// every subsequent access should be an L0 cache hit.
__attribute__((noinline)) long sum_repeated(volatile long *p, int n) {
  long s = 0;
  for (int i = 0; i < n; ++i) {
    *p += 1; // store
    s += *p; // load
  }
  return s;
}

// Loop over DIFFERENT addresses each iteration — defeats the cache,
// every access is a miss.
__attribute__((noinline)) long sum_array(volatile long *arr, int n) {
  long s = 0;
  for (int i = 0; i < n; ++i) {
    arr[i] += 1; // store
    s += arr[i]; // load
  }
  return s;
}

int main() {
  volatile long x = 0;
  long r1 = sum_repeated(&x, 10000);

  volatile long arr[64] = {0};
  long r2 = sum_array(arr, 64);

  // Force both results to be used so the compiler can't discard the calls.
  printf("r1=%ld r2=%ld\n", r1, r2);
  return 0;
}