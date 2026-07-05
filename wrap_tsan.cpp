#include <atomic>
#include <cstdio>

extern "C" {
void __tsan_read1(void *);
void __tsan_read2(void *);
void __tsan_read4(void *);
void __tsan_read8(void *);
void __tsan_read16(void *);
void __tsan_write1(void *);
void __tsan_write2(void *);
void __tsan_write4(void *);
void __tsan_write8(void *);
void __tsan_write16(void *);
}

static std::atomic<long long> g_reads{0}, g_writes{0};

#define WRAP_READ(fn)                                                          \
  static void my_##fn(void *addr) {                                            \
    g_reads.fetch_add(1, std::memory_order_relaxed);                           \
    fn(addr);                                                                  \
  }

#define WRAP_WRITE(fn)                                                         \
  static void my_##fn(void *addr) {                                            \
    g_writes.fetch_add(1, std::memory_order_relaxed);                          \
    fn(addr);                                                                  \
  }

WRAP_READ(__tsan_read1)
WRAP_READ(__tsan_read2)
WRAP_READ(__tsan_read4)
WRAP_READ(__tsan_read8)
WRAP_READ(__tsan_read16)

WRAP_WRITE(__tsan_write1)
WRAP_WRITE(__tsan_write2)
WRAP_WRITE(__tsan_write4)
WRAP_WRITE(__tsan_write8)
WRAP_WRITE(__tsan_write16)

#define DYLD_INTERPOSE(_replacement, _replacee)                                \
  __attribute__((used)) static struct {                                        \
    const void *r;                                                             \
    const void *o;                                                             \
  } _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = {  \
      (const void *)(unsigned long)&_replacement,                              \
      (const void *)(unsigned long)&_replacee}

DYLD_INTERPOSE(my___tsan_read1, __tsan_read1);
DYLD_INTERPOSE(my___tsan_read2, __tsan_read2);
DYLD_INTERPOSE(my___tsan_read4, __tsan_read4);
DYLD_INTERPOSE(my___tsan_read8, __tsan_read8);
DYLD_INTERPOSE(my___tsan_read16, __tsan_read16);
DYLD_INTERPOSE(my___tsan_write1, __tsan_write1);
DYLD_INTERPOSE(my___tsan_write2, __tsan_write2);
DYLD_INTERPOSE(my___tsan_write4, __tsan_write4);
DYLD_INTERPOSE(my___tsan_write8, __tsan_write8);
DYLD_INTERPOSE(my___tsan_write16, __tsan_write16);

__attribute__((destructor)) static void print_counts() {
  fprintf(stderr, "[interpose] reads=%lld writes=%lld total=%lld\n",
          (long long)g_reads.load(), (long long)g_writes.load(),
          (long long)(g_reads.load() + g_writes.load()));
}