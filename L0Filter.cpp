#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>

using namespace std;

constexpr uint32_t CACHE_SIZE = 4096;
constexpr uint64_t CACHE_MASK = 0xFFF; // 12 bits for index (4096 slots)

// Address Tag: Bits 15 to 57
constexpr uint64_t ADDR_TAG_MASK = 0x03FFFFFFFFFF8000ULL;

// Write Flag: Bit 63
constexpr uint64_t WRITE_BIT_MASK = 0x8000000000000000ULL;
constexpr uint64_t IGNORE_WRITE_BIT_MASK = 0x7FFFFFFFFFFFFFFFULL;

uint64_t L0Filter[CACHE_SIZE] = {0};
uint64_t local_epoch = 0;

bool check_read_hit(uint64_t raw_addr) {
  uint32_t index = (raw_addr >> 3) & CACHE_MASK;
  uint64_t cached_entry = L0Filter[index];
  uint64_t expected_read = (raw_addr & ADDR_TAG_MASK) | local_epoch;
  // Mask out the Write Bit because a previous write also satisfies a read check
  if (((cached_entry ^ expected_read) & IGNORE_WRITE_BIT_MASK) == 0) {
    return true; // Cache Hit: Skip TSan
  }

  // Cache Miss: Update the slot and fallback to TSan
  L0Filter[index] = expected_read;
  return false;
}

bool check_write_hit(uint64_t raw_addr) {
  uint32_t index = (raw_addr >> 3) & CACHE_MASK;
  uint64_t cached_entry = L0Filter[index];
  uint64_t expected_write =
      (raw_addr & ADDR_TAG_MASK) | local_epoch | WRITE_BIT_MASK;
  if ((cached_entry ^ expected_write) == 0) {
    return true; // Cache Hit: Skip TSan
  }

  // Cache Miss: Update the slot and fallback to TSan
  L0Filter[index] = expected_write;
  return false;
}

void simulate_synchronization() {
  local_epoch++;
  if (local_epoch > 0x7FFF) [[unlikely]] { // 15-bit rollover
    std::cout << "  [Runtime] Epoch rollover (32767 reached)! Vector-flushing "
                 "cache...\n";
    std::memset(L0Filter, 0, sizeof(L0Filter));
    local_epoch = 0;
  }
}

void reset_state() {
  std::memset(L0Filter, 0, sizeof(L0Filter));
  local_epoch = 1; // Start at 1 for clean testing
}
