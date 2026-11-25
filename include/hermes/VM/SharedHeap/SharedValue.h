/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/VM/SharedHeap/SharedGCCell.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace hermes {
namespace vm {

/// One slot in a shared cell: an element of a SharedArray today, a field of a
/// SharedStruct later.
class SharedValue {
 public:
  /// Reading a slot copies it, and a slot in the heap is read by the collector
  /// while its owning runtime writes it, so the copy is an atomic load.
  ///
  /// An acquire, because a copy that holds a pointer may be followed to the
  /// cell, and the reader must then see the cell as it was built. That takes
  /// an acquire here and a release in the store that published the pointer.
  SharedValue(const SharedValue &o)
      : raw_(o.raw_.load(std::memory_order_acquire)) {}
  /// Not assignable, so a slot cannot be overwritten by a plain store.
  SharedValue &operator=(const SharedValue &) = delete;

  /// An empty slot. Every slot of a freshly created cell must be set to this or
  /// initialized before anything can trace it: allocation hands back
  /// uninitialised storage, so an unwritten slot is whatever the previous
  /// occupant left behind, and the collector would follow it as a pointer.
  static constexpr SharedValue empty() {
    return SharedValue{0};
  }

  /// Encode \p cell into SharedValue.
  /// \pre \p cell cannot be nullptr.
  static SharedValue encodePointer(SharedGCCell *cell) {
    assert(cell && "SharedValue can't encode a nullptr");
    return SharedValue{reinterpret_cast<uint64_t>(cell)};
  }

  /// Check if the current value is empty.
  bool isEmpty() const {
    // Relaxed: nothing is read through the answer.
    return raw_.load(std::memory_order_relaxed) == 0;
  }

  /// Whether the collector should follow this slot.
  bool isPointer() const {
    // Today it's always a pointer unless it's empty. Relaxed: nothing is read
    // through the answer, and following the pointer means getPointer, which
    // loads again with acquire.
    return raw_.load(std::memory_order_relaxed) != 0;
  }

  /// \pre isPointer().
  SharedGCCell *getPointer() const {
    assert(isPointer() && "slot holds no pointer");
    // Acquire, as in the copy constructor: the caller follows the pointer. Do
    // a memcpy here to avoid int-to-ptr compiler warning.
    auto value = static_cast<uintptr_t>(raw_.load(std::memory_order_acquire));
    SharedGCCell *ptr;
    memcpy(&ptr, &value, sizeof(value));
    return ptr;
  }

 private:
  explicit constexpr SharedValue(uint64_t raw) : raw_(raw) {}

  /// The encoded value. Today it's SharedGCCell pointer value.
  /// This must not be nullptr unless it's empty (uninitialized or reset).
  ///
  /// Atomic because the collector traces a slot while its owning runtime
  /// stores to it. And multiple runtimes could read/write the slot
  /// concurrently, but it needs to be tearing-free, which is not guaranteed on
  /// 32-bit platform unless it's atomic.
  std::atomic<uint64_t> raw_;
};

static_assert(
    alignof(SharedValue) <= HeapAlign,
    "a shared slot must not out-align the heap");

} // namespace vm
} // namespace hermes
