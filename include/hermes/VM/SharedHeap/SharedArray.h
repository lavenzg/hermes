/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/VM/CellKind.h"
#include "hermes/VM/HeapAlign.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedGCCell.h"
#include "hermes/VM/SharedHeap/SharedMutator.h"
#include "hermes/VM/SharedHeap/SharedValue.h"

#include "llvh/Support/TrailingObjects.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>

namespace hermes {
namespace vm {

/// A fixed-length array of shared slots.
///
/// Elements sit inline, immediately after the header, following ArrayStorage's
/// use of TrailingObjects. A SharedArray never grows -- TC39 fixes its length
/// at construction -- so there is no butterfly and nothing to reallocate: the
/// element count picks a size bucket once and the cell never changes size. That
/// is also why length_ needs no atomic; only the elements are ever written
/// after construction.
///
/// The length is stored rather than derived from the cell's extent. The only
/// size available is the owning segment's stride, which is the *rounded* size
/// class, so a 24-element array and a 27-element one can share one -- and
/// `length` is JS-observable, so it has to be exact whatever the collector
/// needs. It costs nothing: the four bytes sit in padding the alignment forces
/// anyway.
class SharedArray final
    : public SharedGCCell,
      private llvh::TrailingObjects<SharedArray, SharedValue> {
  friend llvh::TrailingObjects<SharedArray, SharedValue>;

  uint32_t length_;

 public:
  static constexpr CellKind getCellKind() {
    return CellKind::SharedArrayKind;
  }

  static bool classof(const SharedGCCell *cell) {
    return cell->getKind() == getCellKind();
  }

  /// Bytes an array of \p length elements occupies.
  /// \return UINT32_MAX, which makeBoxed and makeA refuse, if no cell can be
  ///   that big.
  static constexpr uint32_t allocationSize(uint32_t length) {
    // In 64 bits: past about 2^29 elements the size does not fit in 32, and a
    // truncated one would be a tiny cell that the constructor overruns.
    const uint64_t size =
        sizeof(SharedArray) + uint64_t{length} * sizeof(SharedValue);
    if (size > SharedGC::maxCellSize())
      return std::numeric_limits<uint32_t>::max();
    return heapAlignSize(static_cast<gcheapsize_t>(size));
  }

#ifdef UNIT_TEST
  /// Allocate an array of \p length elements in \p gc and construct it.
  /// Test-only: the array is in no runtime's box map, so nothing keeps it
  /// alive. Production code allocates one with makeBoxed, sized by
  /// allocationSize.
  ///
  /// \return nullptr if no cell can hold \p length elements or the shared
  ///   heap's budget is exhausted.
  static SharedArray *create(SharedGC &gc, uint32_t length) {
    return gc.makeA<SharedArray>(allocationSize(length), length);
  }
#endif

  uint32_t length() const {
    return length_;
  }

  SharedValue at(uint32_t i) const {
    assert(i < length_ && "index out of range");
    return data()[i];
  }

  /// Store into element \p i of an array nothing else can reach yet.
  /// \pre no other thread can observe this array.
  void initField(uint32_t i, SharedValue v) {
    assert(i < length_ && "index out of range");
    // The checkable half of the precondition. An initField over a slot that
    // already holds something is the shape a barrier violation takes.
    assert(data()[i].isEmpty() && "initField over a live slot");
    data()[i].initField(v);
  }

  /// Overwrite element \p i, recording what it held first.
  void set(uint32_t i, SharedValue v, SharedMutator &mutator) {
    assert(i < length_ && "index out of range");
    data()[i].set(v, mutator);
  }

  /// Elements are contiguous, so a tracer walks length() of them from here.
  SharedValue *begin() {
    return data();
  }
  SharedValue *end() {
    return data() + length_;
  }

 private:
  /// makeA placement-news this; nothing else should.
  friend class SharedGC;

  explicit SharedArray(uint32_t length) : length_(length) {
    // Every element is emptied here, and that is a correctness requirement
    // before it is a semantic one. makeA hands back uninitialised storage, so
    // an element left unwritten holds whatever the slot's previous occupant
    // left behind, and the collector would follow that as a pointer. It also
    // happens to be the observable initial state: a fresh SharedArray reads as
    // undefined at every index.
    //
    // memset, not a loop: this runs under the shared heap's allocation lock,
    // which makeA holds across construction.
    [[maybe_unused]] const SharedValue empty = SharedValue::empty();
    assert(
        !*reinterpret_cast<const uintptr_t *>(&empty) &&
        "SharedValue::empty() is not 0, so memset cannot write it");
    std::memset(static_cast<void *>(data()), 0, length * sizeof(SharedValue));
  }

  SharedValue *data() {
    return getTrailingObjects<SharedValue>();
  }
  const SharedValue *data() const {
    return getTrailingObjects<SharedValue>();
  }
};

static_assert(
    alignof(SharedArray) <= HeapAlign,
    "SharedArray must not out-align the heap");

} // namespace vm
} // namespace hermes
