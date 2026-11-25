/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/VM/CellKind.h"
#include "hermes/VM/HeapAlign.h"

#include <cassert>
#include <cstdint>

namespace hermes {
namespace vm {

/// Root of the shared-heap cell hierarchy. Deliberately **not** a GCCell, so
/// that it can't be passed to any existing functions that expect a cell
/// allocated in the normal heap. Unlike GCCell, SharedGCCell is non-moving,
/// hence no forwarding pointer. It also has no size inline, all cells read
/// the size in the same path.
class SharedGCCell {
  CellKind kind_;

#ifndef NDEBUG
  static constexpr uint16_t kMagic{0x5ce1};
  const uint16_t magic_{kMagic};
  /// From SharedGC's group-wide counter. GCBase::nextObjectID is per-Runtime
  /// and would hand the same id to cells in different Runtimes.
  uint32_t debugAllocationId_{0};
#endif

 public:
  SharedGCCell() = default;
  ~SharedGCCell() = default;

  SharedGCCell(const SharedGCCell &) = delete;
  void operator=(const SharedGCCell &) = delete;

  SharedGCCell(SharedGCCell &&) = delete;
  SharedGCCell &operator=(SharedGCCell &&) = delete;

  CellKind getKind() const {
    return kind_;
  }

  /// NOTE: for the allocator's use. Cells are handed out as raw storage, so
  /// this is how a freshly allocated slot becomes self-identifying.
  void setKind(CellKind kind) {
    assert(isSharedCellKind(kind) && "not a shared cell kind");
    kind_ = kind;
  }

#ifndef NDEBUG
  /// False on raw storage that no leaf type has been constructed into: the
  /// allocator does not run a constructor, so the magic is written by the
  /// placement-new of the concrete type, exactly as for GCCell.
  bool isValid() const {
    return magic_ == kMagic && isSharedCellKind(getKind());
  }

  void setDebugAllocationId(uint32_t id) {
    debugAllocationId_ = id;
  }
  uint32_t getDebugAllocationId() const {
    return debugAllocationId_;
  }
#endif
};

static_assert(
    alignof(SharedGCCell) <= HeapAlign,
    "SharedGCCell's alignment exceeds the heap's");

} // namespace vm
} // namespace hermes
