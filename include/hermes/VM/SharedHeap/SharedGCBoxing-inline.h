/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/VM/SharedHeap/SharedGC.h"

#include "hermes/VM/BoxedSharedObject.h"
#include "hermes/VM/Runtime.h"

namespace hermes {
namespace vm {

template <typename T, typename... Args>
PseudoHandle<BoxedSharedObject>
SharedGC::makeBoxed(Runtime &runtime, uint32_t size, Args &&...args) {
  assert(
      runtime.isAttachedTo(*this) &&
      "only an attached runtime's map is in this heap's root set");
  // Checked before the box is allocated, so a request that cannot be served
  // does not cost one.
  const OptValue<uint32_t> alignedSize = checkedAlignedSize<T>(size);
  if (LLVM_UNLIKELY(!alignedSize))
    return PseudoHandle<BoxedSharedObject>{};

  // The box first, and outside the lock. See the declaration for why.
  struct : public Locals {
    PinnedValue<BoxedSharedObject> box;
  } lv;
  LocalsRAII lraii(runtime, &lv);
  lv.box = BoxedSharedObject::create(runtime);
  // After the box, whose allocation may have found boxes dead, and before the
  // lock, to keep the walk over the map out of the critical section.
  runtime.pruneSharedObjectBoxesIfNeeded();

  std::lock_guard<std::mutex> lk(mtx_);
  T *cell = makeALocked<T>(*alignedSize, std::forward<Args>(args)...);
  if (LLVM_UNLIKELY(!cell))
    return PseudoHandle<BoxedSharedObject>{};

  // Neither of these allocates, which is what makes them safe under the
  // allocation lock.
  lv.box->setSharedCell(cell);
  runtime.setBoxForSharedCell(cell, lv.box);
  return PseudoHandle<BoxedSharedObject>::create(*lv.box);
}

} // namespace vm
} // namespace hermes
