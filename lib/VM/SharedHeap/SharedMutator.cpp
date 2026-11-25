/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/VM/SharedHeap/SharedMutator.h"

#include "hermes/VM/SharedHeap/SharedGC.h"

#include <cassert>

namespace hermes {
namespace vm {

SharedMutator::SharedMutator(SharedGC &gc, Runtime &runtime)
    : gc_(gc), runtime_(runtime) {
  gc_.registerMutator(this);
}

SharedMutator::~SharedMutator() {
#ifndef NDEBUG
  detached_ = true;
#endif
  gc_.unregisterMutator(this);
}

void SharedMutator::push(SharedGCCell *cell, uint32_t seq) {
  assert(!detached_ && "push on a SharedMutator that is being destroyed");
  // An odd stamp means marking was on. An even one means the caller pushed
  // with marking off, and a spill would then accept it into a worklist the
  // cycle has finished with.
  assert(seq % 2 == 1 && "pushed a barrier entry with marking off");
  if (LLVM_UNLIKELY(ring_.full()))
    gc_.spill(*this);
  ring_.push(cell, seq);
}

} // namespace vm
} // namespace hermes
