/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtest/gtest.h"

#include "hermes/VM/BoxedSharedObject.h"
#include "hermes/VM/SharedHeap/SharedArray.h"
#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/VM/SharedHeap/SharedGCBoxing-inline.h"

#include "VMRuntimeTestHelpers.h"

#include <memory>

namespace {

using namespace hermes;
using namespace hermes::vm;

constexpr size_t kAmpleBudget = 8 << 20;

/// A shared heap plus runtimes attached to it through Runtime::create, rather
/// than mutators constructed by hand.
struct SharedAttachTest : public ::testing::Test {
  std::shared_ptr<SharedGC> gc{std::make_shared<SharedGC>(kAmpleBudget)};

  std::shared_ptr<Runtime> attach() {
    return Runtime::create(kTestRTConfig, gc);
  }

  ~SharedAttachTest() override {
    // Runtimes are test locals and already gone, so the registry is empty and
    // a cycle with no roots frees the heap.
    gc->collect();
  }
};

TEST_F(SharedAttachTest, Attach_RegistersAndDeregisters) {
  EXPECT_EQ(0u, gc->mutatorCountSlow());
  {
    auto rt = attach();
    EXPECT_EQ(1u, gc->mutatorCountSlow());
    {
      auto rt2 = attach();
      EXPECT_EQ(2u, gc->mutatorCountSlow());
    }
    EXPECT_EQ(1u, gc->mutatorCountSlow()) << "the second did not deregister";
  }
  EXPECT_EQ(0u, gc->mutatorCountSlow());
}

TEST_F(SharedAttachTest, Attach_IsOptional) {
  // A runtime created with no shared heap must be unaffected by one existing.
  auto plain = Runtime::create(kTestRTConfig);
  EXPECT_EQ(0u, gc->mutatorCountSlow());
}

TEST_F(SharedAttachTest, Deregistration_BeatsMemberTeardown) {
  // The drop belongs in a post-shutdown deleter, not left to the member's own
  // destructor. Both deregister eventually, so only the timing tells them
  // apart: deleters run in registration order, so one registered here runs
  // after the runtime's own and can see that it already happened.
  size_t countInsideDeleter = 99;
  {
    auto rt = attach();
    rt->addPostShutdownDeleter(
        [&] { countInsideDeleter = gc->mutatorCountSlow(); });
    ASSERT_EQ(1u, gc->mutatorCountSlow());
  }
  EXPECT_EQ(0u, countInsideDeleter)
      << "deregistration did not happen in the runtime's own post-shutdown "
         "deleter";
}

TEST_F(SharedAttachTest, Collect_EndToEndThroughRealRuntimes) {
  // The whole thing, with nothing hand-wired: two attached runtimes and cells
  // allocated and boxed through makeBoxed. collect() needs nothing from either
  // runtime: neither is inside a barrier, so it does not wait.
  auto rtA = attach();
  auto rtB = attach();
  Runtime &a = *rtA;
  Runtime &b = *rtB;

  SharedArray *keptByA = nullptr;
  SharedArray *keptByB = nullptr;
  {
    GCScope scope{a, "SharedAttachTest"};
    auto box = a.makeHandle(gc->makeBoxed<SharedArray>(
        a, SharedArray::allocationSize(1), uint32_t{1}));
    ASSERT_TRUE(*box);
    keptByA = vmcast<SharedArray>(box->getSharedCell());
  }
  {
    GCScope scope{b, "SharedAttachTest"};
    auto box = b.makeHandle(gc->makeBoxed<SharedArray>(
        b, SharedArray::allocationSize(1), uint32_t{1}));
    ASSERT_TRUE(*box);
    keptByB = vmcast<SharedArray>(box->getSharedCell());
  }
  // Named by nobody.
  ASSERT_NE(nullptr, SharedArray::create(*gc, 1));
  ASSERT_EQ(3u, gc->liveCellCountSlow());

  gc->collect();

  EXPECT_EQ(2u, gc->liveCellCountSlow())
      << "the unnamed cell should have been the only casualty";
  EXPECT_EQ(1u, keptByA->length());
  EXPECT_EQ(1u, keptByB->length());
}

} // namespace
