/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hermes/VM/SharedHeap/SharedGC.h"
#include "hermes/Support/Compiler.h"
#include "hermes/Support/OSCompat.h"
#include "hermes/VM/Runtime.h"
#include "hermes/VM/SharedHeap/SharedGC-inline.h"
#include "hermes/VM/SharedHeap/SharedMutator.h"

#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <thread>

namespace hermes {
namespace vm {

SharedGC::SharedGC(size_t maxSize) : maxSize_(maxSize) {
  // Checked once here rather than on every region: vm_allocate_aligned asserts
  // both the size and the alignment against the OS page size, so a segment
  // smaller than a page cannot be mapped at all on a 64 KiB-page kernel. In
  // release the mapping succeeds but the kernel commits a whole page per
  // segment, which would make footprint() understate real usage several-fold.
  assert(
      kSegmentSize % oscompat::page_size() == 0 &&
      "kSegmentSize must be a multiple of OS pages, or vm_allocate_aligned "
      "cannot honour the request");
}

SharedGC::~SharedGC() {
  // Releasing segments here would free cells another Runtime may still name, so
  // a non-empty heap at teardown is the group losing track of live cells. The
  // release below is what happens once that assert is compiled out.
  assert(
      footprint_.load(std::memory_order_relaxed) == 0 &&
      "every shared cell must be swept before the group is torn down");
  // A surviving registration means a Runtime outlived the heap it attached to.
  // Its ~SharedMutator would lock a freed mutex and write a freed vector, and
  // ASan would report the mutex rather than the ownership bug.
  assert(mutators_.empty() && "a SharedMutator outlived its SharedGC");

  auto releaseAll = [this](SegmentList &list) {
    while (!list.empty()) {
      SegmentHeader &hdr = list.front();
      list.pop_front();
      releaseRegion(segmentOf(&hdr));
    }
  };
  for (auto &pg : segments_)
    releaseAll(pg.all);
  releaseAll(largeRegions_);
}

SharedGC::Segment *SharedGC::createRegion(
    uint32_t numSegments,
    uint32_t stride,
    uint32_t numCells) {
  assert(numCells >= 1 && "a region must hold at least one cell");
  assert(numCells <= kAllocBits && "a region cannot outgrow its alloc bitmap");
  // markWords covers one segment, so a region may only place cells within it.
  // Bucketed regions are one segment; an oversized region is larger but holds
  // exactly one cell. Anything else would index past markWords into allocBits.
  assert(
      firstGranule() + size_t{numCells - 1} * (stride >> LogHeapAlign) <
          kMarkBits &&
      "the last cell's granule falls outside the mark bitmap");

  const uint64_t bytes64 = (uint64_t)numSegments * kSegmentSize;
  if (LLVM_UNLIKELY(bytes64 > maxSize_))
    return nullptr;
  const size_t bytes = static_cast<size_t>(bytes64);

  size_t cur = footprint_.load(std::memory_order_relaxed);
  do {
    if (LLVM_UNLIKELY(bytes > maxSize_ - cur))
      return nullptr;
  } while (!footprint_.compare_exchange_weak(
      cur, cur + bytes, std::memory_order_relaxed, std::memory_order_relaxed));

  auto res = oscompat::vm_allocate_aligned(bytes, kSegmentSize);
  if (LLVM_UNLIKELY(!res)) {
    // Restore the footprint.
    footprint_.fetch_sub(bytes, std::memory_order_relaxed);
    return nullptr;
  }
  oscompat::vm_name(*res, bytes, "hermes-shared");

  // Value-initialised, which zeroes the mark words and runs BitArray's
  // constructor over the alloc bits, so neither bitmap relies on the mapping
  // arriving zeroed.
  auto *seg = new (*res) Segment();
  SegmentHeader *hdr = &seg->header;
  hdr->numSegments = numSegments;
  hdr->stride = stride;
  hdr->numCells = numCells;
  // Only worth doing where a slot may sit unclaimed: it catches an overflow
  // out of one cell into a neighbour that has never been handed out. An
  // oversized region is one cell that finishAlloc unpoisons immediately, so
  // poisoning it here is unncessary.
  if (numCells > 1) {
    __asan_poison_memory_region(seg->allocRegion, bytes - kPayloadOffset);
  }
  return seg;
}

void SharedGC::releaseRegion(Segment *seg) {
  const size_t bytes = size_t{seg->header.numSegments} * kSegmentSize;
  seg->~Segment();
  __asan_unpoison_memory_region(seg, bytes);
  oscompat::vm_free_aligned(seg, bytes);
  footprint_.fetch_sub(bytes, std::memory_order_relaxed);
}

void *SharedGC::finishAlloc(SegmentHeader *hdr, uint32_t i) {
  void *mem = segmentOf(hdr)->allocRegion + size_t{i} * hdr->stride;
  __asan_unpoison_memory_region(mem, hdr->stride);

  // Nothing is constructed here.
  hdr->allocBits.set(i, true);
  ++hdr->liveCells;
  hdr->nextFreeHint = i + 1;

  // Black allocation, and unconditional for a reason stronger than "harmless".
  // A cell born between certification and the following sweep must be marked
  // or that sweep frees it, so gating this on markingOn() is wrong even though
  // it looks like an obvious saving. Within a cycle the mark is what makes a
  // cell the collector cannot yet reach indistinguishable from one it has
  // already marked.
  setCellMarkBit(static_cast<SharedGCCell *>(mem));
  return mem;
}

void *SharedGC::allocSmallLocked(uint32_t cellSize) {
  const uint32_t bucket = SizeBuckets::indexOf(cellSize);
  Segments &pg = segments_[bucket];

  // Step over segments that filled earlier this cycle, and over the one the
  // sweeper owns. Nothing frees between sweeps, so a segment passed here
  // cannot have become allocatable again. beingSwept is tested first: the
  // sweeper is rewriting that segment's liveCells without mtx_.
  auto skipUnusable = [&pg] {
    while (
        pg.cursor != pg.all.end() &&
        (pg.cursor->beingSwept || pg.cursor->liveCells == pg.cursor->numCells))
      ++pg.cursor;
  };
  skipUnusable();
  // A sweep of this list rewinds the cursor only once it is done, so meanwhile
  // the segments it has freed slots in can be behind the cursor. At the budget,
  // mapping another segment instead would be refused.
  if (LLVM_UNLIKELY(pg.cursor == pg.all.end() && listBeingSwept_ == &pg.all)) {
    pg.cursor = pg.all.begin();
    skipUnusable();
  }

  if (LLVM_UNLIKELY(pg.cursor == pg.all.end())) {
    const uint32_t stride = SizeBuckets::strideOf(bucket);
    Segment *fresh = createRegion(
        1,
        stride,
        static_cast<uint32_t>((kSegmentSize - kPayloadOffset) / stride));
    if (LLVM_UNLIKELY(!fresh))
      return nullptr;
    pg.cursor = pg.all.insert(pg.all.end(), fresh->header);
  }

  // Every slot below the hint is occupied -- nothing frees between sweeps --
  // and every bit at or above numCells is clear, so the search stops at
  // numCells when the segment is full, which the cursor loop above has ruled
  // out.
  SegmentHeader &hdr = *pg.cursor;
  const uint32_t i = static_cast<uint32_t>(
      hdr.allocBits.findNextZeroBitFrom(hdr.nextFreeHint));
  assert(i < hdr.numCells && "the cursor segment must have a free slot");
  return finishAlloc(&hdr, i);
}

void *SharedGC::allocLargeLocked(uint32_t cellSize) {
  // Using uint64_t, otherwise it would overflow on a 32-bit platform.
  const uint64_t bytes = uint64_t{kPayloadOffset} + cellSize;
  const uint32_t numSegments =
      static_cast<uint32_t>((bytes + kSegmentSize - 1) / kSegmentSize);
  // The whole payload is charged to the one cell, as JumboHeapSegment does.
  const uint64_t stride = uint64_t{numSegments} * kSegmentSize - kPayloadOffset;
  assert(
      stride <= std::numeric_limits<uint32_t>::max() &&
      "stride exceeds the width of SegmentHeader::stride");

  Segment *seg =
      createRegion(numSegments, static_cast<uint32_t>(stride), /*numCells*/ 1);
  if (LLVM_UNLIKELY(!seg))
    return nullptr;

  void *mem = finishAlloc(&seg->header, 0);
  largeRegions_.push_back(seg->header);
  return mem;
}

void *SharedGC::allocLocked(uint32_t cellSize) {
  assert(
      isSizeHeapAligned(cellSize) && "callers must round up before allocating");
  assert(cellSize >= sizeof(SharedGCCell) && "a cell must hold its own header");
  assert(cellSize <= maxCellSize() && "a cell must not exceed maxCellSize()");

  return cellSize <= SizeBuckets::kMaxBucketSize ? allocSmallLocked(cellSize)
                                                 : allocLargeLocked(cellSize);
}

bool SharedGC::sweepRegion(SegmentHeader *hdr) {
  // beingSwept keeps allocation off this region, so the sweeper is its only
  // reader and writer and every operation below is plain.
  const size_t strideGranules = hdr->stride >> LogHeapAlign;
  uint32_t live = 0;

  // Skip to the allocated slots rather than testing every one: a segment that
  // decayed to a few survivors costs a word scan per free run instead of a
  // pass per slot. Clearing bit i below cannot disturb the walk, which resumes
  // at i + 1.
  for (uint32_t i = hdr->allocBits.findNextSetBitFrom(0); i < hdr->numCells;
       i = hdr->allocBits.findNextSetBitFrom(i + 1)) {
    const size_t g = firstGranule() + size_t{i} * strideGranules;
    std::atomic<uint8_t> &mw = hdr->markWords[g / kBitsPerWord];
    const uint8_t markMask = static_cast<uint8_t>(1u << (g % kBitsPerWord));

    // One load: beingSwept makes the sweeper the only writer, so the word
    // cannot change under it.
    const uint8_t word = mw.load(std::memory_order_relaxed);
    if (word & markMask) {
      // Clearing survivors here is half of the mark reset. beginCycle resets
      // the whole heap, so this is what makes two sweeps in a row free
      // everything, not what makes beginCycle cheap.
      mw.store(
          static_cast<uint8_t>(word & ~markMask), std::memory_order_relaxed);
      ++live;
    } else {
      hdr->allocBits.set(i, false);
    }
  }

  hdr->liveCells = live;
  hdr->nextFreeHint = 0;
  if (live == 0)
    return true;

#if LLVM_ADDRESS_SANITIZER_BUILD
  // Deferred to here, and skipped entirely for a region that just emptied:
  // releaseRegion unpoisons the whole region before unmapping it, so
  // poisoning a
  // dying region is undone immediately -- for an oversized cell that is a
  // pass
  // over megabytes of shadow for nothing. Poison is only worth writing for a
  // slot that will be handed out again. Idempotent for slots createRegion
  // already poisoned and never handed out.
  for (uint32_t i = 0; i < hdr->numCells; ++i) {
    if (hdr->allocBits.at(i))
      continue;
    __asan_poison_memory_region(
        segmentOf(hdr)->allocRegion + size_t{i} * hdr->stride, hdr->stride);
  }
#endif
  return false;
}

size_t SharedGC::sweep() {
  // Loaded outside the assert on purpose, since an assert is compiled out.
  // Marks are set with relaxed ordering and markers never take mtx_. On the
  // thread that marked, program order makes them visible here; a sweep on
  // another thread sees them only if this acquire reads the store that
  // turned marking off.
  const bool marking = markingOn();
  (void)marking;
  assert(
      !marking &&
      "sweeping clears survivors' marks, so no marker may be running");
#ifndef NDEBUG
  {
    // Not implied by the check above any more. certifyRound turns marking off
    // inside its own critical section and leaves markState_ alive until
    // endCycle, so a driver that swept in that window would pass !markingOn(),
    // free cells the worklists still name, and unmap their segments
    // underneath them.
    std::lock_guard<std::mutex> lk(barrierMtx_);
    assert(!markState_ && "sweeping inside a marking cycle");
    // After abortCycle the observable state is identical to after endCycle --
    // no MarkState, marking is off -- so nothing else here can tell the two
    // apart, and a sweep on a partial trace frees reachable cells.
    assert(
        !marksArePartial_ &&
        "sweeping after an abandoned cycle; begin another cycle instead");
  }
#endif
  if (LLVM_UNLIKELY(marksArePartial_))
    return 0;
#ifndef NDEBUG
  assert(
      !sweeping_.exchange(true, std::memory_order_relaxed) &&
      "sweep() admits one caller at a time");
#endif
  size_t released = 0;

  // One segment at a time, not the whole class. The sweeper has to be a
  // segment's only writer. Flagging one segment leaves every other one
  // allocatable.
  auto sweepList = [this, &released](
                       SegmentList &list, SegmentList::iterator *cursor) {
    std::unique_lock<std::mutex> lk(mtx_);
    // Most buckets are unused in any given heap, and skipping them here is
    // what keeps a tight sweep loop off mtx_ entirely.
    if (list.empty())
      return;
    listBeingSwept_ = &list;
    // The segment that is last when the walk begins. Allocation only appends,
    // so stopping here is what keeps a segment mapped mid-sweep out of this
    // pass.
    const SegmentList::iterator last = std::prev(list.end());

    // The iterators survive the unlock below: allocation only appends, which
    // invalidates nothing on an intrusive list, and the sweeper is the only
    // caller that erases.
    for (auto it = list.begin();;) {
      SegmentHeader &hdr = *it;
      // Read before the erase below, which is the one thing that can leave
      // `last` dangling, and only when the tail is what emptied.
      const bool atLast = it == last;
      hdr.beingSwept = true;
      lk.unlock();

      const bool emptied = sweepRegion(&hdr);

      lk.lock();
      hdr.beingSwept = false;
      if (!emptied) {
        ++it;
      } else {
        const bool cursorHere = cursor && *cursor == it;
        it = list.erase(it);
        // Update the cursor to be valid so that when mtx_ is unlocked below,
        // allocator can still access it correctly.
        if (cursorHere)
          *cursor = it;
        released += size_t{hdr.numSegments} * kSegmentSize;
        // Returned to the OS rather than pooled. This is the only place we can
        // shrink the heap without compaction.
        lk.unlock();
        releaseRegion(segmentOf(&hdr));
        lk.lock();
      }
      if (LLVM_UNLIKELY(sweepHook_)) {
        lk.unlock();
        (*sweepHook_)();
        lk.lock();
      }
      // Already at the last segment (before starting the sweeping), exit.
      if (atLast)
        break;
    }

    // Rewound because this sweep just falsified the cursor's premise that
    // everything behind it is full. So reset it to the beginning so that freed
    // space can be reused.
    if (cursor)
      *cursor = list.begin();
    listBeingSwept_ = nullptr;
  };

  for (uint32_t b = 0; b < SizeBuckets::kNumBuckets; ++b)
    sweepList(segments_[b].all, &segments_[b].cursor);
  sweepList(largeRegions_, nullptr);

#ifndef NDEBUG
  sweeping_.store(false, std::memory_order_relaxed);
#endif
  return released;
}

/// Internal linkage records that this file is the only user, and lets the
/// compiler inline accept() into the drain loop below.
class HERMES_ATTRIBUTE_INTERNAL_LINKAGE SharedGC::MarkAcceptor final {
 public:
  explicit MarkAcceptor(SharedGC &gc) : gc_(gc) {}

  /// Mark and enqueue \p cell unless it is already marked. Every source -- a
  /// slot from SharedSlotVisitor, a root, an entry off a barrier ring --
  /// arrives here.
  /// \return true if this call marked \p cell.
  bool accept(SharedGCCell *cell) {
    // A load then an unconditional set, never a test-and-set: this thread is
    // the only marker, and black allocation only ever sets a different cell's
    // bit.
    //
    // A cell born during the cycle is already marked and stops here. It is
    // never traced, so its outgoing edges rest on every shared reference
    // having been pushed before it could be stored.
    if (SharedGC::isMarked(cell))
      return false;
    SharedGC::setCellMarkBit(cell);
    gc_.markState_->localWorklist.push(cell);
    return true;
  }

 private:
  SharedGC &gc_;
};

void SharedGC::beginCycle() {
  {
    // Only the emplace needs this. spill() decides "is there still a cycle" by
    // testing markState_ under this lock and then dereferencing it, so the
    // existence transition has to be serialised against it. Nothing below
    // touches markState_, and keeping the scope tight matters as soon as
    // anything expensive joins this function.
    std::lock_guard<std::mutex> lk(barrierMtx_);
    assert(!markState_ && "a marking cycle is already in progress");
    markState_.emplace();
  }

  // Even to odd: turning marking on. A barrier that reads an odd counter
  // lock-free only stamps an integer with it, and it reaches the worklist only
  // through barrierMtx_, whose acquire pairs with the release above -- so it
  // cannot see a half-built markState_, and the two statements need no order
  // between them.
  //
  // The window this leaves -- markState_ alive, marking still off -- is not a
  // new state: certifyRound turns marking off inside its own critical section
  // and leaves markState_ alive until endCycle, which is why endCycle asserts
  // that pair. A barrier preempted across the previous cycle's end can reach
  // spill() in it; it finds an even cycleSeq(), so drainInto's stamp test drops
  // every entry without reading a cell -- the same outcome as the discardAll()
  // branch it would have taken a moment earlier.
  //
  // The bulk reset and turning marking on, together, under the allocation lock.
  //
  // Cells are born marked whether or not a cycle is running, and only a sweep
  // clears a mark, so the heap reaching here can hold marks left by allocation
  // since the last sweep -- or by a cycle that was abandoned. A mark means both
  // "reachable" and "already enqueued, do not trace its children", and the
  // second is only true within one cycle: leaving one behind makes accept()
  // refuse to enqueue that cell, so its children are never traced and the
  // sweep frees them while they are still reachable.
  //
  // Both statements are under mtx_ so nothing can land between them. A cell
  // born between the two would keep its birth mark and not be traced this
  // cycle, so a value stored into it before marking turned on, by a runtime
  // that then dropped its own key for that value, would have no record.
  //
  // A birth mark lasts only until the next cycle starts, so a cell held only
  // as a raw pointer on a C++ stack across a cycle boundary is not covered by
  // it. Not reachable while only tests start a cycle.
  std::lock_guard<std::mutex> allocLk(mtx_);
  whitenAllLocked();
  cycleSeq_.fetch_add(1, std::memory_order_seq_cst);
  assert(markingOn() && "beginCycle must leave marking on");
}

void SharedGC::endCycle() {
  std::lock_guard<std::mutex> lk(barrierMtx_);
  assert(markState_ && "no marking cycle in progress");
  // Certification turns marking off inside its own critical section, so
  // marking still being on here means the cycle never certified -- which is
  // abortCycle's job, because the caller must then not sweep.
  assert(
      !markingOn() &&
      "this cycle never certified; abandon it with abortCycle instead");
  // Repaired in release as well, not just asserted. cycleSeq_'s parity is what
  // every barrier reads, so getting here with marking on and leaving it that
  // way inverts the parity for good: during the next cycle every barrier sees
  // marking off and records nothing, the cycle certifies immediately, and its
  // sweep frees whatever was stored during it.
  if (LLVM_UNLIKELY(markingOn())) {
    cycleSeq_.fetch_add(1, std::memory_order_seq_cst);
    // And its marks are a partial trace, which no sweep may act on.
    marksArePartial_ = true;
  }
  assert(markState_->localWorklist.empty() && "cells were left untraced");
  // The barrier worklist is not bookkeeping either: what is in it are cells
  // that must be marked before the cycle may end, with the same consequence as
  // localWorklist if dropped -- an unmarked live cell, swept. Certification
  // drains it, so reaching here non-empty is a driver bug.
  assert(
      markState_->barrierWorklist.empty() &&
      "spilled cells were left untraced");
  markState_.reset();
}

void SharedGC::abortCycle() {
  std::lock_guard<std::mutex> lk(barrierMtx_);
  assert(markState_ && "no marking cycle in progress");
  assert(markingOn() && "this cycle certified; finish it with endCycle");
  // Guarded, not unconditional: abandoning a cycle that already certified
  // would otherwise turn marking back on with no MarkState behind it, so
  // every mutator would take the slow path and push into rings that spill()
  // throws away.
  if (LLVM_LIKELY(markingOn()))
    cycleSeq_.fetch_add(1, std::memory_order_seq_cst);
  // Neither worklist has to be drained, unlike endCycle. Their contents are
  // records for a cycle that will never certify.
  markState_.reset();
  // The marks this cycle set are a partial trace: unmarked no longer implies
  // unreachable, so sweeping now would free live cells. beginCycle's reset is
  // what clears them.
  marksArePartial_ = true;
}

void SharedGC::whitenAllLocked() {
#ifndef NDEBUG
  assert(
      !sweeping_.load(std::memory_order_relaxed) &&
      "whitening races the sweeper, which reads and writes mark words without "
      "mtx_ once beingSwept is set");
#endif
  marksArePartial_ = false;
  auto clear = [](SegmentList &list) {
    for (SegmentHeader &hdr : list) {
      // memset, not a loop of relaxed stores. By this function's precondition
      // there is no concurrent accessor, so the atomicity buys nothing here
      // and costs about nineteen times the time -- and this pass runs under
      // the allocation lock, where every allocating mutator is waiting. The
      // words are lock-free atomics, asserted where they are declared, so
      // writing them as bytes is sound while we are their only accessor.
      static_assert(
          sizeof(hdr.markWords) == kMarkWords * sizeof(std::atomic<uint8_t>),
          "mark words must be contiguous for a bulk clear");
      std::memset(static_cast<void *>(hdr.markWords), 0, sizeof(hdr.markWords));
    }
  };
  for (auto &pg : segments_)
    clear(pg.all);
  clear(largeRegions_);
}

void SharedGC::markRoots(llvh::ArrayRef<SharedGCCell *> roots) {
  assert(markState_ && "no marking cycle in progress");
  MarkAcceptor acceptor(*this);
  for (SharedGCCell *root : roots)
    acceptor.accept(root);
}

std::vector<SharedGCCell *> SharedGC::copyRootKeys(
    const std::function<void(const Runtime &)> *beforeCopy) const {
  std::vector<SharedMutator *> registered;
  {
    std::lock_guard<std::mutex> lk(barrierMtx_);
    registered = mutators_;
  }

  std::vector<SharedGCCell *> keys;
  std::vector<SharedGCCell *> mapKeys;
  for (SharedMutator *mutator : registered) {
    const Runtime *runtime = nullptr;
    // Outside the loop, so that it is still held for the copy.
    std::unique_lock<std::mutex> mapLock;
    while (true) {
      std::unique_lock<std::mutex> lk(barrierMtx_);
      // Unregistered since the snapshot: its keys are no longer roots, and
      // its map may be gone.
      if (std::find(mutators_.begin(), mutators_.end(), mutator) ==
          mutators_.end())
        break;
      mapLock = std::unique_lock<std::mutex>(
          mutator->runtime().sharedObjectBoxesLock(), std::try_to_lock);
      if (mapLock.owns_lock()) {
        runtime = &mutator->runtime();
        break;
      }
      // Busy. Waiting for the map here would hold up every spill for as long
      // as the map is held; waiting without barrierMtx_ would let the runtime
      // unregister, and its map go, before the wait ends.
      lk.unlock();
      std::this_thread::yield();
    }
    if (!runtime)
      continue;
    if (beforeCopy)
      (*beforeCopy)(*runtime);
    // Under the map lock alone, and into a vector of its own, so the hold is
    // a copy of this map and nothing more: an insert waits for it holding
    // mtx_.
    mapKeys.clear();
    runtime->appendSharedRootKeysUnlocked(mapKeys);
    mapLock.unlock();
    keys.insert(keys.end(), mapKeys.begin(), mapKeys.end());
  }
  return keys;
}

size_t SharedGC::markRuntimeRootsImpl(
    const std::function<void(const Runtime &)> *beforeCopy) {
  const std::vector<SharedGCCell *> keys = copyRootKeys(beforeCopy);
  // No lock is held. accept() writes the collector-local worklist, which no
  // mutator touches.
  markRoots(keys);
  return keys.size();
}

size_t SharedGC::drainMarkWorklist() {
  assert(markState_ && "no marking cycle in progress");
  MarkAcceptor acceptor(*this);
  auto &localWorklist = markState_->localWorklist;
  size_t traced = 0;
  while (!localWorklist.empty()) {
    // Pop before visiting, not after: markCell pushes onto this same stack,
    // so a later pop would discard a newly found cell instead of this one.
    SharedGCCell *cell = localWorklist.top();
    localWorklist.pop();
    markCell(acceptor, cell);
    ++traced;
  }
  return traced;
}

void SharedGC::registerMutator(SharedMutator *mutator) {
  std::lock_guard<std::mutex> lk(barrierMtx_);
  assert(
      std::find(mutators_.begin(), mutators_.end(), mutator) ==
          mutators_.end() &&
      "mutator registered twice");
  mutators_.push_back(mutator);
}

void SharedGC::unregisterMutator(SharedMutator *mutator) {
  unlinkMutator(mutator);
  // A root scan that found this runtime registered may still be copying its
  // keys, holding the map lock. Taking that lock waits for the copy, so the
  // map outlives it; once unlinked, no later scan can find the runtime.
  std::lock_guard<std::mutex> lk(mutator->runtime().sharedObjectBoxesLock());
}

void SharedGC::unlinkMutator(SharedMutator *mutator) {
  std::lock_guard<std::mutex> lk(barrierMtx_);
  // Flush before dropping the registration. What is still in the ring are SATB
  // entries, and dropping them can leave a cell unmarked that another runtime
  // still reaches -- sweep would then free it.
  if (markState_)
    mutator->ring_.drainInto(markState_->barrierWorklist, cycleSeq());
  auto it = std::find(mutators_.begin(), mutators_.end(), mutator);
  assert(it != mutators_.end() && "unregistering an unknown mutator");
  // Checked in release too. A double unregister, or a SharedMutator built
  // against a different SharedGC, would otherwise write through end() and
  // shrink -- heap corruption in exactly the build with no assert to catch it.
  if (LLVM_UNLIKELY(it == mutators_.end()))
    return;
  // Order does not matter: the only walk is a steal, which visits every entry.
  *it = mutators_.back();
  mutators_.pop_back();
}

void SharedGC::spill(SharedMutator &mutator) {
  std::lock_guard<std::mutex> lk(barrierMtx_);
  if (!markState_) {
    // A barrier preempted across the end of a cycle lands here. Everything in
    // the ring belongs to a cycle that has ended, so the cells may already
    // be swept and their segments unmapped: consume without reading.
    mutator.ring_.discardAll();
    return;
  }
  mutator.ring_.drainInto(markState_->barrierWorklist, cycleSeq());
}

size_t SharedGC::stealBarrierEntries() {
  std::lock_guard<std::mutex> lk(barrierMtx_);
  assert(markState_ && "no marking cycle in progress");
  // Holding the lock across the walk is what keeps a Runtime from being
  // destroyed mid-walk, and what makes this thread each ring's only consumer.
  std::vector<SharedGCCell *> &barrierWorklist = markState_->barrierWorklist;
  const size_t before = barrierWorklist.size();
  for (SharedMutator *mutator : mutators_)
    mutator->ring_.drainInto(barrierWorklist, cycleSeq());
  return barrierWorklist.size() - before;
}

size_t SharedGC::drainBarrierWorklist() {
  std::vector<SharedGCCell *> taken;
  {
    std::lock_guard<std::mutex> lk(barrierMtx_);
    assert(markState_ && "no marking cycle in progress");
    taken.swap(markState_->barrierWorklist);
    // Swapping hands the capacity to `taken`, so give it back rather than let
    // the next round of spills rebuild it one reallocation at a time under
    // this lock, where a spilling mutator is waiting.
    markState_->barrierWorklist.reserve(taken.size());
  }
  // Outside the lock on purpose: marking is unbounded work, and a mutator
  // whose ring fills must not wait on it.
  MarkAcceptor acceptor(*this);
  size_t newlyMarked = 0;
  for (SharedGCCell *cell : taken)
    newlyMarked += acceptor.accept(cell);
  return newlyMarked;
}

bool SharedGC::certifyRound() {
  assert(markState_ && "no marking cycle in progress");
  assert(markingOn() && "this cycle has already certified");
  assert(
      markState_->localWorklist.empty() &&
      "drain the local worklist before certifying, or a cell it already holds "
      "counts as new");
  // Mark the backlog first, outside the lock. Everything spilled during the
  // whole tracing phase has piled up in the worklist, and marking it under
  // barrierMtx_ would stall every mutator whose ring fills -- that is, every
  // mutator doing a store -- for as long as it takes. Doing it here costs
  // atomicity nothing: this only removes work, and anything arriving after it
  // is consumed in the critical section below.
  if (drainBarrierWorklist() != 0)
    return false;

  std::lock_guard<std::mutex> lk(barrierMtx_);
  // No fence is needed before the drains. The trace read every slot with an
  // acquire load, a barrier's slot store releases the push made before it,
  // and the next writer's pre-read acquires that store in turn. So every push
  // made before the value the trace read, along the chain of writers to that
  // slot, is already visible here. The lock's acquire could not stand in for
  // the trace's: a barrier does not take this lock on its fast path, so there
  // is no release of it to pair with.
  //
  // Rings drain into the same worklist, so the two halves are one observation
  // rather than two checks a spill could slip between.
  std::vector<SharedGCCell *> &worklist = markState_->barrierWorklist;
  // Hoisted: this thread is the only writer of cycleSeq_ and it holds the lock.
  const uint32_t seq = cycleSeq();
  for (SharedMutator *mutator : mutators_)
    mutator->ring_.drainInto(worklist, seq);

  MarkAcceptor acceptor(*this);
  size_t newlyMarked = 0;
  for (SharedGCCell *cell : worklist)
    newlyMarked += acceptor.accept(cell);
  worklist.clear();
  if (newlyMarked != 0)
    return false;

  // Turn marking off before releasing the lock. Outside it, a spill could land
  // between the last drain above and turning marking off, and nothing would
  // ever look at it.
  cycleSeq_.fetch_add(1, std::memory_order_seq_cst);
  return true;
}

} // namespace vm
} // namespace hermes
