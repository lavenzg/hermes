/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/ADT/BitArray.h"
#include "hermes/VM/HeapAlign.h"
#include "hermes/VM/SharedHeap/SharedGCCell.h"

#include "llvh/ADT/ilist_node.h"
#include "llvh/ADT/simple_ilist.h"
#include "llvh/Support/MathExtras.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>

namespace hermes {
namespace vm {

/// A heap shared by several Runtimes on several threads, holding cells that
/// outlive any one Runtime.
///
/// Storage is a set of self-aligned regions, so masking a cell pointer reaches
/// the region header that owns it. Cells carry no GC metadata beyond their
/// CellKind: mark bits live in the region header, and a cell's size comes
/// either from its size bucket or, for an oversized cell, from its region.
/// Besides its kind, a cell costs two bitmap bits, plus a share of the header's
/// fixed size that grows with the stride.
///
/// **Every cell here is a SharedGCCell, never a GCCell.** That is not a naming
/// choice: a SharedGCCell* does not convert to a GCCell*, so no shared cell can
/// interact with GCPointer, HermesValue or a local write barrier, and the
/// masking sites those lead to cannot be handed an address whose segment is not
/// a Hades segment.
///
/// Cells in this heap never move. Only dead cells will be swept. So there
/// should never be a stale SharedGCCell pointer reached unless memory is
/// corrupted.
class SharedGC {
 public:
  //===--------------------------------------------------------------------===//
  // Segment geometry

  /// Every region of segments is a multiple of this and aligned to it, so
  /// `addr & ~(kSegmentSize - 1)` reaches the header from any cell pointer.
  static constexpr size_t kLogSegmentSize = 14;
  static constexpr size_t kSegmentSize = size_t{1} << kLogSegmentSize;

  /// The shared heap's size buckets.
  ///
  /// Two regions. Up to kPreciseCutoff, buckets step by HeapAlign, so a 40-byte
  /// cell occupies exactly 40 bytes. Above it, each doubling (2^k, 2^(k+1)] is
  /// cut into kNumGroup evenly spaced buckets, so the *first* step of a group
  /// is x1.25 and the rest are progressively finer. That bounds rounding waste
  /// at 1/(kNumGroup + 1). This is very similar to the buckets in the OG of
  /// HadesGC, but with smaller precise cutoff and finer progressing in large
  /// buckets.
  ///
  /// Note that for large buckets (within a standard segment), it may waste the
  /// space in the tail of the segment. For example, a 8192 bytes cell would
  /// occupy the entire segment but the second half is almost unused. We may
  /// need to tweak the buckets to optimize the space sage.
  struct SizeBuckets {
    /// Cells above this get a region of segments to themselves.
    static constexpr uint32_t kLogMaxBucketSize = 13;
    static constexpr uint32_t kMaxBucketSize = 1u << kLogMaxBucketSize;
    /// Below this, buckets step by HeapAlign; above, in groups.
    static constexpr uint32_t kLogPreciseCutoff = 7;
    static constexpr uint32_t kPreciseCutoff = 1u << kLogPreciseCutoff;
    /// Buckets per doubling above the cutoff.
    static constexpr uint32_t kLogNumGroup = 2;
    static constexpr uint32_t kNumGroup = 1u << kLogNumGroup;

    static constexpr uint32_t kNumPreciseBuckets =
        kPreciseCutoff >> LogHeapAlign;
    static constexpr uint32_t kNumBuckets = kNumPreciseBuckets +
        (kLogMaxBucketSize - kLogPreciseCutoff) * kNumGroup;

    // Every bucket must be heap-aligned, which for the group region means every
    // delta must be. The smallest is the first group's, 2^kLogPreciseCutoff /
    // kNumGroup.
    static_assert(
        kLogPreciseCutoff >= kLogNumGroup + LogHeapAlign,
        "the first group's delta would fall below HeapAlign");

    /// Allocation size of bucket \p bucket.
    static constexpr uint32_t strideOf(uint32_t bucket) {
      if (bucket < kNumPreciseBuckets)
        return (bucket + 1) << LogHeapAlign;
      const uint32_t j = bucket - kNumPreciseBuckets;
      // The group covers (2^k, 2^(k+1)] in kNumGroup steps of 2^k / kNumGroup,
      // so bucket `offset` of it is (kNumGroup + offset + 1) such steps.
      const uint32_t k = kLogPreciseCutoff + (j >> kLogNumGroup);
      const uint32_t offset = j & (kNumGroup - 1);
      return (kNumGroup + offset + 1) << (k - kLogNumGroup);
    }

    /// Smallest bucket whose stride covers \p size. Below kPreciseCutoff the
    /// buckets step by HeapAlign, so the stride is the size exactly.
    /// \pre 0 < \p size <= kMaxBucketSize.
    /// \pre isSizeHeapAligned(size) is true.
    static uint32_t indexOf(uint32_t size) {
      assert(isSizeHeapAligned(size) && "size must be aligned to HeapAlign");
      assert(size > 0 && size <= kMaxBucketSize && "size has no bucket");
      if (size <= kPreciseCutoff)
        return (size - 1) >> LogHeapAlign;
      // `size - 1` rather than `size` so that a size sitting exactly on 2^(k+1)
      // is the *last* bucket of group k rather than the first of group k+1.
      const uint32_t k = llvh::Log2_32(size - 1);
      // Shifting by k - kLogNumGroup keeps kLogNumGroup + 1 bits, the topmost
      // being the leading 1 that k names, so it always lands in
      // [kNumGroup, 2 * kNumGroup).
      const uint32_t offset =
          ((size - 1) >> (k - kLogNumGroup)) & (kNumGroup - 1);
      return kNumPreciseBuckets + (k - kLogPreciseCutoff) * kNumGroup + offset;
    }
  };

 private:
  //===--------------------------------------------------------------------===//
  // Segment layout

  /// Bits in the granule-indexed mark bitmap, covering one segment.
  /// This is enough for both standard segment and oversized segment (which will
  /// only have one cell).
  static constexpr size_t kMarkBits = kSegmentSize >> LogHeapAlign;
  /// Cells per segment never exceeds one per granule.
  static constexpr size_t kAllocBits = kMarkBits;
  /// Width of a mark word. A byte rather than a machine word: it is the
  /// narrowest unit std::atomic is lock-free on everywhere -- a 64-bit atomic
  /// is not, on the 32-bit targets Hermes still ships to -- and it keeps a
  /// single word to 64 bytes of heap rather than 512. The alloc bitmap is a
  /// BitArray and picks its own width.
  static constexpr size_t kBitsPerWord = 8;
  static constexpr size_t kMarkWords = kMarkBits / kBitsPerWord;

  /// Header at offset 0 of every segment in shared heap. It has two arrays:
  /// 1. Array for mark bits. It's atomic so that mutator and marker can access
  /// it concurrently.
  /// 2. Array for allocation bits. It's non-atomic since allocation is
  /// protected by \c mtx_.
  class SegmentHeader : public llvh::ilist_node<SegmentHeader> {
   public:
#ifndef NDEBUG
    /// Canary to catch wild pointers and memory corruption.
    static constexpr uint32_t kMagic = 0x5ea5ed9c;
    uint32_t magic{kMagic};
#endif

    /// Segments in this region. Always 1 for a bucketed region; only an
    /// oversized cell spans more than one.
    uint32_t numSegments{1};
    /// Allocation size of each cell here. For an oversized region, the whole
    /// payload.
    uint32_t stride{0};
    /// Number of cell slots in this region.
    uint32_t numCells{0};
    /// Slots currently allocated.
    uint32_t liveCells{0};
    /// Set while the sweeper owns this segment. Allocation skips it: the
    /// sweeper rewrites allocBits and liveCells without mtx_, so it has to be
    /// the only writer.
    bool beingSwept{false};
    /// Where the next allocation scan starts, in cell numbers.
    uint32_t nextFreeHint{0};

    /// Individual bits set set lock-free by the collector. Black allocation and
    /// reset are protected by mtx_.
    std::atomic<uint8_t> markWords[kMarkWords];
    static_assert(
        std::atomic<uint8_t>::is_always_lock_free,
        "a locked mark bitmap would deadlock against the collector");
    /// Written only under the allocation lock, or by the sweeper once
    /// beingSwept has made it single-writer, so a plain BitArray serves. The
    /// mark bitmap above cannot use one: its writers hold no lock.
    BitArray<kAllocBits> allocBits;
  };

  /// The layout of a segment: the header, then the cells.
  ///
  /// AlignedHeapSegment::Contents in miniature. allocRegion states the payload
  /// alignment as a declaration rather than as arithmetic on sizeof, and its
  /// offset is where the cells begin.
  class Segment {
   public:
    SegmentHeader header;
    /// The first byte of the allocation region, which extends past the "end"
    /// of the class to the end of the mapping that contains it.
    alignas(HeapAlign) char allocRegion[1];
  };

  using SegmentList = llvh::simple_ilist<SegmentHeader>;

  /// One list per size bucket, holding every segment of that bucket.
  class Segments {
   public:
    SegmentList all;
    /// First segment not known to be full. Everything before it filled earlier
    /// this cycle; since cells are freed only by sweep, occupancy rises
    /// monotonically between sweeps and this never moves backwards then. The
    /// sweeper works the list in place and rewinds this when it is done; until
    /// then, allocation may move it back to a segment the sweep has freed
    /// slots in.
    SegmentList::iterator cursor{all.end()};
  };

  /// \return the header for \p cell.
  static SegmentHeader *headerOf(const void *cell) {
    auto *seg = reinterpret_cast<Segment *>(
        reinterpret_cast<uintptr_t>(cell) & ~(kSegmentSize - 1));
    assert(
        seg->header.magic == SegmentHeader::kMagic &&
        "cell is not the start of a cell in the shared heap");
    return &seg->header;
  }
  /// \return the granule of \p cell within the region \p hdr heads, which is
  /// its index into the mark bitmap.
  /// \pre \p cell is the *base* of a slot, not an interior pointer.
  static size_t granuleOf(const SegmentHeader *hdr, const void *cell) {
    const size_t g = (reinterpret_cast<uintptr_t>(cell) -
                      reinterpret_cast<uintptr_t>(hdr)) >>
        LogHeapAlign;
    assert(
        g >= firstGranule() && g < kMarkBits && "granule outside the region");
    assert(
        (g - firstGranule()) % (hdr->stride >> LogHeapAlign) == 0 &&
        "not the base of a slot");
    return g;
  }
  /// The segment \p hdr heads.
  static Segment *segmentOf(SegmentHeader *hdr) {
    static_assert(
        offsetof(Segment, header) == 0,
        "headerOf masks to the segment base, so the header must lead it");
    return reinterpret_cast<Segment *>(hdr);
  }

  /// Granule of cell number 0; consecutive cells step by stride >>
  /// LogHeapAlign.
  static constexpr size_t firstGranule() {
    return kPayloadOffset >> LogHeapAlign;
  }

  /// Allocate a slot of \p cellSize bytes in shared heap and mark it black.
  /// \pre cellSize is aligned to HeapAlign.
  /// \pre mtx_ is held, and stays held until the caller has constructed a cell
  ///   into the returned storage. Holding it across construction is what keeps
  ///   a slot that is allocated and marked but not yet a cell invisible to
  ///   anything that walks the heap.
  /// \return raw storage, or nullptr if the budget is exhausted or the mapping
  ///   fails.
  void *allocLocked(uint32_t cellSize);
  void *allocSmallLocked(uint32_t cellSize);
  void *allocLargeLocked(uint32_t cellSize);

  /// Map a region of \p numSegments segments and initialise its header to
  /// hold \p numCells slots of \p stride bytes. Charges the budget.
  /// \return nullptr if the budget is exhausted or the mapping fails.
  Segment *
  createRegion(uint32_t numSegments, uint32_t stride, uint32_t numCells);
  /// Unmap \p seg and refund the budget.
  void releaseRegion(Segment *seg);

  /// Claim slot \p i of \p hdr and return its storage, marked black. Nothing
  /// is constructed into it; makeA's placement-new does that. Caller holds
  /// mtx_.
  void *finishAlloc(SegmentHeader *hdr, uint32_t i);

  /// Sweep one region. \pre hdr->beingSwept is set, which is what makes the
  /// sweeper its only writer. \return true if it should be released.
  bool sweepRegion(SegmentHeader *hdr);

  //===--------------------------------------------------------------------===//
  // State

  /// Guards every allocator-side structure.
  /// Held across cell construction by the typed allocation entry points, so a
  /// shared cell's constructor must not allocate again since this is not
  /// recursive.
  mutable std::mutex mtx_;

  Segments segments_[SizeBuckets::kNumBuckets];
  SegmentList largeRegions_;
  /// The list sweep() is working through, or null. Guarded by mtx_.
  const SegmentList *listBeingSwept_{nullptr};

  const size_t maxSize_;
  std::atomic<size_t> footprint_{0};
  std::atomic<bool> markingBarriers_{false};

#ifndef NDEBUG
  /// Shared across the group: GCBase::nextObjectID is per-Runtime and would
  /// hand the same id to cells in different Runtimes.
  std::atomic<uint32_t> debugAllocationId_{0};
  /// Set for the duration of sweep(), to catch a second concurrent sweeper.
  std::atomic<bool> sweeping_{false};
#endif

 public:
  /// Offset of the first cell in any segment.
  static constexpr size_t kPayloadOffset = offsetof(Segment, allocRegion);

  // A bucketed segment must hold at least one cell. Without this, raising
  // kLogMaxBucketSize past the point where the stride outgrows the payload
  // gives numCells == 0, and only a debug assert in createRegion stands
  // between that and a release build writing a cell past the mapping.
  static_assert(
      SizeBuckets::kMaxBucketSize <= kSegmentSize - kPayloadOffset,
      "the largest size bucket no longer fits in a segment");

  /// The largest allocatable cell.
  static constexpr uint32_t maxCellSize() {
    return llvh::alignTo<HeapAlign>(
        std::numeric_limits<uint32_t>::max() - kSegmentSize + 1);
  }

  /// \param maxSize the group-wide budget in mapped bytes.
  explicit SharedGC(size_t maxSize);
  ~SharedGC();

  SharedGC(const SharedGC &) = delete;
  SharedGC &operator=(const SharedGC &) = delete;

#ifndef NDEBUG
  uint32_t nextDebugAllocationId() {
    return debugAllocationId_.fetch_add(1, std::memory_order_relaxed);
  }
#endif

  /// Free every unmarked cell and clear the mark on every survivor, so that
  /// starting the next cycle costs a single store.
  ///
  /// Allocation may run concurrently; sweeping may not. The sweeper claims a
  /// segment by setting beingSwept and then rewrites its bitmaps without the
  /// lock, so a second sweeper would race the first on the same segment --
  /// beingSwept excludes allocation, not other sweepers.
  ///
  /// \pre markingOn() is false, and this runs on the thread that marked:
  ///   marks are set with relaxed atomics, so program order is what makes
  ///   them visible here. The acquire at the top of sweep() helps a sweep on
  ///   another thread only if it reads the store that turned marking off.
  /// \return mapped bytes returned to the OS.
  size_t sweep();

  //===--------------------------------------------------------------------===//
  // Mark bits. Reachability is not this class's concern: these are the
  // primitives whoever is tracing calls.

  /// Set \p cell's mark bit. Mask and shift, with no load in between.
  ///
  /// Safe before \p cell is constructed -- black allocation marks a slot as it
  /// is claimed -- because nothing here dereferences it; the address alone
  /// picks out the segment and the bit.
  static void setCellMarkBit(const SharedGCCell *cell) {
    SegmentHeader *hdr = headerOf(cell);
    const size_t bit = granuleOf(hdr, cell);
    std::atomic<uint8_t> &w = hdr->markWords[bit / kBitsPerWord];
    // Every writer sets the same bit to 1, so a plain load/store pair is as
    // correct as a fetch_or and cheaper -- but only against other marks. Black
    // allocation can touch a different bit in this word concurrently, so the
    // read-modify-write has to be atomic.
    w.fetch_or(
        static_cast<uint8_t>(1u << (bit % kBitsPerWord)),
        std::memory_order_relaxed);
  }

  static bool isMarked(const SharedGCCell *cell) {
    SegmentHeader *hdr = headerOf(cell);
    const size_t bit = granuleOf(hdr, cell);
    return hdr->markWords[bit / kBitsPerWord].load(std::memory_order_relaxed) >>
        (bit % kBitsPerWord) &
        1;
  }

  /// The space this cell occupies. For a small cell its size is its bucket's
  /// stride. For a large cell it's the entire available allocation region in
  /// its segment.
  static uint32_t allocatedSize(const SharedGCCell *cell) {
    return headerOf(cell)->stride;
  }

  /// Mapped bytes. Unlike a malloc-based heap this is exact, not modelled:
  /// every byte comes from a region we mapped ourselves.
  size_t footprint() const {
    return footprint_.load(std::memory_order_relaxed);
  }
  size_t maxSize() const {
    return maxSize_;
  }

  /// Whether the shared collector is in its marking phase. Allocation does not
  /// read this: newly allocated object is marked by default.
  bool markingOn() const {
    return markingBarriers_.load(std::memory_order_seq_cst);
  }
  void setMarkingOn(bool on) {
    markingBarriers_.store(on, std::memory_order_seq_cst);
  }

#ifdef UNIT_TEST
  /// \return the number of allocated slots. Walks every region.
  size_t liveCellCountSlow() const {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t n = 0;
    for (const auto &pg : segments_)
      for (const SegmentHeader &hdr : pg.all)
        n += hdr.liveCells;
    for (const SegmentHeader &hdr : largeRegions_)
      n += hdr.liveCells;
    return n;
  }

  /// \return the number of mapped regions. Walks every region.
  size_t regionCountSlow() const {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t n = 0;
    for (const auto &pg : segments_)
      n += std::distance(pg.all.begin(), pg.all.end());
    n += std::distance(largeRegions_.begin(), largeRegions_.end());
    return n;
  }
#endif
};

} // namespace vm
} // namespace hermes
