/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "hermes/ADT/BitArray.h"
#include "hermes/Support/OptValue.h"
#include "hermes/VM/HeapAlign.h"
#include "hermes/VM/SharedHeap/SharedGCCell.h"

#include "llvh/ADT/ilist_node.h"
#include "llvh/ADT/simple_ilist.h"
#include "llvh/Support/MathExtras.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

namespace hermes {
namespace vm {

class BoxedSharedObject;
class Runtime;
template <typename T>
class PseudoHandle;

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
  ///
  /// 64 KiB because vm_allocate_aligned requires whole OS pages for both the
  /// size and the alignment, and 64 KiB is the largest page Hermes meets.
  static constexpr size_t kLogSegmentSize = 16;
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
  /// \pre cellSize <= maxCellSize().
  /// \return raw storage, or nullptr if the budget is exhausted or the mapping
  ///   fails.
  void *allocLocked(uint32_t cellSize);
  void *allocSmallLocked(uint32_t cellSize);
  void *allocLargeLocked(uint32_t cellSize);

  /// Round \p size up to HeapAlign and reject it if no region can hold a cell
  /// that big.
  /// \return the rounded size, or None if \p size cannot be served.
  template <typename T>
  static OptValue<uint32_t> checkedAlignedSize(uint32_t size) {
    // On the raw size, not the rounded one -- rounding a too-short size up to
    // sizeof(T) would hide exactly the caller bug this is here to catch.
    assert(size >= sizeof(T) && "size must at least cover the cell itself");

    // 64-bit so that rounding a near-UINT32_MAX size cannot wrap into a small
    // one, which would allocate a cell far too short for the caller.
    const uint64_t aligned = llvh::alignTo<HeapAlign>(size);
    if (LLVM_UNLIKELY(aligned > maxCellSize()))
      return llvh::None;
    return static_cast<uint32_t>(aligned);
  }

  /// The half of makeA that runs with mtx_ already held, so an entry point can
  /// hold the lock across more than the allocation alone.
  /// \pre mtx_ is held, and \p size has been through checkedAlignedSize<T>.
  template <typename T, typename... Args>
  T *makeALocked(uint32_t size, Args &&...args) {
    static_assert(
        std::is_base_of<SharedGCCell, T>::value,
        "T must be a cell in the shared heap");
    static_assert(
        isSharedCellKind(T::getCellKind()),
        "T's CellKind must be in the shared range");
    static_assert(HeapAlign % alignof(T) == 0, "T out-aligns the shared heap");
    // Nothing ever runs a shared cell's destructor: sweepRegion only clears
    // alloc bits, and releaseRegion destroys the Segment, not the cells in it.
    // A T that owned anything would leak silently on every collection.
    static_assert(
        std::is_trivially_destructible<T>::value,
        "a shared cell is never destroyed, so T must not need to be");

    void *mem = allocLocked(size);
    if (LLVM_UNLIKELY(!mem))
      return nullptr;

    T *cell = new (mem) T(std::forward<Args>(args)...);
    cell->setKind(T::getCellKind());
#ifndef NDEBUG
    cell->setDebugAllocationId(nextDebugAllocationId());
#endif
    return cell;
  }

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
  /// Called after each region sweep() sweeps, with mtx_ released. Only the
  /// UNIT_TEST overload of sweep() sets it.
  const std::function<void()> *sweepHook_{nullptr};

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

  //===--------------------------------------------------------------------===//
  // Allocation and collection

#ifdef UNIT_TEST
  /// Allocate \p size bytes and construct a \p T into them with \p args.
  /// Test-only: the cell is in no runtime's box map, the collector's root set,
  /// so nothing keeps it alive. Production code allocates with makeBoxed.
  ///
  /// **mtx_ is held across T's constructor.** That is what keeps a slot which
  /// is allocated and marked, but not yet a cell, invisible to anything that
  /// walks the heap. It constrains T in exchange: its constructor must not
  /// allocate, take a lock, or trigger allocation in Runtime's heap, which may
  /// deadlock. \p size is rounded up to HeapAlign before allocation.
  /// \return nullptr if the budget is exhausted, the mapping fails, or \p size
  ///   exceeds maxCellSize().
  template <typename T, typename... Args>
  T *makeA(uint32_t size, Args &&...args) {
    const OptValue<uint32_t> alignedSize = checkedAlignedSize<T>(size);
    if (LLVM_UNLIKELY(!alignedSize))
      return nullptr;

    std::lock_guard<std::mutex> lk(mtx_);
    return makeALocked<T>(*alignedSize, std::forward<Args>(args)...);
  }
#endif

  /// Allocate a shared cell and its box together, publishing the pair into
  /// \p runtime's map before the allocation lock is released.
  ///
  /// The publish has to be inside that lock. A collection's bulk reset takes
  /// the same one, so holding it from the allocation through the insert is what
  /// makes a collection either see the cell in the map -- the root set it marks
  /// -- or not have started yet. Outside it, the cell exists for a window in no
  /// map at all: the reset whitens it, the root scan misses it, and it is swept
  /// while its runtime is one instruction from publishing it.
  ///
  /// The box is allocated **first**, deliberately. It is the only step here
  /// that can trigger a local GC, so doing it before the lock is what leaves
  /// the critical section free of anything that allocates. T's constructor
  /// also runs under the lock, with the constraints listed on makeA.
  ///
  /// Between the box and the lock, \p runtime's map is pruned, as a lookup
  /// would: a runtime that only ever allocates would otherwise keep every key
  /// whose box has died, and with it the cell.
  ///
  /// \pre \p runtime belongs to the calling thread. This is the one entry
  ///   point here that touches per-runtime state: a local-heap allocation and a
  ///   map that is not thread-safe.
  /// \return the box, or an empty PseudoHandle if \p size exceeds
  ///   maxCellSize(), the budget is exhausted, or the mapping fails. The box
  ///   allocated for a failed attempt is simply garbage. The map holds the box
  ///   only weakly, so it is the caller that must root it.
  ///
  /// Declared here and defined in SharedGCBoxing-inline.h, which callers must
  /// include: the definition needs Runtime and BoxedSharedObject complete, and
  /// this header deliberately knows about neither.
  template <typename T, typename... Args>
  PseudoHandle<BoxedSharedObject>
  makeBoxed(Runtime &runtime, uint32_t size, Args &&...args);

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

#ifdef UNIT_TEST
  /// sweep(), calling \p afterEachRegion after each region it sweeps, with
  /// mtx_ released, so a test can allocate in the middle of a sweep.
  size_t sweep(const std::function<void()> &afterEachRegion) {
    sweepHook_ = &afterEachRegion;
    const size_t released = sweep();
    sweepHook_ = nullptr;
    return released;
  }
#endif

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
