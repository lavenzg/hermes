# The Shared Heap Allocator

Date: 2026-08-25
Revised: 2026-11-24
Status: Implemented as `SharedGC`, built only with `HERMES_ENABLE_SHARED_HEAP` (off by default;
enable it with `-DHERMES_ENABLE_SHARED_HEAP=ON` in CMake or `-c hermes.enable_shared_heap=true` in
Buck). `SharedGC::collect()` runs a whole cycle. A `SharedGC` created with a `CollectorConfig` also
runs a collector thread, which calls `collect()` when the trigger says so ("Budget, mappings and
exhaustion"); only tests create a `SharedGC` yet. After turning marking on, `collect()` makes two
passes of marking.md's in-flight check, each waiting only for the barriers in progress when it
looks, so it needs nothing from the runtimes. The materialiser (marking.md P4, P11) is implemented
too; only tests call it.
Relates to: [marking.md](marking.md) (the mark protocol), [barrier.md](barrier.md) (the rings and
`barrierMtx_`), [boxing.md](boxing.md) (the box map and `makeBoxed`)

---

## Summary

How shared cells are allocated and freed; which cells get marked is marking.md's subject. Assumed
throughout: cells never move; a cell's size never changes after allocation (no trimming); no shared
cell has a destructor, finalizer or external memory; and shared allocation is rare, so one global
allocation lock, `mtx_`, is acceptable.

```
storage:  regions of 64 KiB segments from oscompat::vm_allocate_aligned, aligned to 64 KiB
header:   at offset 0 of each region: counters, the stride (slot size), mark and alloc bitmaps
buckets:  40 size buckets up to 8 KiB, each with one segment list and a cursor;
          a larger cell gets a region to itself, listed on largeRegions_
per cell: no GC metadata except its CellKind (plus debug fields)

allocate: under mtx_: skip full segments, take the first free slot, set its alloc and mark bits
mark:     header = addr & ~(kSegmentSize - 1); bit = (addr - header) >> 3; atomic fetch_or
reset:    under mtx_: clear every mark bit, then turn marking on (beginCycle)
sweep:    one segment at a time: free unmarked cells, clear survivors' marks, unmap empty ones
```

Only the sweep frees. New cells are born marked ("black allocation").

---

## Segment layout (implemented)

A **segment** is 64 KiB (`kSegmentSize`). A **region** is one or more contiguous segments, aligned
to `kSegmentSize`, with a `SegmentHeader` at offset 0, so `addr & ~(kSegmentSize - 1)` finds the
header from any cell pointer with no load. Mark bits and cell sizes live there, not in the cells.
Cells start at the fixed offset `kPayloadOffset`, one per **slot**; all slots in a region have the
same size, its **stride**.

**The header is a fixed ~2 KiB** (2088 bytes in a 64-bit release build): counters and two 1 KiB
bitmaps sized for the worst case, one cell per 8-byte granule. With a fixed payload offset, cell 0
sits at the same granule in every region, so the sweeper needs no per-bucket table. Cost per cell:

| Stride | Cells per segment | Header cost per cell |
|---|---|---|
| 8 B | 7931 | 2.1 bits |
| 64 B | 991 | 2.1 B |
| 1 KiB | 61 | 34 B |
| 8 KiB | 7 | 298 B |

**Oversized cells.** A cell above 8 KiB (`SizeBuckets::kMaxBucketSize`) gets a region of as many
segments as it needs, with one slot whose stride is the whole payload. The cell starts at
`kPayloadOffset` in the first segment, the only one with a header, so masking finds the header as
for any region. Such regions sit on `largeRegions_`, which needs no cursor: each is full from
birth.

**Why 64 KiB (settled).** `vm_allocate_aligned` needs whole OS pages, and one AArch64 binary can
meet 4, 16 or 64 KiB pages; the `SharedGC` constructor asserts that `kSegmentSize` is a multiple of
the page size. A 16 KiB segment would abort on a 64 KiB-page kernel in debug builds and, in release
builds, commit a whole page per segment while `footprint()` counted a quarter of it. The cost: one
surviving cell pins 64 KiB. The gain: large buckets pack better, and Windows' 64 KiB `VirtualAlloc`
granularity fits.

---

## Size buckets (implemented)

`SharedGC::SizeBuckets` computes the buckets; there is no table. Up to 128 B (`kPreciseCutoff`)
they step by 8 B, so a 40-byte cell takes exactly 40 bytes. Above that, each doubling
(2^k, 2^(k+1)] is split into `kNumGroup` = 4 even steps, up to 8 KiB, for 40 buckets:

```
8 16 24 ... 128 | 160 192 224 256 | 320 384 448 512 | 640 768 896 1024
              | 1280 1536 1792 2048 | 2560 3072 3584 4096 | 5120 6144 7168 8192
```

Group and offset are bit fields of the size, so `indexOf` is a `Log2_32` and a mask, and `strideOf`
a shift. Rounding waste is at most 1/(`kNumGroup` + 1) = 20%. Sizes are rounded up to 8 bytes
first.

---

## The two bitmaps (implemented)

| Bitmap | Indexed by | Why |
|---|---|---|
| `markWords` | granule: `(addr - header) >> LogHeapAlign` | marking must find a bit from a bare pointer, with no division |
| `allocBits` | cell number | allocation finds a free slot with one bit scan |

`allocBits` is a plain `BitArray`, written only under `mtx_` or by the segment's sweeper.
`markWords` is an array of `std::atomic<uint8_t>`, because markers write it without a lock. This is
marking.md P9's atomic mark bitmap. Only the sweeper converts between granule numbers and cell
numbers.

**Mark words are one byte,** covering 64 bytes of heap, because `std::atomic<uint8_t>` is lock-free
on every target (a `static_assert` checks). A lock-based atomic would hide a lock in the
collector's marking loop and under `mtx_`, where black allocation sets bits. Whether 64-bit atomics
are lock-free on all of Hermes's 32-bit targets is unverified (the comment on `kBitsPerWord` says
they are not; boxing.md O-B1 doubts it), and byte words avoid the question. Contention is not the
reason: a cache line is 64 bytes either way. Neighbouring cells can share a word, so a bit is set
with a relaxed atomic `fetch_or`.

---

## Allocation (implemented)

**`makeA<T>(size, args...)`**, declared only in unit-test builds (open question 10), rounds `size`
up to 8 bytes in 64-bit arithmetic, returns `nullptr` above `maxCellSize()`, takes `mtx_`, claims a
slot, and placement-news `T` into it. It then sets the cell's kind and, in debug builds, a
group-wide allocation id; both come after construction, which would discard earlier writes. The
allocator writes nothing else into the slot.

**`mtx_` is held across `T`'s constructor,** so a heap walk under `mtx_` never sees a claimed but
unconstructed slot. The constructor therefore must not allocate (`mtx_` is not recursive, and a
local allocation can run a local GC under it), take a lock (that would order the lock after
`mtx_`), or make a barriered store (`set()` can call `spill()`, which takes `barrierMtx_` after
`mtx_`, the forbidden order). Fill fields with `initField` or `memset`, as `SharedArray` does. `T`
must be trivially destructible: no destructor ever runs.

**Small cells** (`allocSmallLocked`): step the bucket's **cursor**, an iterator to the first segment
not known to be full, past segments that are full or being swept (`beingSwept` first, because the
sweeper writes `liveCells` without `mtx_`); if none is left while a sweep works through this list,
look again from its head, since the sweep rewinds the cursor only at the end; if still none is left,
map and append a segment; take the first zero alloc bit at or after `nextFreeHint`. `finishAlloc`
claims the slot and sets its mark bit. **Large cells** (`allocLargeLocked`) map a one-slot region,
claim it, and list it on `largeRegions_`.

**Invariants.**

* Every slot below `nextFreeHint` is allocated: allocation takes the first zero at or after it, and
  a sweep resets it to 0.
* The cursor's segment has a free slot, because the cursor loop has just checked. Only a debug
  assert checks that the slot found is in range (open question 5).
* Between sweeps the cursor only moves forward, so allocation is amortised O(1); a sweep rewinds it
  when it is done. While a sweep of the list runs, the look from the head can move it back to a
  segment the sweep has freed slots in, so an allocation at the budget is not refused while there
  is room.
* The per-bucket lists are intrusive (`llvh::simple_ilist`), so they never allocate.
* Occupancy comes from `liveCells`, the count the write depends on, not from list membership.

**Black allocation is unconditional.** `finishAlloc` marks every new cell without reading
`cycleSeq_`, as Hades's `OldGen::finishAlloc` does. Gating it on marking would be wrong: a cell born
after marking turns off but before the sweep would be freed. Within a cycle a new cell looks already
traced, so the collector never reads its fields. Anything stored into it must therefore already be
marked, be a map key that the root scan copies (or that `unregisterMutator` copies if its runtime
leaves before the scan), or have been pushed (marking.md's P4-dominance; channel 9, heap-edge
insertion). That relies on the prune push (P6) and the acquisition push (P4).

**Size limits.** Cells carry no size field. `maxCellSize()`, 4 GiB less one segment, keeps an
oversized region's stride within the 32-bit `SegmentHeader::stride`; the budget bounds real requests
far lower. Visit a cell's fields up to its own length (e.g. `SharedArray::length()`), never up to
`allocatedSize()`: the stride rounds up, and the bytes past the last field are never initialised.
`allocatedSize()` is the slot's extent, which is what accounting must use.

### Allocating a cell with its box (implemented)

The box maps are the collector's root set, so every shared cell a runtime references must be in its
map (boxing.md). `makeBoxed(runtime, size, args...)` allocates the local box first, outside `mtx_`,
because that can run a local GC. Then, holding `mtx_`, it allocates the cell, binds the box, and
publishes the pair into the map (`setBoxForSharedCell` takes the map lock). The mark reset also
takes `mtx_`, so either the reset comes first and the cell is born marked after it, or the cell was
published before the reset and starts the cycle unmarked like any older cell. It is then found
through its map key or through the heap; a key pruned during the cycle is pushed first (P6), and a
runtime that leaves during the cycle copies its keys into the barrier worklist first (marking.md
P12). `makeA` alone publishes nothing (open question 10). The runtime must be attached to this
heap: the root scan reads only registered runtimes' maps, so a cell published into the map of a
runtime attached to another heap, or to none, is not a root here, and unless a traced cell reaches
it the next cycle sweeps it while its box still names it; the other heap's collector would also
mark and trace a cell it does not own. `makeBoxed` asserts it (`Runtime::isAttachedTo`).

---

## Starting a cycle: the mark reset (implemented)

`SharedGC::beginCycle` implements marking.md P8 (black allocation and the reset), taking its locks
one at a time, never nested:

1. Holding `barrierMtx_`, create `markState_`, the per-cycle worklists; `spill()` tests `markState_`
   under that lock.
2. Holding `mtx_`, clear every mark bit (`whitenAllLocked`, a `memset` of each region's mark words)
   and turn marking on (`cycleSeq_` becomes odd), so no allocation lands between the two.
3. Holding `barrierMtx_` again, note every runtime that is inside a barrier window (marking.md P1).

**Why reset.** Cells are born marked whether or not a cycle runs, and only a sweep clears marks. A
cell allocated since the last sweep, or marked by an aborted cycle, would start the cycle marked.
Since a mark also means "already queued", its fields would not be traced, and the sweep would free
what only it references. The sweep's clearing of survivors does not cover cells born after it
passed.

**Why under `mtx_`, with turning marking on.** A `memset` racing black allocation's `fetch_or` could
erase a birth mark, so the reset excludes allocation. In one critical section with turning marking
on, every cell is allocated either before both (unmarked, so judged like any older cell) or after
both (born marked, and nothing clears the mark during the cycle). The reset clears 1 KiB per region
while allocation waits.

**No cycle may start while a sweep runs.** The sweeper frees what is unmarked, and clears a
survivor's mark with a relaxed load and a separate store, not an atomic read-modify-write. Racing
the reset, the sweeper would free survivors whose marks were just cleared, and its stale store can
bring cleared marks back: those cells look already traced, so their children are never traced and
can be freed. Racing a marker, it can erase the marker's bits and free cells the marker has not
reached yet. Only a debug assert in `whitenAllLocked` checks this. `collect()` runs its steps in
sequence and holds `cycleMtx_` throughout, so cycles never overlap, whichever thread starts them
(open question 6).

---

## Sweeping (implemented)

`SharedGC::sweep()` runs after certification (marking.md P10's termination check) has turned
marking off, and after `endCycle`; mutators keep allocating. Debug builds assert that it does not
run while a cycle's state still exists (between certification and `endCycle` the worklists still
name cells), after `abortCycle` (the marks are a partial trace; start a new cycle instead), or twice
at once.

```
for each bucket's list, then largeRegions_ (which has no cursor):
    lock mtx_; skip the list if it is empty; note it as the list being swept
    last = the list's last segment             // segments added during the sweep wait
    for each segment s, from the first through last:
        set s.beingSwept; unlock mtx_
        for each allocated slot in s:          // no lock: the sweeper owns s
            marked:   clear the mark           // a survivor
            unmarked: clear the alloc bit      // freed
        set s.liveCells; s.nextFreeHint = 0
        lock mtx_; clear s.beingSwept
        if s is empty: move the cursor off s, unlink s, unlock, unmap s, lock
    rewind the cursor to the first segment; note no list as being swept; unlock
```

**`beingSwept` makes the sweeper the segment's only writer**: allocation skips that segment and
appends new ones after `last`. So the bitmap work needs no lock and no atomic read-modify-write, and
iterators survive the unlocks (the list is intrusive, allocation only appends, and only the sweeper
erases). `MakeA_DuringASweep_ReusesSlotsTheSweepFreed` allocates at the budget between two regions
of a sweep, through a test-only hook, and must get a slot that sweep freed. Other tests allocate
while `collect()` sweeps on another thread (`Concurrent_MoveBetweenBoxedCellsRetainsTheTarget` and
the collector-thread tests), but leave the interleaving to timing.

**The cursor moves off an emptied segment in the same `mtx_` critical section as the unlink.**
Otherwise an allocation that takes `mtx_` while the sweeper unmaps the segment follows the cursor
into it: `beingSwept` is already clear and `liveCells` is 0, so the segment looks allocatable.

**Exactly one sweeper.** `beingSwept` does not exclude a second sweeper. It would claim the same
segments, rewrite their bitmaps without a lock, and could erase and unmap a segment twice; the
iterator argument above also assumes one sweeper. Release builds do not check this; only the debug
`sweeping_` assert does.

**No cell is read:** liveness comes from the bitmaps and size from the stride. Clearing survivors'
marks lets two sweeps in a row, with no allocation between, free everything (tests rely on this);
it does not replace the reset.

**Empty segments are unmapped at once** (`releaseRegion`), outside `mtx_` after being unlinked
under it. That is how memory moves between buckets without compaction, and it beats pooling for
RSS. It also means a pointer to a cell freed in an earlier cycle may point at unmapped memory. That
is why entries in the rings (each runtime's buffer of pushed cells) carry a cycle stamp, and stale
ones are dropped unread (barrier.md, "The stamp: a push that outlives its cycle").

**Floating garbage.** A cell allocated after a cycle starts survives that cycle's sweep; the next
cycle's reset clears its mark, and that cycle judges it. This is the usual cost of SATB
(snapshot-at-the-beginning) marking.

**What publishes the marks to the sweeper.** Black allocation marks under `mtx_`, which the sweeper
also takes. The collector marks with relaxed atomics, without `mtx_`, so sweep on the thread that
traced (`endCycle`'s contract), as `collect()` does: program order then covers it. On another
thread, the seq_cst load of `cycleSeq_` at the top of `sweep()` acquires the marks only if it reads
`certifyRound`'s increment or a later value, and an unsynchronised thread may read an even value
from before the cycle began. The load is a backstop: keep it outside the `assert` that uses it,
since asserts compile out of release builds, and keep the increment seq_cst.

---

## Locks (implemented)

| Lock | Guards |
|---|---|
| `mtx_` (`SharedGC`) | the segment lists, cursors and `beingSwept`; each segment's `allocBits`, `liveCells` and `nextFreeHint`, except while it is being swept; black allocation's mark bits; the mark reset and turning marking on. Held across a cell's constructor and across `makeBoxed`'s publish. |
| `barrierMtx_` (`SharedGC`) | the runtime registry, whether `markState_` exists, the barrier worklist, ring drains, and certification, which turns marking off |
| map lock, `sharedObjectBoxesMtx_` (each `Runtime`) | the box map's structure: insert (may rehash), prune (erase), and the root scan's key copy, which `unregisterMutator` waits for by taking it after the unlink. Not the local GC's weak-root walk, which only rewrites values. |
| `gcMutex_` (each local heap) | unchanged |
| `cycleMtx_` (`SharedGC`) | one `collect()` at a time, the collector thread's included; held for the whole cycle |
| `collectorMtx_` (`SharedGC`) | the collector thread's sleep and `stopCollector_`; held while the thread decides whether a cycle is due, briefly by each wake, and by the destructor to stop the thread |

**Lock order.** The full order is in marking.md, "State and locks": `cycleMtx_` comes before every
other lock, since `collect()` holds it for the whole cycle, and `collectorMtx_` after `mtx_` and
`barrierMtx_`, under which a wake takes it; nothing is taken under it. The rules that involve
`mtx_`:

* `mtx_` then a map lock: `makeBoxed` publishes under both. Never `mtx_` under a map lock.
* `barrierMtx_` then `mtx_`, if both are ever held; nothing holds both today. Never `mtx_` then
  `barrierMtx_`. Hence no barriered store in a shared cell's constructor, and
  `setBoxForSharedCell`, called under `mtx_` by `makeBoxed`, never prunes, since the prune push
  (marking.md P6) can spill (the `Insert_DoesNotPrune` test checks this); `makeBoxed` prunes
  before it takes `mtx_`.
  Allocation itself never pushes to a ring.
* `mtx_` and a runtime's `gcMutex_` are never nested, in either order. A local allocation under
  `mtx_` can take `gcMutex_` and run a whole local GC while all shared allocation waits. Code that
  runs under `gcMutex_`, such as `Runtime::markWeakRoots`, must not allocate in the shared heap.
  Taking them in both orders can deadlock.

**Who writes what.**

| Data | Writers | Protection |
|---|---|---|
| `cycleSeq_` | the collector only: `beginCycle` (even to odd, under `mtx_`); `beginSecondInFlightPass` (a `fetch_add(0)` that keeps the value, under no lock; marking.md P1); `certifyRound` (odd to even, the normal end of marking), `abortCycle`, and a release-build repair in `endCycle` (all under `barrierMtx_`) | atomic, seq_cst `fetch_add` |
| mark bits, set | the collector while tracing; black allocation, under `mtx_` | atomic relaxed `fetch_or` |
| mark bits, cleared | the reset, under `mtx_` with no sweep running; the sweeper, on a segment it owns | `memset`; relaxed load and store |
| `allocBits`, `liveCells`, `nextFreeHint` | the allocator under `mtx_`; the sweeper under `beingSwept` | one writer at a time, plain |
| `beingSwept`, lists, cursors | the allocator and the sweeper | `mtx_` |
| `footprint_` | `createRegion`, under `mtx_`; `releaseRegion`, from the sweeper without `mtx_` | atomic; reserved by compare-and-swap before mapping |

---

## Budget, mappings and exhaustion

**The budget, `maxSize_`, belongs to the group** (implemented): fixed when the `SharedGC` is
created, never charged to any runtime's `MaxHeapSize`. `footprint()` is exact mapped bytes.
`createRegion` reserves bytes by compare-and-swap on `footprint_` before mapping and refunds them
on failure, so the budget is a hard bound even while the sweeper refunds without `mtx_`. Charging
is per region. Not yet implemented: exposing the budget as a group-scoped counter and as a group
key in `CrashManager` data, never as a heap-size key.

**One mapping per region.** 100 MB of bucketed segments is up to about 1600 mappings, against
Linux's default `vm.max_map_count` of 65530 per process, each one or more syscalls under `mtx_`. If
that shows up, carve segments from larger reservations, as Hades does through `StorageProvider`. On
Android, regions are named `hermes-shared` in `/proc/<pid>/maps` (`oscompat::vm_name`, which does
nothing on other platforms).

**Exhaustion returns `nullptr`, never `oom()`** (implemented). On budget exhaustion, a failed
mapping or an oversized request, `makeA` returns `nullptr` and `makeBoxed` an empty handle; the
caller is meant to throw a JS `RangeError` (no JS caller exists yet). `oom()`, or `checkedMalloc`,
which calls `hermes_fatal`, would kill every runtime in the group for one allocation.

**Collecting before throwing** (open: a failed allocation at most wakes the collector thread, as
below, and its caller gets an empty handle). A Hades mutator that runs out does the collector's
work itself (`waitForCollectionToFinish`). A shared-heap runtime could do the same, since a cycle
needs no action from other runtimes (both in-flight passes wait only for barriers already in
progress): it could run `collect()` itself, or block with a timeout until the collector thread
finishes; one cycle runs at a time, so a second runtime that runs out must block regardless. In
every case, retry only after `makeBoxed` returns, holding neither `mtx_` nor a map lock
(`beginCycle` takes `mtx_`; the root scan takes every map lock), and never wait while inside a
barrier, because the collector waits for the runtime to leave it. The retry also has to settle:

* **Which failures to retry.** `makeBoxed` returns the same empty handle for a request that can
  never succeed (above `maxCellSize()`, or above the whole budget, which wakes nothing) as for an
  exhausted budget or a failed mapping, so its caller cannot tell when a retry is pointless.
* **How to wait for a cycle.** Nothing signals that one has ended: only tests read `cycleCount_`,
  and nothing notifies a waiter. A refusal makes a cycle due, but the thread still waits out the
  gap below first; `collect()` called directly skips it.
* **What a cycle can free.** Only cells no key reaches. A key whose box is dead stays a root until
  its runtime prunes, at its next lookup or box allocation (boxing.md, "Pruning: flag dead entries
  during GC, erase at the next lookup or box allocation"), and a dead old-generation box is found
  dead only when an old-generation collection completes its marking (`completeMarking`). So the
  refused runtime may need a local collection and a prune before it collects, and other runtimes'
  dead keys stay until those runtimes prune or leave.
* **How many attempts.** Retention is peak-live per bucket (open question 3), so a cycle can free
  cells and still leave the request's bucket full and no budget for a new segment. How many cycles
  to try before throwing the `RangeError` is open.

**The collector thread and its trigger** (implemented). Only a `SharedGC` created with a
`CollectorConfig` has the thread; without one, cycles run only when something calls `collect()`. The
thread sleeps until a cycle is due, waits out the gap, and calls `collect()`. A cycle is due when:

* **Footprint:** `footprint_` reaches `triggerBytes_`. After its sweep, each `collect()` sets
  `triggerBytes_` to twice the footprint the sweep left, or that footprint plus `minTriggerBytes`
  if that is larger, but never more than halfway from that footprint to the budget, so a cycle
  starts while there is still room to allocate. If that allows no growth, as when live cells fill
  the budget, the footprint never makes a cycle due; otherwise the thread would collect back to
  back while nothing happens. The constructor starts the trigger at its value for an empty heap,
  `minTriggerBytes` or half the budget, whichever is less; a `sweep()` called on its own leaves it
  alone. `createRegion` compares after each reservation.
* **Exhaustion:** a reservation the budget refuses for lack of room, or a failed mapping, since the
  last cycle began. It wakes the thread too. A region larger than the whole budget does not count,
  since no cycle can make room for it.
* **Time:** something was allocated, or a runtime left, since the last cycle began, and `maxGap`
  has passed since the thread's last cycle ended, or since the thread started if it has run none.
  A runtime that leaves turns the cells only it named into garbage without allocating anything, so
  it counts too. A group in which neither happens never wakes the thread.

**The gap.** The thread starts a cycle no sooner than `minGap` after its last cycle ended, and no
sooner than that cycle took, which keeps it busy at most half the time. It measures only its own
cycles: one that a caller runs through `collect()` clears the exhaustion and time triggers and
resets `triggerBytes_`, but moves neither clock, so the thread may start a cycle as soon as the
caller's ends.

`CollectorConfig` defaults to a `minTriggerBytes` of 4 MiB, a `minGap` of 100 ms and a `maxGap` of
30 s; each gap is capped at a year, so that no deadline overflows the clock. The rules and these
defaults are unmeasured guesses (open question 12).

**Waking costs a lock and a notify,** and each kind of event wakes the thread once per cycle: the
reservation that crosses the trigger, the first refusal, and the first allocation or departure. The
wake is taken under `mtx_`, or `barrierMtx_` for a departure, so both come before `collectorMtx_`;
the thread holds `collectorMtx_` only while it decides or sleeps, never while it collects.

**The exhaustion and time triggers clear as a cycle begins,** in `beginCycle`, under `mtx_` and
before marking turns on. A refusal or a first allocation, both under `mtx_`, either comes before and
that cycle covers it, or comes after and makes the next one due. A departure holds only
`barrierMtx_`. One that finds marking on, or that the root scan still finds, turns the time trigger
on after the clear and makes the next cycle due, since this one keeps what the runtime named; one
that leaves earlier, this cycle covers.

**Cycles take turns.** `collect()` holds `cycleMtx_` for the whole cycle, so a cycle the thread
starts and one a caller starts run one after the other. The step-by-step entry points (`beginCycle`
through `endCycle` and `sweep`) take no such lock and must not be called while the thread runs.

**Teardown.** `~SharedGC` stops the thread, which finishes any cycle it is running. With no runtime
attached nothing is a root, so it then runs one more `collect()`, whose sweep frees every cell left,
asserts that the heap is empty and releases the regions (boxing.md O-B4).

---

## Debug and ASan checks (implemented unless noted)

The ASan macros are no-ops outside ASan builds, so the poison calls are unconditional, as in
`HadesGC.cpp`; only the sweep's re-poisoning loop is compiled out.

* **Free slots are poisoned.** `createRegion` poisons a multi-cell segment's payload and
  `finishAlloc` unpoisons each slot it claims, so an overflow into an unclaimed slot, or past the
  last one, is caught; a fresh mapping is otherwise writable. An oversized region's one cell is
  unpoisoned at once, so it is not poisoned. The sweep re-poisons freed slots only in segments that
  survive, since a segment about to be unmapped is unpoisoned anyway.
* **Regions are unpoisoned before unmapping.** ASan's shadow outlives the mapping, so a later
  mapping at that address would start out poisoned.
* **Header canary** (debug). `headerOf()` checks `SegmentHeader::kMagic` on every
  `setCellMarkBit`, `isMarked` and `allocatedSize`, and `granuleOf` asserts that the pointer starts
  a slot. A `GCCell *` does not convert to `SharedGCCell *`, but a wild pointer, or an interior
  pointer into an oversized region, would otherwise mask onto arbitrary memory and be written.
* **Cell canary and group-wide allocation ids** (debug). `SharedGCCell`'s constructor writes a
  magic number that `isValid()` checks along with the kind. The id counter lives on `SharedGC`,
  because `GCBase::nextObjectID` is per runtime and would repeat across runtimes.
* **Reachability checks** (`HERMES_SLOW_DEBUG`; marking.md, "A safety net for P10"). `endCycle`
  re-traces from the box-map keys (`countReachableUnmarked`) and asserts that every cell it reaches
  is marked and still allocated, and `collect()` asserts after its sweep that no key names a freed
  cell (`countDanglingKeys`). Both test a pointer with `isAllocatedLocked`, which finds its region
  among the regions sorted by address and reads its alloc bit, never the cell, so a freed or
  unmapped cell is safe to test.
* **Not implemented:** a quarantine mode that holds swept slots out of reuse for some cycles,
  standing in for `HERMESVM_SANITIZE_HANDLES`, which cannot work on a heap that never moves; and an
  exact census of cells, probably never needed, since the header canary and `isAllocatedLocked`
  cover its uses and a known cell's shared-ness is a `CellKind` range test.
* **Lost by owning the storage:** LSan cannot see a leaked cell in a mapped region, and TSan has no
  `free()` to treat as synchronisation.

---

## Rationale and rejected alternatives

**Sources.** JSC runs the same shape (one size per block, mark bits in block metadata, blocks freed
only when empty) at scale without moving cells, which is the evidence that a size-class heap needs
no compaction. The self-aligned segment comes from Hades and the bucket layout from jemalloc. Left
out of the Hades port: generational machinery (the shared heap has one generation, and boxing
forbids shared-to-local edges), moving machinery, trimming, finalizers, external memory, and
compressed pointers. There is no shared `PointerBase`, and `_sh_cp_encode_non_null` would truncate
a shared address, so never call `registerHeapSegment` for shared storage.

**Rejected.**

* **Mixed sizes per segment with a free list (V8's old space):** packs tighter but fragments, which
  V8 answers with compaction, ruled out here.
* **A per-cell header plus a registry entry:** about 24 B per cell, more than the segment header
  costs for any cell under about 700 bytes; above that the segment header costs more, but is small
  next to rounding waste.
* **Geometric size classes:** inverting them needs a 1 KiB table and a load.
* **A radix tree from address to metadata (jemalloc 5) instead of masking:** masking is what lets a
  cell carry no header.
* **JSC's half-alignment trick for large cells:** not needed, since an oversized region is
  self-aligned like any other. That also lets `HeapAlign` stay 8 rather than JSC's 16-byte atom.
* **A 64-bit `SegmentHeader::stride`:** would remove the `maxCellSize()` clamp, at 4 bytes per
  header, for sizes no real cell reaches.
* **Separate lists for partly used and full segments:** duplicate `liveCells`; if the two disagreed,
  a release build would write past a full segment.
* **`std::list` for segments:** allocates on the allocation path, and under `-fno-exceptions` a
  failed allocation ends the process.
* **Detaching a bucket's whole list for the sweep:** keeps allocation off the bucket's existing
  segments until the sweep ends; flagging one segment at a time does not.

---

## Future options (not planned)

* **A free list inside each segment** (JSC): drops the alloc bitmap, half the header, at the cost
  of writing into dead cells. Worth it only if the bit scan shows up.
* **Per-runtime local allocators:** take `mtx_` off the common path if shared allocation stops
  being rare, as JSC's per-thread `LocalAllocator` does. marking.md's future options describe a
  variant that marks whole chunks at refill. Either changes black allocation and the reset, so it
  must be designed with marking.md.
* **Compaction:** ruled out, hence peak-live retention (open question 3). Boxing would make it
  possible later, since every outside reference to a shared cell is a box in some runtime's map; the
  hard part is raw `SharedGCCell *` values on C++ stacks.

---

## Open questions

1. **Segment size.** Settled at 64 KiB (see Segment layout).
2. **Bucket constants.** `kPreciseCutoff = 128` and `kNumGroup = 4` are guesses; only their shape is
   principled. Shared structs have declared shapes, so the size distribution should be narrow and
   measurable. More buckets cut rounding waste (`kNumGroup = 8`: 11% with 64 buckets), but each can
   hold a segment open, the wrong trade while retention is peak-live (question 3). Also, a 9 KiB
   cell gets a whole 64 KiB region and wastes about 53 KiB; `kLogMaxBucketSize` can rise to 15
   before the largest bucket stops fitting in a segment (a `static_assert` checks). Raising it makes
   the largest bucket pack worse: 8 KiB gives 7 slots per segment (about 1/8 lost to header and
   tail), 16 KiB gives 3 (1/4), 32 KiB gives 1 (1/2).
3. **Retention is peak-live per bucket, not live.** A bucket that grew and then shrank keeps every
   segment that still holds one survivor; only compaction fixes that. The bet is that shared structs
   have few, stable sizes. Check it against a real workload.
4. **Floating garbage for a cell born during a sweep.** Resolved: the reset in `beginCycle` clears
   its birth mark, so the next cycle judges it.
5. **`assert(i < hdr.numCells)` should be a release check.** If the cursor invariant ever breaks, a
   release build writes past the segment, and the branch is cheap.
6. **Whether to keep the rule that no cycle starts during a sweep.** It lets the sweeper clear marks
   without atomic read-modify-writes; overlap would need a per-segment cycle number recording which
   cycle's marks the bitmap holds. Not needed while `cycleMtx_` keeps whole cycles from overlapping.
7. **Publishing a new cell to other threads.** Resolved: `SharedValue::set`'s slot store is seq_cst,
   so a release, and every slot read that may be followed to a cell is at least an acquire (the copy
   constructor, `getPointer`, and `getPointerOrNull`, which the tracer uses). A reader on another
   thread, the collector included, that reads a cell's address from a slot therefore sees the cell's
   construction and birth mark (boxing.md O-B10). `isEmpty`, `isPointer` and `initField` stay
   relaxed: nothing is read through the first two, and `initField` is only for a cell no other
   thread can reach yet.
8. **Work under `mtx_`.** `makeBoxed`'s map insert can rehash and call `malloc` under `mtx_`, and
   `reserve()` alone does not bound it, because the prune's tombstones can force a same-size rehash,
   and that still allocates a new table. The insert also takes the map lock, so while the root scan
   copies that runtime's keys the insert waits for the copy holding `mtx_`, and every other
   runtime's shared allocation waits behind it (boxing.md, "The collector's read: one lock per
   runtime"). Mapping a region is also one or more syscalls under `mtx_`.
9. **Sizes computed by callers.** Settled for `SharedArray`: `allocationSize` computes in 64 bits
   and returns `UINT32_MAX`, which `makeBoxed` and `makeA` refuse, for a length whose cell would
   exceed `maxCellSize()`. In 32 bits, about 2^29 elements would wrap to a tiny cell that the
   constructor overruns. Both still trust their size apart from rounding, so a future cell type
   sized by its contents needs the same care.
10. **Raw allocation is test-only.** Settled: `makeA` and `SharedArray::create` put nothing in a
    map, so their cells can be swept while still held. They are declared only in `UNIT_TEST` builds,
    which the tests need, and production code allocates through `makeBoxed` (boxing.md O-B12).
11. **Teardown with cells left.** Settled: `~SharedGC` runs one more cycle, with no roots, whose
    sweep frees whatever is left (boxing.md O-B4).
12. **The collector thread's trigger.** `minTriggerBytes`, the doubling, the halfway cap and both
    gaps are guesses (marking.md, "Trigger tuning"); measure them on a real workload. Also open:
    stores and prunes make garbage without allocating, as when a runtime drops boxes and a later
    lookup prunes their keys, but neither makes a cycle due, so a group that only stores and prunes
    keeps that garbage mapped until a runtime allocates, fails to allocate, or leaves.
