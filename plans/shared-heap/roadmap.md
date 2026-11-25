# Remaining Work on the Shared Heap

Date: 2026-11-24
Status: Plan. The shared heap short of any JS-visible surface is implemented and unit-tested, built
only with `HERMES_ENABLE_SHARED_HEAP`. Nothing outside the unit tests creates a `SharedGC`, and no
JS can reach a shared cell.
Relates to: [boxing.md](boxing.md), [marking.md](marking.md), [barrier.md](barrier.md),
[allocator.md](allocator.md), [marking-implementation.md](marking-implementation.md)

## Where it stands

Implemented and unit-tested:

* **Allocator:** 64 KiB self-aligned segments, size buckets, a group-wide budget, sweep, and
  teardown that frees whatever is left.
* **Boxing:** per-runtime box maps as the root set, pruned at lookups and box allocations;
  `makeBoxed`, which asserts that its runtime is attached; and the materialiser.
* **Marking:** SATB marking with per-runtime barrier rings, the two-pass in-flight check,
  certification, and a post-certification check in `HERMES_SLOW_DEBUG` builds.
* **Runtimes:** attaching at `Runtime::create`, handing the keys over on detach while marking is
  on, and a collector thread with footprint, exhaustion and time triggers.

Not started: everything JS-visible (boxing.md, Appendices A to D), an embedder API, and Workers
joining a group.

Each item gives what exists today, the work, the tests that show it works, and a size: S is one
diff, M two to four, L a design note and several diffs. Items cite the open questions in the design
docs rather than repeating them.

## Step 0: ready the stack for review

### 0.1 Tests that order threads by sleeping

A review of the whole stack found no hole in the marking protocol, and its other findings are fixed
in the commits that introduced them. Five tests still sleep for a fixed time and assume the other
thread has reached a state by then. Under load they can pass without reaching the interleaving they
are about, though they cannot fail falsely:

* `Scan_LetsARuntimeSpillWhileAMapIsBusy` and `Scan_SkipsARuntimeThatLeavesWhileItsMapIsBusy`
  (`SharedRootScanTest`) sleep 200 ms so that the scan reaches the busy map first. A test hook in
  `copyRootKeys` that runs when it finds a map busy, like the hook
  `Scan_CopiesUnderTheMapLockAlone` uses before each copy, would start the other thread at the
  right moment.
* `Collect_WaitsForABarrierInFlight` and `Collect_WaitsForAWindowOpenedDuringTheFirstWait`
  (`SharedCollectTest`) hold a window open for 50 ms, long enough for a `collect()` that skips the
  wait they test to finish the cycle. Waiting until `collect()` has polled the open window, through
  a test-only count of the polls in each pass, would replace the sleep: a `collect()` that skips
  the wait never polls it.
* `Detach_WaitsForACopyOfItsKeysToFinish` sleeps 200 ms to show that a detach does not finish
  while a copy holds its map. That checks for something not happening, so a bound in time stays.

The timers in `SharedCollectorThreadTest` sleep because the triggers they test are timed.

**Size:** S.

### 0.2 Before review

* Run `check-hermes` with the flag on, not only the unit tests.
* Add a lit test that runs code compiled by shermes with the flag on. An inline `typeof` in such
  code is what showed that generated code must see the flag.
* Measure `libhermesvm` with the flag off at the base and at the top of the stack; it should not
  change.

**Size:** S.

### 0.3 Review order

Four design-doc commits sit at the bottom of the stack, and the marking implementation plan sits in
its middle: land them first, or move them into one doc commit at the top. Review the code in six
parts, each a run of consecutive code commits:

1. Cells and the allocator: `SharedGCCell`, `SharedValue`, the allocator, `makeA`.
2. The box and its map: `BoxedSharedObject`, the box map and its lock, allocating a cell with its
   box, `SharedArray`.
3. Tracing: `SharedSlotVisitor`, `SharedMarkAcceptor`, `SharedBarrierRing`, `SharedMutator` and the
   barrier registry, the root scan.
4. Ending and starting a cycle: `certifyRound`, the SATB barrier in `SharedValue::set`, the reset
   and `abortCycle`, the in-flight check, `collect`.
5. The runtime glue: attaching a runtime, the keys a runtime hands over when it leaves mid-cycle,
   the publication test, the post-certification check, the materialiser.
6. The collector thread and teardown.

## Step 1: production readiness, before any JS

### 1.1 What a refused allocation does

**Today.** `makeBoxed` returns an empty handle both for a size above `maxCellSize()` and when
`createRegion` is refused by the budget or fails to map; the first refusal since a cycle began sets
`starved_` and wakes the collector thread. Nothing outside the unit tests calls `makeBoxed`.

**Work** (allocator.md, "Collecting before throwing"):

* Tell the failures apart. A request above `maxCellSize()`, or above the whole budget, can never
  succeed and should throw at once; only exhaustion and a failed mapping are worth a retry.
* A signal that a cycle ended: `collect()` notifies a condition variable when it bumps
  `cycleCount_`, and a runtime waits, with a timeout, for the count to pass the value it read.
* The retry, in one `Runtime` helper that every JS allocation path calls: prune, run a local
  collection if dead boxes may not have been found yet, run `collect()` or wait for the thread's
  cycle, retry, and after a bounded number of attempts throw `RangeError`.
* Retry only after `makeBoxed` returns, holding neither `mtx_` nor a map lock, and never inside a
  barrier window, since the collector waits for every runtime to leave its window.

**Open:** whether the refused runtime runs the cycle itself, as a Hades mutator does in
`waitForCollectionToFinish`, or waits for the thread; the number of attempts; the timeout.

**Tests:** an allocation refused while another runtime's dropped boxes fill the budget succeeds
after the retry; an oversized request throws without starting a cycle; a budget full of live cells
throws after the last attempt; under TSan, retries from two runtimes race the collector thread.

**Size:** M. Every JS allocation path needs it.

### 1.2 An embedder API

**Today.** `Runtime::create(config, sharedGC)` attaches the new runtime when given a `SharedGC`.
`RuntimeConfig` has no shared-heap field, and the JSI factory `makeHermesRuntime` and the C ABI's
runtime creation (`hermes_vtable.cpp`) call `Runtime::create(config)` without one.

**Work:**

* A public, opaque group type that owns a `SharedGC`: its budget, its `CollectorConfig` and a
  `CrashManager` for its own keys (1.7).
* A way to create a JSI runtime in a group: a `RuntimeConfig` field, which like `CrashMgr` would be
  a `shared_ptr` and has to exist with the flag off too, since `RuntimeConfig` is a public type; or
  a `makeHermesRuntime` overload.
* Lifetime: each runtime holds a reference to its group, and the last holder runs the teardown in
  `~SharedGC`, possibly inside a runtime's destructor on that runtime's thread.

**Open:** a config field or a factory parameter; whether the C ABI and the sandbox get it now.

**Tests:** two JSI runtimes created in one group are both attached to it; destroying them in either
order tears the group down; a runtime created without a group behaves as it does today.

**Size:** S to M.

### 1.3 Workers join their parent's group

**Today.** `startWorker` (`API/hermes/extensions/Worker.cpp`) creates each worker with
`makeHermesRuntime(RuntimeConfig())`, so a worker never shares a heap with its parent.

**Work:** create the worker in its parent's group, if the parent has one, through 1.2. Decide what
else a worker takes from its parent's configuration.

**Tests:** a worker of a runtime in a group is attached to the same `SharedGC`; a worker of a
runtime without one is not.

**Size:** S, after 1.2. The `postMessage` hand-off (2.4) needs it.

### 1.4 Garbage from prunes and stores makes a cycle due

**Today.** The time trigger, `changedSinceCycle_`, is set by the first allocation since a cycle
began (`finishAlloc`) and by a runtime leaving (`unlinkMutator`). A prune that erases keys and a
store that drops a cell's last reference make garbage but set nothing, so a group that only stores
and prunes keeps that garbage mapped until a runtime allocates, is refused or leaves (allocator.md,
open question 12).

**Work:** set the trigger in `Runtime::pruneSharedObjectBoxes` when it erases a key. For stores,
decide whether the barrier's pre-read finding a pointer should set it too, at the cost of a relaxed
load on every store that overwrites a pointer.

**Tests:** with a short `maxGap`, a group that only prunes gets a cycle.

**Size:** S.

### 1.5 A bound on the barrier worklist

**Today.** `barrierWorklist` has no cap. Every entry a runtime pushes while marking is on lands
there when its ring spills, repeats of marked cells included, and `collect()` empties it only in
`certifyRound`. A runtime storing in a tight loop during a long trace grows it, in memory the budget
does not count (barrier.md, open question 7).

**Work:** accept the growth, or bound it: drain the worklist between batches of tracing, or drop at
the spill the entries whose cell is already marked.

**Tests:** a runtime storing in a loop while a long trace runs keeps the worklist below a bound,
through a test-only size probe; the certification tests still pass.

**Size:** S to M.

### 1.6 Backoff in the collector's waits

**Today.** `collect()` spins on `pollInFlightBarriers()` with `std::this_thread::yield()` through
both in-flight waits, and `copyRootKeys` retries a busy map after a `yield`. A runtime descheduled
inside a window keeps the collector thread spinning for as long as it is off the CPU.

**Work:** spin a few times, then yield, then sleep for growing intervals up to a cap of about a
millisecond, in both loops. The mutator side does not change.

**Tests:** a window held open for two seconds costs the collector thread little CPU time, measured
with the thread's CPU clock; the existing wait tests still pass.

**Size:** S.

### 1.7 Exposing the group's numbers

**Today.** `footprint()` and the budget are readable only from C++. Hades sets the `CrashManager`
keys `HermesGC` and one per heap segment, and JSI's `getHeapInfo` returns `hermes_`-prefixed keys
for one runtime's heap.

**Work:** footprint, budget, cycle count and refusals, readable from the group handle (1.2); group
keys in its `CrashManager` data, such as the region extents Hades records for its segments. Never
add the shared footprint to a runtime's `hermes_heapSize`: every runtime in the group would count
it (allocator.md, "Budget, mappings and exhaustion").

**Open:** whether `getHeapInfo` gets separate keys for the group.

**Size:** S.

### 1.8 32-bit targets

**Today.** Each `SharedValue` slot is a `std::atomic<uint64_t>`, as is `cycleCount_`, and nothing
checks that they are lock-free; the only `is_always_lock_free` check is on the byte-wide mark
words. The comment on `kBitsPerWord` says a 64-bit atomic is not lock-free on the 32-bit targets
Hermes ships to. That is unverified: ARMv7-A has `LDREXD`/`STREXD`, and x86-32 has `CMPXCHG8B`
(boxing.md, O-B1).

**Work:** check `is_always_lock_free` with the toolchains Hermes ships with for armeabi-v7a and
x86; add a `static_assert` so a target where it fails stops the build; correct the comment; run the
shared-heap suites in a 32-bit `HEAP_HV_PREFER32` build. If a target is not lock-free, slot accesses
stay correct but take a lock; narrowing the slot to 4 bytes with its own tagging avoids that
(O-B1).

**Size:** S, or L if a narrow slot is needed.

### 1.9 A termination safety net for release builds

**Today.** `HERMES_SLOW_DEBUG` builds re-trace from the box-map keys at every `endCycle`, assert
that every cell reached is marked and allocated, and check after each sweep that no key names a
freed cell. Release builds have none of it, and a sweep frees rather than whitens, so one store that
skipped its barrier frees a reachable cell silently (marking.md, "A safety net for P10").

**Work:** choose a guard, or decide on none. Candidates: never free a cell in the cycle in which it
was last seen reachable, at the cost of a cycle of floating garbage; or run the check on a sample of
cycles in release builds and record a mismatch in crash data instead of asserting.

**Tests:** a store that skips its barrier, through a test-only route, is caught by the guard.

**Size:** M.

### 1.10 Trigger tuning

`minTriggerBytes` (4 MiB), the doubling and its halfway cap, `minGap` (100 ms) and `maxGap` (30 s)
are guesses (allocator.md, open question 12). Measure footprint over time, cycle frequency and the
collector's CPU time on a workload in which several runtimes share data. That workload comes with
the first prototype in Step 2.

**Size:** S to M.

### 1.11 CI

**Today.** No CI job builds the shared heap. The CMake jobs in `.github/workflows/build.yml` pass no
shared-heap flag, and Buck builds it only with `-c hermes.enable_shared_heap=true`. CMake Debug
builds define `HERMES_SLOW_DEBUG` by default and the `BUCK` file never defines it, so under Buck the
post-certification checks do not run.

**Work:** one CMake Debug job with `-DHERMES_ENABLE_SHARED_HEAP=ON`, which runs those checks; a TSan
job over the shared-heap suites in Debug and in Release, since an assert that loads with acquire can
hide a race a release build has; and one Buck configuration with the flag on. The flag stays off by
default.

**Size:** S.

### 1.12 Cheap checks the docs ask for

* `assert(i < hdr.numCells)` in allocation becomes a release check: if the cursor invariant breaks,
  a release build writes past the segment (allocator.md, open question 5).
* The provenance rules, which nothing enforces (barrier.md, open question 3): only `makeBoxed`,
  under `mtx_`, and the materialiser, inside its window, should call `setBoxForSharedCell`. Limit
  it to those two callers, and assert the window on the materialiser's path.

**Size:** S.

## Step 2: the JS surface

The first prototype exposes only `SharedArray`, so it needs indexed access and no field names.
Shared objects have a fixed layout, so when structs come, a field is found through a per-type map
from its name to its slot, and objects in the shared heap need no hidden classes. A box keeps its
own: an empty, non-extensible root that only reserves the slots its cell pointer overlaps.

The prototype runs in the interpreter first; 2.6 covers Static Hermes and the JIT.

### 2.1 `SharedArray` in JS

**Today.** The `SharedArray` cell has a length fixed at construction, as TC39 requires, and
`at`, `set(i, v, mutator)` and `initField`; production code can only create one through
`makeBoxed`, and nothing does. The box's `vt` names the seven indexed hooks, but they are
`JSObject`'s defaults: an empty range, every lookup missing, every store refused. Its class has no
`indexedStorage` flag, so an index is looked up as a named property, which the empty,
non-extensible class neither has nor accepts.

**Work:**

* A `SharedArray` constructor in runtimes attached to a group, which allocates through `makeBoxed`
  and 1.1's retry, with every element `undefined`.
* The seven indexed hooks for a box over a `SharedArray`: the range `[0, length)`; each element
  writable and enumerable but not configurable; a read returns `materializeSharedSlot`'s box for a
  pointer and decodes anything else (2.2); a write unboxes a box to its cell (boxing.md, B.3) and
  stores through `SharedValue::set` with the runtime's `SharedMutator`; a delete is refused;
  `checkAllOwnIndexed` follows O-B6.
* `indexedStorage` belongs to the class and never changes once set (`ClassFlags::indexedStorage`),
  so a `SharedArray` box needs a root class of its own, chosen when the box is bound, inside the
  materialiser's window, or before the box is allocated (O-B7).
* `length` is named, not indexed: either a read hook that knows only `length`, or a slot for it in
  the `SharedArray` box class. A hook needs the read-cache guard (O-B8) even in the prototype.
* What a read past the end returns (`undefined`), what a write past the end does, and what a store
  of a value the prototype cannot hold does (2.2).

**Tests:** in one runtime, through JS: create, store and read numbers and nested arrays,
`a[0] === a[0]` for a nested array, reads and writes past the end, `delete`, `Object.freeze`; across
runtimes once 2.4 exists; under TSan, two runtimes storing to and reading one array.

**Size:** L. **Needs:** 1.1 and 2.2, and 2.3 for the generic paths a box reaches.

### 2.2 Non-pointer values in `SharedValue`

**Today.** A slot holds a cell pointer or 0, which means empty; there are no tags. Every load that
may be followed to a cell is at least an acquire, including `getPointerOrNull`, which the tracer
uses, and the seq_cst variant the barrier's pre-read uses.

**Work:**

* Encode `undefined`, `null`, booleans and numbers in the 64 bits, so that `getPointerOrNull` stays
  a single load and returns null for anything that is not a pointer (marking-implementation.md,
  open question 3). Empty stays distinct from `undefined`.
* Convert between `HermesValue` and `SharedValue` in both directions.
* Decide what storing a string, a symbol, a BigInt or a local object does in the prototype: throw a
  `TypeError`, or for strings, box them as Appendix D.1 plans.

**Tests:** a round trip of each kind, including NaN, -0 and the extreme doubles; the tracer and the
barrier skip a slot holding a number; a pointer overwritten by a number is still recorded.

**Size:** M.

### 2.3 Where the VM must tell a box apart

**Today.** There is no `isBoxedSharedObject()`, and the box's class sets none of `hostObject`,
`proxyObject` and `lazyObject`, so generic paths treat a box as an ordinary empty object and succeed
with the wrong answer (boxing.md, section 4). `PropertyFlags` has a free bit for the
`sharedObject` descriptor flag (section 1.6).

**Work:** add `isBoxedSharedObject()`, a `vmisa` test on the cell kind, and the descriptor flag.
Then take the Appendix C rows a `SharedArray` box reaches, each through the hooks or with a throw:
enumeration (`Object.keys`, `for`-`in`, spread), `JSON.stringify`, `structuredClone` (row 39),
`WeakMap` keys and `WeakRef` (row 40), `setNativeState` (row 37), and `Object.defineProperty` on an
index. boxing.md counts about 83 decision points among the existing `isHostObject()` and
`isProxyObject()` tests (O-B9); the rest wait for structs.

**Tests:** one per row covered, each failing before its fix.

**Size:** L.

### 2.4 Handing a cell to another runtime

**Today.** `postMessage` in the Worker extension, which is built only with `JSI_UNSTABLE`,
serializes through `jsi::ISerialization`, which `HermesRuntimeImpl` implements with
`vm::serialize_RJS` in `SerializedValue.cpp`. A box falls through to the ordinary-object case and
arrives as an empty object (Appendix C, row 39). A message waits in a queue for as long as the
receiver takes, while the sender may drop its box and prune the key (O-B15).

**Work:**

* A serialized record for a shared cell.
* A root for the cell while it is in transit: a strong reference to the sender's box, released on
  the sender's thread once delivery is known; or a group-wide table of cells in transit, which the
  root scan copies and whose removal pushes while marking is on, as the prune does (P6).
* A way for the receiver to box a cell it was handed: `materializeSharedSlot` boxes only what it
  reads from a slot.
* Releasing the root of a message that is never delivered, as when a worker terminates.

**Open:** which root; the table reopens "A separate root log" (boxing.md, section 5).

**Tests:** a sender drops its array right after posting it and cycles run before the receiver
reads it, which still sees the contents; a message never delivered releases its root; TSan.

**Size:** L. **Needs:** 1.3.

### 2.5 `Atomics` on shared slots

**Today.** Hermes has no `Atomics` object and no `SharedArrayBuffer`.

**Work:** `Atomics.load`, `store`, `exchange` and `compareExchange` on `SharedArray` elements.
`load` is a seq_cst load and `store` goes through `SharedValue::set`. `exchange` and
`compareExchange` run the store barrier's seq_cst pre-read inside a window and then a seq_cst
read-modify-write (marking.md, channel 10). Each boxes what it returns. `compareExchange` has to say
what it compares: the encoded bits compare cells by identity, but also tell +0 from -0. `Atomics` on
typed arrays is not needed for this.

**Tests:** two runtimes counting with `compareExchange` loops end with the exact total; `exchange`
returns the old element, boxed; TSan.

**Size:** M. **Needs:** 2.1 and 2.2.

### 2.6 Static Hermes and the JIT

**Today.** A typed checked cast to a class type tests only that a value is an object (`SH.cpp`), so
a box passes it, and `PrLoad` and `PrStore` then read and write the overlap slots that hold
`sharedCell_` (boxing.md, A.5). The arm64 JIT's indexed fast paths check for a `JSArray` or a
typed-array kind, so a box takes the slow path there; its property inline cache compares
`hiddenClassLazyJITId`, and `emitParentSpecialization` reads a parent's slot directly (A.4).
Generated code links only against a VM built with the same model string, which includes
`_sharedheap` when the flag is on.

**Work:**

* A box must never reach code typed with a user class: assert it in debug builds now, and check it
  in casts once they check classes.
* Run the prototype's tests under shermes, untyped and typed, and with the JIT forced on.
* The read-cache guard (O-B8), if 2.1 reads `length` through a hook.

**Size:** M.

### 2.7 Structs, later: the proposal revision and how JS defines a struct type

* Which revision of the TC39 structs proposal to follow, and which of its parts: struct types,
  `SharedArray`, and the locks recent revisions add (`Atomics.Mutex` and `Atomics.Condition`).
* Syntax (`shared struct`), which needs the parser, IRGen and the type checker; or an API object
  first, which needs none of them.
* A threat model. Every slot access is atomic, so races between runtimes are not data races in
  C++. What one runtime can still do to another: exhaust the budget, so that the others get
  `RangeError`, and hold windows open, which delays cycles.

**Size:** M, mostly design.

### 2.8 The per-type field map and per-runtime symbolization (O-B13)

* A type cell in the shared heap that holds a struct type's field names in slot order, immutable
  once created. Instances point to it.
* Each runtime maps a pair of type and `SymbolID` to a slot, filled on first use, since a
  `SymbolID` means something only in its own runtime.
* Where the names live: as shared strings, which needs Appendix D, or copied into each runtime when
  it first meets the type.

**Size:** L. **Needs:** 2.7.

### 2.9 Named-property hooks (boxing.md, Appendices A and B)

The read hook (A.1) and the write hook (B.1), keyed off the `sharedObject` descriptor flag (section
1.6); the read-cache guard (O-B8); the prototype walk (A.1); `writable` (O-B6); and enumeration,
which three places implement (O-B5, C.3). Then the rest of Appendix C.

**Size:** L. **Needs:** 2.8.

### 2.10 Shared strings (boxing.md, Appendix D)

v1 boxes a shared string and unboxes it at use (D.1). That leaves three sites C++ cannot reach: the
JIT's inline tag tests for `typeof` and `===`, and JSI's `isString()`, which out-of-tree code reads.
The end state is a Str-tagged view cell (D.2). D.6's memory arithmetic decides between a view and a
copy, perhaps by length, and BigInt is open.

**Size:** L.

## Step 3: hardening and performance

### 3.1 Measure on arm64 devices

The store barrier (a window's entry and exit, a seq_cst pre-read and a seq_cst store), the rings
under contention, allocation under `mtx_` while other runtimes allocate, the collector's latency,
and the box map's memory per runtime. C++ microbenchmarks first, then the prototype's workloads.

### 3.2 Stress and fuzzing

Soak tests with several runtimes in `HERMES_SLOW_DEBUG` builds, so that the post-certification
checks run at every cycle; a JS fuzzing mode for `SharedArray`, and later structs, once Step 2
lands.

### 3.3 Model-check the code

The GenMC models check the design, with one or two cells, three or four runtimes and at most two
cycles (marking.md, "Open questions"). Check the barrier, the in-flight check and the materialiser
as written, with GenMC or a comparable checker.

### 3.4 Memory

* Retention is peak-live per bucket (allocator.md, open question 3); check that bet on a real
  workload.
* The bucket constants (open question 2).
* One mapping per region: carve segments from larger reservations, as Hades does, if the mapping
  count or the syscalls under `mtx_` show up.
* Work under `mtx_` (open question 8): a map insert can rehash and allocate under it.

### 3.5 Allocation throughput

If `mtx_` shows up in profiles: per-runtime allocation buffers with black allocation per chunk
(marking.md, "Future options").

## Milestones

1. **Reviewable core:** Step 0.
2. **A group usable from C++:** 1.1 to 1.3, 1.6, 1.7, 1.11 and 1.12. A host can create a group, put
   runtimes and workers in it and read its numbers; a refused allocation throws cleanly; CI builds
   and tests the flag on.
3. **The `SharedArray` prototype:** 2.1 to 2.6. Two workers share a `SharedArray` and update it with
   `Atomics`, in the interpreter, in shermes and with the JIT; CI runs it under TSan.
4. **Structs:** 2.7 to 2.10.
5. **Production:** Step 3, with 1.4, 1.5 and 1.8 to 1.10.

## Order

* 1.2, then 1.3, then 2.4.
* 1.1 and 2.2, then 2.1, then 2.5 and 2.6; 2.3 grows with 2.1.
* The prototype, then 1.10 and 3.1, which need its workloads.
* 2.7, then 2.8, then 2.9. 2.10 can start once 2.3 is done.

Review of the first parts can start alongside 0.1.

## Open decisions

* 0.3: where the design docs land, and how the review is split.
* 1.1: whether a refused runtime runs the cycle itself or waits for the thread, and how many
  attempts it makes.
* 1.2: a `RuntimeConfig` field or a factory parameter.
* 1.5: accept the barrier worklist's growth, or bound it.
* 1.9: which guard release builds get, if any.
* 2.1: how `length` is read, and when a `SharedArray` box gets its class (O-B7).
* 2.2: what storing a string does in the prototype.
* 2.4: which root holds a cell in transit.
* 2.7: the proposal revision, and syntax or an API.
* 2.10: a view or a copy for strings (D.6).
