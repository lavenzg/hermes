/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "loop1-jit.h"

#include "asmjit/a64.h"

#include "hermes/Support/ErrorHandling.h"
#include "hermes/VM/static_h.h"

#include "llvh/Support/raw_ostream.h"

#include <deque>
#include <functional>

namespace a64 = asmjit::a64;

static asmjit::JitRuntime s_jitRT;

// x19 is runtime
static constexpr auto xRuntime = a64::x19;
// x20 is frame
static constexpr auto xFrame = a64::x20;
// x0 < xDoubleLim means that it is a double.
//    cmp   x0, xDoubleLim
//    b.hs  slowPath
static constexpr auto xDoubleLim = a64::x21;

class ErrorHandler : public asmjit::ErrorHandler {
  virtual void handleError(
      asmjit::Error err,
      const char *message,
      asmjit::BaseEmitter *origin) override {
    llvh::errs() << "AsmJit error: " << err << ": "
                 << asmjit::DebugUtils::errorAsString(err) << ": " << message
                 << "\n";
    hermes::hermes_fatal("AsmJit error");
  }
};

#define EMIT_RUNTIME_CALL(em, func) (em).call(offsetof(SHRuntime, func), #func)

class Emitter {
  std::unique_ptr<asmjit::FileLogger> fileLogger_{};
  asmjit::FileLogger *logger_ = nullptr;
  ErrorHandler errorHandler_;
  std::deque<std::function<void()>> slowPaths_{};

 public:
  asmjit::CodeHolder code{};
  a64::Assembler a{};

  explicit Emitter(asmjit::JitRuntime &jitRT) {
    fileLogger_ = std::make_unique<asmjit::FileLogger>(stdout);
    if (fileLogger_)
      logger_ = fileLogger_.get();

    if (logger_)
      logger_->setIndentation(asmjit::FormatIndentationGroup::kCode, 4);

    code.init(jitRT.environment(), jitRT.cpuFeatures());
    code.setErrorHandler(&errorHandler_);
    if (logger_)
      code.setLogger(logger_);
    code.attach(&a);
  }

  /// Log a comment.
  /// Annotated with printf-style format.
  void comment(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
    if (!logger_)
      return;
    va_list args;
    va_start(args, fmt);
    char buf[80];
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    a.comment(buf);
  }

  JitFn addToRuntime() {
    emitSlowPaths();

    code.detach(&a);
    JitFn fn;
    asmjit::Error err = s_jitRT.add(&fn, &code);
    if (err) {
      llvh::errs() << "AsmJit failed: "
                   << asmjit::DebugUtils::errorAsString(err) << "\n";
      hermes::hermes_fatal("AsmJit failed");
    }
    return fn;
  }

  void frameSetup() {
    a.sub(a64::sp, a64::sp, 10 * 8);
    //  0-3: SHLocals
    //  4: x22
    //  5: x21
    //  6: x20
    //  7: x19
    //  8: x29 <- new x29 points here
    //  9: x30

    a.stp(a64::x22, a64::x21, a64::Mem(a64::sp, 4 * 8));
    a.stp(a64::x20, a64::x19, a64::Mem(a64::sp, 6 * 8));
    a.stp(a64::x29, a64::x30, a64::Mem(a64::sp, 8 * 8));
    a.add(a64::x29, a64::sp, 8 * 8);
  }

  void leaveFrame() {
    comment("// leaveFrame");
    a.ldp(a64::x29, a64::x30, a64::Mem(a64::sp, 8 * 8));
    a.ldp(a64::x20, a64::x19, a64::Mem(a64::sp, 6 * 8));
    a.ldp(a64::x22, a64::x21, a64::Mem(a64::sp, 4 * 8));
    a.add(a64::sp, a64::sp, 10 * 8);
    a.ret(a64::x30);
  }

  void call(int32_t ofs, const char *name) {
    comment("// call %s", name);
    a.ldr(a64::x8, a64::Mem(xRuntime, ofs));
    a.blr(a64::x8);
  }

  /// Create an a64::Mem to a specifc frame register.
  static constexpr inline a64::Mem frameReg(uint32_t index) {
    // FIXME: check if the offset fits
    return a64::Mem(xFrame, index * sizeof(SHLegacyValue));
  }

  template <typename R>
  void loadFrame(R dest, uint32_t rFrom) {
    // FIXME: check if the offset fits
    a.ldr(dest, frameReg(rFrom));
  }
  template <typename R>
  void storeFrame(R src, uint32_t rFrom) {
    // FIXME: check if the offset fits
    a.str(src, frameReg(rFrom));
  }

  void movFrameAddr(a64::GpX dst, uint32_t frameReg) {
    // FIXME: check range of frameReg * 8
    if (frameReg == 0)
      a.mov(dst, xFrame);
    else
      a.add(dst, xFrame, frameReg * sizeof(SHLegacyValue));
  }

  void loadParam(uint32_t rRes, uint32_t paramIndex) {
    comment("// LoadParam r%u, %u", rRes, paramIndex);
    a.mov(a64::x0, xFrame);
    a.mov(a64::w1, paramIndex);
    EMIT_RUNTIME_CALL(*this, _sh_ljs_param);
    storeFrame(a64::x0, rRes);
  }

  void mov(uint32_t rRes, uint32_t rInput) {
    comment("// %s r%u, r%u", "mov", rRes, rInput);
    loadFrame(a64::x0, rInput);
    storeFrame(a64::x0, rRes);
  }

  template <typename FastPath>
  void binOp(
      uint32_t rRes,
      uint32_t rLeft,
      uint32_t rRight,
      const char *name,
      FastPath fast,
      int32_t slowCallOfs,
      const char *slowCallName) {
    //  ldr x0, [x20]
    //  cmp x0, x21
    //  b.hi    SLOW_1
    //
    //  fmov d0, x0
    //  fmov d1, #-1.0
    //  fadd d0, d0, d1
    //  str d0, [x20, 3*8]
    //  CONT_1:

    comment("// %s r%u, r%u, r%u", name, rRes, rLeft, rRight);
    asmjit::Label slowPathLab = newSlowPathLabel();
    asmjit::Label contLab = newContLabel();

    loadFrame(a64::x0, rLeft);
    a.cmp(a64::x0, xDoubleLim);
    a.b_hs(slowPathLab);
    loadFrame(a64::x1, rRight);
    a.cmp(a64::x1, xDoubleLim);
    a.b_hs(slowPathLab);

    a.fmov(a64::d0, a64::x0);
    a.fmov(a64::d1, a64::x1);
    fast(a64::d0, a64::d0, a64::d1);
    storeFrame(a64::d0, rRes);
    a.bind(contLab);

    //  SLOW_3:
    //  // frame[1] = _sh_ljs_mul_rjs(shr, frame + 1, frame + 3);
    //  mov x0, x19
    //  add x1, x20, 1*8
    //  add x2, x20, 3*8
    //  bl __sh_ljs_mul_rjs
    //  str x0, [x20, 1*8]
    //  b CONT_3
    slowPaths_.emplace_back([this,
                             name,
                             rRes,
                             rLeft,
                             rRight,
                             slowPathLab,
                             contLab,
                             slowCallOfs,
                             slowCallName]() {
      comment("// %s r%u, r%u, r%u", name, rRes, rLeft, rRight);
      a.bind(slowPathLab);
      a.mov(a64::x0, xRuntime);
      movFrameAddr(a64::x1, rLeft);
      movFrameAddr(a64::x2, rRight);
      call(slowCallOfs, slowCallName);
      storeFrame(a64::x0, rRes);
      a.b(contLab);
    });
  }

  void mul(uint32_t rRes, uint32_t rLeft, uint32_t rRight) {
    binOp(
        rRes,
        rLeft,
        rRight,
        "mul",
        [this](a64::VecD res, a64::VecD dl, a64::VecD dr) {
          a.fmul(res, dl, dr);
        },
        offsetof(SHRuntime, _sh_ljs_mul_rjs),
        "_sh_ljs_mul_rjs");
  }

  template <typename FastPath>
  void unop(
      uint32_t rRes,
      uint32_t rInput,
      const char *name,
      FastPath fast,
      int32_t slowCallOfs,
      const char *slowCallName) {
    //  ldr x0, [x20]
    //  cmp x0, x21
    //  b.hi    SLOW_1
    //
    //  fmov d0, x0
    //  fmov d1, #-1.0
    //  fadd d0, d0, d1
    //  str d0, [x20, 3*8]
    //  CONT_1:

    comment("// %s r%u, r%u", name, rRes, rInput);
    asmjit::Label slowPathLab = newSlowPathLabel();
    asmjit::Label contLab = newContLabel();

    loadFrame(a64::x0, rInput);
    a.cmp(a64::x0, xDoubleLim);
    a.b_hs(slowPathLab);

    a.fmov(a64::d0, a64::x0);
    fast(a64::d0, a64::d1);
    storeFrame(a64::d0, rRes);
    a.bind(contLab);

    // frame[3] = _sh_ljs_dec_rjs(shr, frame + 0);
    //  SLOW_1:
    //  mov x0, x19     ; runtime
    //  mov x1, x20     ; frame + 0
    //  bl  __sh_ljs_dec_rjs
    //  str x0, [x20, 3*8]  ; frame[3] =
    //  b CONT_1
    slowPaths_.emplace_back([this,
                             name,
                             rRes,
                             rInput,
                             slowPathLab,
                             contLab,
                             slowCallOfs,
                             slowCallName]() {
      comment("// Slow path: %s r%u, r%u", name, rRes, rInput);
      a.bind(slowPathLab);
      a.mov(a64::x0, xRuntime);
      movFrameAddr(a64::x1, rInput);
      call(slowCallOfs, slowCallName);
      storeFrame(a64::x0, rRes);
      a.b(contLab);
    });
  }

  void dec(uint32_t rRes, uint32_t rInput) {
    unop(
        rRes,
        rInput,
        "dec",
        [this](a64::VecD d, a64::VecD tmp) {
          a.fmov(tmp, -1.0);
          a.fadd(d, d, tmp);
        },
        offsetof(SHRuntime, _sh_ljs_dec_rjs),
        "_sh_ljs_dec_rjs");
  }
  void inc(uint32_t rRes, uint32_t rInput) {
    unop(
        rRes,
        rInput,
        "inc",
        [this](a64::VecD d, a64::VecD tmp) {
          a.fmov(tmp, 1.0);
          a.fadd(d, d, tmp);
        },
        offsetof(SHRuntime, _sh_ljs_inc_rjs),
        "_sh_ljs_inc_rjs");
  }

  void jGreater(
      bool invert,
      uint32_t rLeft,
      uint32_t rRight,
      const asmjit::Label &target) {
    // if (!_sh_ljs_greater_rjs(shr, frame + 3, frame + 2)) goto L1;
    //  ldr x0, [x20, 3*8]
    //  cmp x0, x21
    //  b.hi    SLOW_2
    //  ldr x1, [x20, 2*8]
    //  cmp x1, x21
    //  b.hi    SLOW_2
    //
    //  fmov    d0, x0
    //  fmov    d1, x1
    //  fcmp d0, d1
    //  b.gt CONT_2
    //  b L1
    //  CONT_2:

    comment(
        "// %s Lx, r%u, r%u", invert ? "jNGreater" : "jGreater", rLeft, rRight);
    asmjit::Label slowPathLab = newSlowPathLabel();
    asmjit::Label contLab = newContLabel();

    loadFrame(a64::x0, rLeft);
    a.cmp(a64::x0, xDoubleLim);
    a.b_hs(slowPathLab);
    loadFrame(a64::x1, rRight);
    a.cmp(a64::x1, xDoubleLim);
    a.b_hs(slowPathLab);

    a.fmov(a64::d0, a64::x0);
    a.fmov(a64::d1, a64::x1);
    a.fcmp(a64::d0, a64::d1);
    if (!invert) {
      a.b_gt(target);
    } else {
      a.b_gt(contLab);
      a.b(target);
    }
    a.bind(contLab);

    //  SLOW_2:
    //  if (!_sh_ljs_greater_rjs(shr, frame + 3, frame + 2)) goto L1;
    //  mov x0, x19
    //  add x1, x20, 3*8
    //  add x2, x20, 2*8
    //  bl __sh_ljs_greater_rjs
    //  cbz w0, L1
    //  b CONT_2
    //
    slowPaths_.emplace_back(
        [this, invert, target, rLeft, rRight, slowPathLab, contLab]() {
          comment(
              "// Slow path: %s Lx, r%u, r%u",
              invert ? "jNGreater" : "jGreater",
              rLeft,
              rRight);
          a.bind(slowPathLab);
          a.mov(a64::x0, xRuntime);
          movFrameAddr(a64::x1, rLeft);
          movFrameAddr(a64::x2, rRight);
          EMIT_RUNTIME_CALL(*this, _sh_ljs_greater_rjs);
          if (!invert) {
            a.cbz(a64::w0, target);
            a.b(contLab);
          } else {
            a.cbz(a64::w0, contLab);
            a.b(target);
          }
        });
  }

  void loadConstUInt8(uint32_t rRes, uint8_t val) {
    comment("// LoadConstUInt8 r%u, %u", rRes, val);

    if (a64::Utils::isFP64Imm8((double)val)) {
      a.fmov(a64::d0, val);
    } else {
      // TODO: use a constant pool.
      a.mov(a64::w0, val);
      a.ucvtf(a64::d0, a64::w0);
    }
    a.str(a64::d0, frameReg(rRes));
  }

 private:
  asmjit::Label newPrefLabel(const char *pref, size_t index) {
    char buf[8];
    snprintf(buf, sizeof(buf), "%s%lu", pref, index);
    return a.newNamedLabel(buf);
  }
  inline asmjit::Label newSlowPathLabel() {
    return newPrefLabel("SLOW_", slowPaths_.size());
  }
  inline asmjit::Label newContLabel() {
    return newPrefLabel("CONT_", slowPaths_.size());
  }

  void emitSlowPaths() {
    while (!slowPaths_.empty()) {
      slowPaths_.front()();
      slowPaths_.pop_front();
    }
  }
};

JitFn compile_loop1(void) {
  Emitter em(s_jitRT);
  a64::Assembler &a = em.a;

  em.frameSetup();

  em.comment("// xRuntime");
  a.mov(xRuntime, a64::x0);

  //  _sh_check_native_stack_overflow(shr);
  EMIT_RUNTIME_CALL(em, _sh_check_native_stack_overflow);

  //  SHLegacyValue *frame = _sh_enter(shr, &locals.head, 6);
  em.comment("// _sh_enter");
  a.mov(a64::x0, xRuntime);
  a.mov(a64::x1, a64::sp);
  a.mov(a64::w2, 6);
  EMIT_RUNTIME_CALL(em, _sh_enter);
  em.comment("// xFrame");
  a.mov(xFrame, a64::x0);

  //  locals.head.count = 0;
  em.comment("// locals.head.count = 0");
  a.mov(a64::w1, 0);
  a.str(a64::w1, a64::Mem(a64::sp, offsetof(SHLocals, count)));

  // ((uint64_t)HVTag_First << kHV_NumDataBits)
  em.comment("// xDoubleLim");
  a.mov(xDoubleLim, ((uint64_t)HVTag_First << kHV_NumDataBits));

  // frame[0] = _sh_ljs_param(frame, 1);
  em.loadParam(0, 1);

  // frame[3] = _sh_ljs_dec_rjs(shr, frame + 0);
  em.dec(3, 0);

  // frame[2] = _sh_ljs_double(1);
  em.loadConstUInt8(2, 1);

  // frame[1] = frame[0];
  em.mov(1, 0);
  // frame[0] = frame[1];
  em.mov(0, 1);

  asmjit::Label L1 = a.newLabel();
  em.jGreater(true, 3, 2, L1);

  //  L2:
  asmjit::Label L2 = a.newLabel();
  a.bind(L2);

  // frame[1] = _sh_ljs_mul_rjs(shr, frame + 1, frame + 3);
  em.mul(1, 1, 3);

  // frame[3] = _sh_ljs_dec_rjs(shr, frame + 3);
  em.dec(3, 3);

  // frame[0] = frame[1];
  em.mov(0, 1);

  // if (_sh_ljs_greater_rjs(shr, frame + 3, frame + 2)) goto L2;
  em.jGreater(false, 3, 2, L2);

  //  L1:
  a.bind(L1);

  // SHLegacyValue tmp = frame[0];
  em.loadFrame(a64::x22, 0);

  // _sh_leave(shr, &locals.head, frame);
  a.mov(a64::x0, xRuntime);
  a.mov(a64::x1, a64::sp);
  a.mov(a64::x2, xFrame);
  EMIT_RUNTIME_CALL(em, _sh_leave);

  a.mov(a64::x0, a64::x22);
  em.leaveFrame();

  return em.addToRuntime();
}