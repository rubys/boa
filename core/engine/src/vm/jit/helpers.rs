//! Runtime helper functions callable from JIT-compiled code.
//!
//! These are `extern "C"` functions that the JIT emits calls to. They bridge
//! between the native code and Boa's runtime, operating on `Context` and its
//! register file. Each helper corresponds to one or more bytecode opcodes.
//!
//! As the JIT matures, many of these will be inlined into the generated code.
//! For now, calling back into Rust is the simplest correct approach.

use crate::{Context, JsValue, value::JsVariant};

/// Increment the refcount for a GC'd JsValue given its raw NaN-boxed u64.
///
/// Called by the JIT after copying a pointer-typed value to a new register.
/// The caller has already checked that the tag indicates a pointer type.
pub(super) extern "C" fn jit_clone_value(raw: u64) {
    // Reconstruct the JsValue from the raw bits, clone it (bumps refcount),
    // then forget both copies to avoid decrementing.
    let val = unsafe { std::mem::transmute::<u64, JsValue>(raw) };
    let _cloned = val.clone();
    std::mem::forget(val);
    std::mem::forget(_cloned);
    // Net effect: refcount += 1 (clone increments, neither drop decrements).
}

/// Decrement the refcount for a GC'd JsValue given its raw NaN-boxed u64.
///
/// Called by the JIT when overwriting a register that held a pointer-typed value.
pub(super) extern "C" fn jit_drop_value(raw: u64) {
    // Reconstruct the JsValue and let it drop normally (decrements refcount).
    let _val = unsafe { std::mem::transmute::<u64, JsValue>(raw) };
    // _val drops here, decrementing the refcount.
}

/// Set register `dst` to integer 0.
///
/// Implements: `StoreZero { dst }`
pub(super) extern "C" fn jit_store_zero(ctx: &mut Context, dst: u32) {
    ctx.vm.set_register(dst as usize, JsValue::from(0));
}

/// Set register `dst` to integer 1.
///
/// Implements: `StoreOne { dst }`
pub(super) extern "C" fn jit_store_one(ctx: &mut Context, dst: u32) {
    ctx.vm.set_register(dst as usize, JsValue::from(1));
}

/// Set register `dst` to an i8 value.
///
/// Implements: `StoreInt8 { dst, value }`
pub(super) extern "C" fn jit_store_int8(ctx: &mut Context, dst: u32, value: i8) {
    ctx.vm
        .set_register(dst as usize, JsValue::from(i32::from(value)));
}

/// Set register `dst` to an i16 value.
///
/// Implements: `StoreInt16 { dst, value }`
pub(super) extern "C" fn jit_store_int16(ctx: &mut Context, dst: u32, value: i16) {
    ctx.vm
        .set_register(dst as usize, JsValue::from(i32::from(value)));
}

/// Set register `dst` to an i32 value.
///
/// Implements: `StoreInt32 { dst, value }`
pub(super) extern "C" fn jit_store_int32(ctx: &mut Context, dst: u32, value: i32) {
    ctx.vm.set_register(dst as usize, JsValue::from(value));
}

/// Copy register `src` to register `dst`.
///
/// Implements: `Move { dst, src }`
pub(super) extern "C" fn jit_move(ctx: &mut Context, dst: u32, src: u32) {
    let value = ctx.vm.get_register(src as usize).clone();
    ctx.vm.set_register(dst as usize, value);
}

/// Set the accumulator (implicit return value) from a register.
///
/// Implements: `SetAccumulator { src }`
pub(super) extern "C" fn jit_set_accumulator(ctx: &mut Context, src: u32) {
    let value = ctx.vm.get_register(src as usize).clone();
    ctx.vm.set_return_value(value);
}

/// Push a register value onto the stack.
///
/// Implements: `PushFromRegister { src }`
pub(super) extern "C" fn jit_push_from_register(ctx: &mut Context, src: u32) {
    let value = ctx.vm.get_register(src as usize).clone();
    ctx.vm.stack.push(value);
}

/// Pop a value from the stack into a register.
///
/// Implements: `PopIntoRegister { dst }`
pub(super) extern "C" fn jit_pop_into_register(ctx: &mut Context, dst: u32) {
    let value = ctx.vm.stack.pop();
    ctx.vm.set_register(dst as usize, value);
}

/// Get the i-th function argument and store it in register `dst`.
///
/// Implements: `GetArgument { index, dst }`
pub(super) extern "C" fn jit_get_argument(ctx: &mut Context, index: u32, dst: u32) {
    let value = ctx
        .vm
        .stack
        .get_argument(ctx.vm.frame(), index as usize)
        .cloned()
        .unwrap_or_default();
    ctx.vm.set_register(dst as usize, value);
}

/// Binary `+` operator. Returns 0 on success, 1 on exception.
///
/// Implements: `Add { dst, lhs, rhs }`
pub(super) extern "C" fn jit_add(ctx: &mut Context, dst: u32, lhs: u32, rhs: u32) -> u64 {
    let l = ctx.vm.get_register(lhs as usize);
    let r = ctx.vm.get_register(rhs as usize);

    // Fast path: try numeric add without cloning.
    if let Some(value) = JsValue::add_fast(l, r) {
        ctx.vm.set_register(dst as usize, value.into());
        return 0;
    }

    // Slow path: full type coercion.
    let l = l.clone();
    let r = r.clone();
    match l.add(&r, ctx) {
        Ok(value) => {
            ctx.vm.set_register(dst as usize, value.into());
            0
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
    }
}

/// Binary `-` operator. Returns 0 on success, 1 on exception.
///
/// Implements: `Sub { dst, lhs, rhs }`
pub(super) extern "C" fn jit_sub(ctx: &mut Context, dst: u32, lhs: u32, rhs: u32) -> u64 {
    let l = ctx.vm.get_register(lhs as usize);
    let r = ctx.vm.get_register(rhs as usize);

    if let Some(value) = JsValue::sub_fast(l, r) {
        ctx.vm.set_register(dst as usize, value.into());
        return 0;
    }

    let l = l.clone();
    let r = r.clone();
    match l.sub(&r, ctx) {
        Ok(value) => {
            ctx.vm.set_register(dst as usize, value.into());
            0
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
    }
}

/// Binary `*` operator. Returns 0 on success, 1 on exception.
pub(super) extern "C" fn jit_mul(ctx: &mut Context, dst: u32, lhs: u32, rhs: u32) -> u64 {
    let l = ctx.vm.get_register(lhs as usize);
    let r = ctx.vm.get_register(rhs as usize);

    if let Some(value) = JsValue::mul_fast(l, r) {
        ctx.vm.set_register(dst as usize, value.into());
        return 0;
    }

    let l = l.clone();
    let r = r.clone();
    match l.mul(&r, ctx) {
        Ok(value) => {
            ctx.vm.set_register(dst as usize, value.into());
            0
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
    }
}

/// Binary `|` operator. Returns 0 on success, 1 on exception.
pub(super) extern "C" fn jit_bit_or(ctx: &mut Context, dst: u32, lhs: u32, rhs: u32) -> u64 {
    let l = ctx.vm.get_register(lhs as usize);
    let r = ctx.vm.get_register(rhs as usize);

    if let Some(value) = JsValue::bitor_fast(l, r) {
        ctx.vm.set_register(dst as usize, value.into());
        return 0;
    }

    let l = l.clone();
    let r = r.clone();
    match l.bitor(&r, ctx) {
        Ok(value) => {
            ctx.vm.set_register(dst as usize, value.into());
            0
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
    }
}

/// Macro to generate binary op helpers with fast path.
macro_rules! binop_helper {
    ($name:ident, $fast_fn:ident, $slow_fn:ident) => {
        pub(super) extern "C" fn $name(
            ctx: &mut Context,
            dst: u32,
            lhs: u32,
            rhs: u32,
        ) -> u64 {
            let l = ctx.vm.get_register(lhs as usize);
            let r = ctx.vm.get_register(rhs as usize);

            if let Some(value) = JsValue::$fast_fn(l, r) {
                ctx.vm.set_register(dst as usize, value.into());
                return 0;
            }

            let l = l.clone();
            let r = r.clone();
            match l.$slow_fn(&r, ctx) {
                Ok(value) => {
                    ctx.vm.set_register(dst as usize, value.into());
                    0
                }
                Err(err) => {
                    ctx.vm.pending_exception = Some(err);
                    1
                }
            }
        }
    };
}

binop_helper!(jit_div, div_fast, div);
binop_helper!(jit_mod, rem_fast, rem);
binop_helper!(jit_pow, pow_fast, pow);
binop_helper!(jit_bit_and, bitand_fast, bitand);
binop_helper!(jit_bit_xor, bitxor_fast, bitxor);
binop_helper!(jit_shl, shl_fast, shl);
binop_helper!(jit_shr, shr_fast, shr);
binop_helper!(jit_ushr, ushr_fast, ushr);
binop_helper!(jit_lt, lt_fast, lt);
binop_helper!(jit_le, le_fast, le);
binop_helper!(jit_gt, gt_fast, gt);
binop_helper!(jit_ge, ge_fast, ge);
binop_helper!(jit_eq, equals_fast, equals);
binop_helper!(jit_ne, not_equals_fast, not_equals);

/// `StrictEq` — infallible, no slow path needed.
pub(super) extern "C" fn jit_strict_eq(ctx: &mut Context, dst: u32, lhs: u32, rhs: u32) {
    let l = ctx.vm.get_register(lhs as usize);
    let r = ctx.vm.get_register(rhs as usize);
    let result = l.strict_equals(r);
    ctx.vm.set_register(dst as usize, JsValue::from(result));
}

/// `StrictNotEq` — infallible.
pub(super) extern "C" fn jit_strict_ne(ctx: &mut Context, dst: u32, lhs: u32, rhs: u32) {
    let l = ctx.vm.get_register(lhs as usize);
    let r = ctx.vm.get_register(rhs as usize);
    let result = !l.strict_equals(r);
    ctx.vm.set_register(dst as usize, JsValue::from(result));
}

/// Unary `--` operator. Returns 0 on success, 1 on exception.
pub(super) extern "C" fn jit_dec(ctx: &mut Context, dst: u32, src: u32) -> u64 {
    let value = ctx.vm.take_register(src as usize);

    match value.variant() {
        JsVariant::Integer32(number) if number > i32::MIN => {
            ctx.vm.set_register(src as usize, JsValue::from(number));
            ctx.vm.set_register(dst as usize, JsValue::from(number - 1));
            0
        }
        _ => match value.to_numeric(ctx) {
            Ok(crate::value::Numeric::Number(number)) => {
                ctx.vm.set_register(src as usize, JsValue::from(number));
                ctx.vm
                    .set_register(dst as usize, JsValue::from(number - 1f64));
                0
            }
            Ok(crate::value::Numeric::BigInt(bigint)) => {
                ctx.vm
                    .set_register(src as usize, JsValue::from(bigint.clone()));
                ctx.vm.set_register(
                    dst as usize,
                    JsValue::from(crate::JsBigInt::sub(&bigint, &crate::JsBigInt::one())),
                );
                0
            }
            Err(err) => {
                ctx.vm.pending_exception = Some(err);
                1
            }
        },
    }
}

/// Unary `++` operator. Returns 0 on success, 1 on exception.
///
/// Implements: `Inc { dst, src }`
/// Writes the original value back to `src` and the incremented value to `dst`.
pub(super) extern "C" fn jit_inc(ctx: &mut Context, dst: u32, src: u32) -> u64 {
    let value = ctx.vm.take_register(src as usize);

    match value.variant() {
        JsVariant::Integer32(number) if number < i32::MAX => {
            ctx.vm.set_register(src as usize, JsValue::from(number));
            ctx.vm.set_register(dst as usize, JsValue::from(number + 1));
            0
        }
        _ => match value.to_numeric(ctx) {
            Ok(crate::value::Numeric::Number(number)) => {
                ctx.vm.set_register(src as usize, JsValue::from(number));
                ctx.vm
                    .set_register(dst as usize, JsValue::from(number + 1f64));
                0
            }
            Ok(crate::value::Numeric::BigInt(bigint)) => {
                ctx.vm
                    .set_register(src as usize, JsValue::from(bigint.clone()));
                ctx.vm.set_register(
                    dst as usize,
                    JsValue::from(crate::JsBigInt::add(&bigint, &crate::JsBigInt::one())),
                );
                0
            }
            Err(err) => {
                ctx.vm.pending_exception = Some(err);
                1
            }
        },
    }
}

/// Increment the loop iteration counter. Returns 0 on success, 1 on limit exceeded.
///
/// Implements: `IncrementLoopIteration`
pub(super) extern "C" fn jit_increment_loop_iteration(ctx: &mut Context) -> u64 {
    let frame = ctx.vm.frame_mut();
    frame.loop_iteration_count += 1;
    let limit = ctx.vm.runtime_limits.loop_iteration_limit();
    if limit > 0 && ctx.vm.frame().loop_iteration_count > limit {
        ctx.vm.pending_exception = Some(
            crate::error::RuntimeLimitError::LoopIteration.into(),
        );
        1
    } else {
        0
    }
}

/// Compare `lhs < rhs` for JumpIfNotLessThan. Returns 1 if lhs >= rhs (should jump), 0 if lhs < rhs.
/// Returns 2 on error.
///
/// Implements the condition check for: `JumpIfNotLessThan { lhs, rhs, address }`
pub(super) extern "C" fn jit_not_less_than(ctx: &mut Context, lhs: u32, rhs: u32) -> u64 {
    let l = ctx.vm.get_register(lhs as usize);
    let r = ctx.vm.get_register(rhs as usize);

    if let Some(result) = JsValue::lt_fast(l, r) {
        return if result { 0 } else { 1 };
    }

    let l = l.clone();
    let r = r.clone();
    match l.lt(&r, ctx) {
        Ok(result) => {
            if result { 0 } else { 1 }
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            2
        }
    }
}

/// `GetNameGlobal` — look up a global binding. Returns 0 on success, 1 on exception.
///
/// Implements: `GetNameGlobal { dst, binding_index, ic_index }`
pub(super) extern "C" fn jit_get_name_global(
    ctx: &mut Context,
    dst: u32,
    binding_index: u32,
    _ic_index: u32,
) -> u64 {
    // Simplified version: look up the binding via the runtime.
    // TODO: use inline cache (ic_index) for faster lookups.
    let mut binding_locator =
        ctx.vm.frame().code_block.bindings[binding_index as usize].clone();

    if let Err(err) = ctx.find_runtime_binding(&mut binding_locator) {
        ctx.vm.pending_exception = Some(err);
        return 1;
    }

    match ctx.get_binding(&binding_locator) {
        Ok(value) => {
            ctx.vm.set_register(dst as usize, value.unwrap_or_default());
            0
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
    }
}

/// `Call` — call a function and run it to completion. Returns 0 on success, 1 on exception.
///
/// This helper pushes the callee frame, executes it (which may trigger JIT
/// compilation of the callee via the pc==0 check in the interpreter loop),
/// and returns after the callee completes. The result is on the stack.
pub(super) extern "C" fn jit_call(ctx: &mut Context, argument_count: u32) -> u64 {
    let func = ctx
        .vm
        .stack
        .calling_convention_get_function(argument_count as usize);

    let Some(object) = func.as_object() else {
        ctx.vm.pending_exception = Some(
            crate::JsNativeError::typ()
                .with_message("not a callable function")
                .into(),
        );
        return 1;
    };

    // resolve() sets up the frame. If it returns Ok(true) = Complete,
    // the result is already on the stack (native function).
    // If Ok(false) = Ready, we need to run the frame.
    match object.__call__(argument_count as usize).resolve(ctx) {
        Ok(true) => {
            // Native function completed, result on stack.
            return 0;
        }
        Ok(false) => {
            // Frame pushed, need to run it.
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            return 1;
        }
    }

    // Try to run the callee via JIT. This handles:
    // 1. Already compiled → call JIT function directly (native call!)
    // 2. Pending + threshold reached → compile and call
    // 3. Pending + below threshold → fall through to interpreter
    // 4. Unsupported → fall through to interpreter
    #[cfg(feature = "jit")]
    {
        use crate::vm::code_block::JitState;

        let code = ctx.vm.frame().code_block.clone();
        let state = code.jit.get();

        let jit_fn = match state {
            JitState::Compiled(jit_fn) => Some(jit_fn),
            JitState::Pending { call_count } => {
                let new_count = call_count + 1;
                if new_count >= crate::Context::JIT_THRESHOLD {
                    let compiler = ctx
                        .vm
                        .jit_compiler
                        .get_or_insert_with(|| {
                            super::JitCompiler::new().expect("JIT compiler should initialize")
                        });
                    match compiler.compile(&code) {
                        Some(f) => {
                            code.jit.set(JitState::Compiled(f));
                            Some(f)
                        }
                        None => {
                            code.jit.set(JitState::Unsupported);
                            None
                        }
                    }
                } else {
                    code.jit.set(JitState::Pending { call_count: new_count });
                    None
                }
            }
            JitState::Unsupported => None,
        };

        if let Some(jit_fn) = jit_fn {
            // Direct native call to the callee's JIT function!
            let rp = ctx.vm.frame().rp as usize;
            let reg_base = ctx.vm.stack.stack[rp..].as_mut_ptr().cast::<u64>();
            let tag = unsafe { jit_fn.call_raw(ctx as *mut Context, reg_base) };
            return tag;
        }
    }

    // Fallback: run the callee frame via a mini interpreter loop.
    let target_depth = ctx.vm.frames.len() - 1;
    loop {
        let Some(byte) = ctx
            .vm
            .frame()
            .code_block
            .bytecode
            .bytes
            .get(ctx.vm.frame().pc as usize)
        else {
            return 1;
        };

        let opcode = crate::vm::opcode::Opcode::decode(*byte);
        use std::ops::ControlFlow;

        let pc = ctx.vm.frame().pc as usize;
        match crate::vm::opcode::OPCODE_HANDLERS[opcode as usize](ctx, pc) {
            ControlFlow::Continue(()) => {
                if ctx.vm.frames.len() <= target_depth + 1 {
                    return 0;
                }
                // Check for JIT at frame boundaries in nested calls.
                #[cfg(feature = "jit")]
                if ctx.vm.frame().pc == 0 {
                    if let Some(record) = ctx.try_run_jit() {
                        match record {
                            crate::vm::CompletionRecord::Normal(_) => {
                                if ctx.vm.frames.len() <= target_depth + 1 {
                                    return 0;
                                }
                                continue;
                            }
                            crate::vm::CompletionRecord::Return(_) => return 0,
                            crate::vm::CompletionRecord::Throw(err) => {
                                ctx.vm.pending_exception = Some(err);
                                return 1;
                            }
                        }
                    }
                }
            }
            ControlFlow::Break(record) => {
                match record {
                    crate::vm::CompletionRecord::Throw(err) => {
                        ctx.vm.pending_exception = Some(err);
                        return 1;
                    }
                    _ => return 0,
                }
            }
        }
    }
}

/// CheckReturn + Return sequence.
///
/// Returns 0 for ControlFlow::Continue, 1 for ControlFlow::Break(Return).
///
/// For non-constructor calls, CheckReturn is a no-op and Return calls
/// `handle_return()`. We combine them into one helper since in the JIT
/// they always appear together at the end of a function.
pub(super) extern "C" fn jit_check_return_and_return(ctx: &mut Context) -> u64 {
    use std::ops::ControlFlow;

    // CheckReturn: for non-constructor calls this is a no-op.
    // We only handle non-constructor for now.
    let frame = ctx.vm.frame();
    if frame.construct() {
        // Fall back — this shouldn't happen since we don't JIT constructors yet,
        // but be safe.
        return 2; // signal error
    }

    // Return: truncate stack, push result, pop frame.
    match ctx.handle_return() {
        ControlFlow::Continue(()) => 0,
        ControlFlow::Break(_) => 1,
    }
}
