//! Runtime helper functions callable from JIT-compiled code.
//!
//! These are `extern "C"` functions that the JIT emits calls to. They bridge
//! between the native code and Boa's runtime, operating on `Context` and its
//! register file. Each helper corresponds to one or more bytecode opcodes.
//!
//! As the JIT matures, many of these will be inlined into the generated code.
//! For now, calling back into Rust is the simplest correct approach.

use crate::{Context, JsValue};

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
