//! Runtime helper functions callable from JIT-compiled code.
//!
//! These are `extern "C"` functions that the JIT emits calls to. They bridge
//! between the native code and Boa's runtime, operating on `Context` and its
//! register file. Each helper corresponds to one or more bytecode opcodes.
//!
//! As the JIT matures, many of these will be inlined into the generated code.
//! For now, calling back into Rust is the simplest correct approach.

use crate::{Context, JsValue, value::JsVariant};

/// Report the memory layout offsets needed for inline caching.
#[cfg(test)]
pub(super) fn report_ic_offsets() {
    use crate::JsObject;

    let obj = JsObject::with_null_proto();
    let js_val = JsValue::from(obj.clone());
    let raw_bits: u64 = unsafe { std::mem::transmute_copy(&js_val) };
    let gc_ptr = (raw_bits & 0x0000_FFFF_FFFF_FFFF) as *const u8;

    let borrowed = obj.borrow();
    let shape_ptr = &borrowed.properties().shape as *const _ as *const u8;
    let storage_ptr = &borrowed.properties().storage as *const _ as *const u8;

    let shape_offset = unsafe { shape_ptr.offset_from(gc_ptr) };
    let storage_offset = unsafe { storage_ptr.offset_from(gc_ptr) };

    eprintln!("=== JIT IC Layout ===");
    eprintln!("gc_ptr: {gc_ptr:p}");
    eprintln!("shape offset from gc_ptr: {shape_offset}");
    eprintln!("storage offset from gc_ptr: {storage_offset}");
    eprintln!("sizeof Shape: {}", size_of::<crate::object::shape::Shape>());

    // Check what shape.to_addr_usize() returns vs the raw bytes at the shape offset
    let shape_addr = borrowed.properties().shape.to_addr_usize();
    let raw_at_shape = unsafe { *(shape_ptr as *const usize) };
    eprintln!("shape.to_addr_usize(): 0x{shape_addr:x}");
    eprintln!("raw usize at shape offset: 0x{raw_at_shape:x}");
    eprintln!("=== End IC Layout ===");
}

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

/// `GetPropertyByValue` — `obj[key]` read. Returns 0 on success, 1 on exception.
pub(super) extern "C" fn jit_get_property_by_value(
    ctx: &mut Context,
    dst: u32,
    key: u32,
    receiver: u32,
    object: u32,
) -> u64 {
    let key_val = ctx.vm.get_register(key as usize).clone();
    let receiver_val = ctx.vm.get_register(receiver as usize).clone();
    let object_val = ctx.vm.get_register(object as usize).clone();

    let result = (|| {
        let object = object_val.to_object(ctx)?;
        let key = key_val.to_property_key(ctx)?;
        object.__get__(&key, receiver_val, &mut ctx.into())
    })();

    match result {
        Ok(value) => {
            ctx.vm.set_register(dst as usize, value);
            0
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
    }
}

/// `GetPropertyByValuePush` — `obj[key]` read, keeps object on stack for compound assignment.
/// Returns 0 on success, 1 on exception.
pub(super) extern "C" fn jit_get_property_by_value_push(
    ctx: &mut Context,
    dst: u32,
    key: u32,
    receiver: u32,
    object: u32,
) -> u64 {
    // Same as GetPropertyByValue — the "push" variant in the interpreter
    // pushes the object for a later SetPropertyByValue, but in the JIT
    // the object stays in its register.
    jit_get_property_by_value(ctx, dst, key, receiver, object)
}

/// `SetPropertyByValue` — `obj[key] = value`. Returns 0 on success, 1 on exception.
pub(super) extern "C" fn jit_set_property_by_value(
    ctx: &mut Context,
    value: u32,
    key: u32,
    receiver: u32,
    object: u32,
) -> u64 {
    let value_val = ctx.vm.get_register(value as usize).clone();
    let key_val = ctx.vm.get_register(key as usize).clone();
    let receiver_val = ctx.vm.get_register(receiver as usize).clone();
    let object_val = ctx.vm.get_register(object as usize).clone();

    let result = (|| {
        let object = object_val.to_object(ctx)?;
        let key = key_val.to_property_key(ctx)?;
        object.__set__(key, value_val, receiver_val, &mut ctx.into())
    })();

    match result {
        Ok(_) => 0,
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
    }
}

/// `GetName` — look up a binding in the environment chain. Returns 0 on success, 1 on exception.
pub(super) extern "C" fn jit_get_name(ctx: &mut Context, dst: u32, binding_index: u32) -> u64 {
    let mut binding_locator =
        ctx.vm.frame().code_block.bindings[binding_index as usize].clone();

    if let Err(err) = ctx.find_runtime_binding(&mut binding_locator) {
        ctx.vm.pending_exception = Some(err);
        return 1;
    }

    match ctx.get_binding(&binding_locator) {
        Ok(Some(value)) => {
            ctx.vm.set_register(dst as usize, value);
            0
        }
        Ok(None) => {
            let name = binding_locator.name().to_std_string_escaped();
            ctx.vm.pending_exception = Some(
                crate::JsNativeError::reference()
                    .with_message(format!("{name} is not defined"))
                    .into(),
            );
            1
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
    }
}

/// `GetPropertyByName` — `obj.prop` read with IC fast path.
/// Returns 0 on success, 1 on exception.
pub(super) extern "C" fn jit_get_property_by_name(
    ctx: &mut Context,
    dst: u32,
    object: u32,
    ic_index: u32,
) -> u64 {
    use crate::object::shape::slot::SlotAttributes;

    let object_val = ctx.vm.get_register(object as usize).clone();

    let result = (|| -> crate::JsResult<JsValue> {
        let Some(object_obj) = object_val.as_object() else {
            // Non-object: fall through to slow path
            let object_obj = object_val.to_object(ctx)?;
            let key = ctx.vm.frame().code_block().ic[ic_index as usize].name.clone();
            let key = crate::property::PropertyKey::from(key);
            return object_obj.__get__(&key, object_val, &mut ctx.into());
        };

        // IC fast path: check the inline cache
        let ic = &ctx.vm.frame().code_block().ic[ic_index as usize];
        let object_borrowed = object_obj.borrow();
        if let Some((shape, slot)) = ic.get(object_borrowed.shape()) {
            let mut result = if slot.attributes.contains(SlotAttributes::PROTOTYPE) {
                let prototype = shape.prototype().expect("prototype should have value");
                let prototype = prototype.borrow();
                prototype.properties().storage[slot.index as usize].clone()
            } else {
                object_borrowed.properties().storage[slot.index as usize].clone()
            };

            drop(object_borrowed);
            if slot.attributes.has_get() && result.is_object() {
                result = result.as_object().expect("should be getter").call(
                    &object_val,
                    &[],
                    ctx,
                )?;
            }
            return Ok(result);
        }
        drop(object_borrowed);

        // IC miss: full lookup
        let key = ic.name.clone();
        let key = crate::property::PropertyKey::from(key);
        object_obj.__get__(&key, object_val, &mut ctx.into())
    })();

    match result {
        Ok(value) => {
            ctx.vm.set_register(dst as usize, value);
            0
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
    }
}

/// `GetLengthProperty` — `obj.length` read (specialized). Returns 0 on success, 1 on exception.
pub(super) extern "C" fn jit_get_length_property(
    ctx: &mut Context,
    dst: u32,
    object: u32,
    _ic_index: u32,
) -> u64 {
    // Simplified: just get the "length" property.
    let object_val = ctx.vm.get_register(object as usize).clone();

    let result = (|| {
        let object_obj = object_val.to_object(ctx)?;
        let key = crate::property::PropertyKey::from(
            crate::JsString::from("length"),
        );
        object_obj.__get__(&key, object_val, &mut ctx.into())
    })();

    match result {
        Ok(value) => {
            ctx.vm.set_register(dst as usize, value);
            0
        }
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
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
pub(super) extern "C" fn jit_call(
    ctx: &mut Context,
    argument_count: u32,
    reg_base_ptr: *mut u64, // pointer to caller's reg_base stack slot
) -> u64 {
    let result = jit_call_inner(ctx, argument_count);

    // Update the caller's reg_base pointer. The stack Vec may have been
    // reallocated during the callee's execution (frame push resizes the stack).
    // The caller will reload reg_base from this slot after we return.
    let rp = ctx.vm.frame().rp as usize;
    let new_base = ctx.vm.stack.stack[rp..].as_mut_ptr().cast::<u64>();
    unsafe { reg_base_ptr.cast::<*mut u64>().write(new_base) };

    result
}

fn jit_call_inner(ctx: &mut Context, argument_count: u32) -> u64 {
    use crate::vm::call_frame::CallFrameFlags;

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

    // Run the callee by delegating to the main interpreter loop's run().
    // Set exit_early so run() returns after the callee completes.
    ctx.vm.frame_mut().flags |= CallFrameFlags::EXIT_EARLY;

    match ctx.run() {
        crate::vm::CompletionRecord::Return(result) => {
            // exit_early: handle_return truncated stack but didn't pop frame.
            // Pop the callee's frame and push the result for the caller.
            let frame = ctx.vm.frames.last().expect("callee frame must exist");
            ctx.vm.stack.truncate_to_frame(frame);
            ctx.vm.pop_frame();
            ctx.vm.stack.push(result);
            0
        }
        crate::vm::CompletionRecord::Throw(err) => {
            ctx.vm.pending_exception = Some(err);
            // Frame may or may not be popped depending on where the throw happened.
            // handle_throw() already unwinds frames. Just propagate.
            1
        }
        crate::vm::CompletionRecord::Normal(_) => {
            // Shouldn't happen with exit_early, but handle gracefully.
            0
        }
    }
}

/// `CheckReturn` — handles constructor return value logic.
/// Returns 0 on success, 1 on exception.
///
/// For non-constructor calls this is a no-op. For constructors,
/// it checks if the return value is an object (keep it) or not
/// (return `this` instead), matching the interpreter's behavior.
pub(super) extern "C" fn jit_check_return(ctx: &mut Context) -> u64 {
    let frame = ctx.vm.frame();
    if !frame.construct() {
        return 0;
    }

    let this = ctx.vm.stack.get_this(frame).clone();
    let result = ctx.vm.take_return_value();

    if result.is_object() {
        ctx.vm.set_return_value(result);
        return 0;
    }

    if !this.is_undefined() {
        ctx.vm.set_return_value(this);
        return 0;
    }

    if !result.is_undefined() {
        ctx.vm.pending_exception = Some(
            crate::JsNativeError::typ()
                .with_message("derived constructor can only return an Object or undefined")
                .into(),
        );
        return 1;
    }

    // Need to get `this` from the environment for derived constructors.
    let frame = ctx.vm.frame();
    if frame.has_this_value_cached() {
        ctx.vm.set_return_value(this);
        return 0;
    }

    match ctx.vm.frame().environments.get_this_binding() {
        Err(err) => {
            ctx.vm.pending_exception = Some(err);
            1
        }
        Ok(Some(this)) => {
            ctx.vm.set_return_value(this);
            0
        }
        Ok(None) => {
            let this = ctx.realm().global_this().clone().into();
            ctx.vm.set_return_value(this);
            0
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

    // CheckReturn is now handled by jit_check_return before this is called.
    // Just do the Return: truncate stack, push result, pop frame.
    match ctx.handle_return() {
        ControlFlow::Continue(()) => 0,
        ControlFlow::Break(crate::vm::CompletionRecord::Return(val)) => {
            // exit_early case: handle_return took the return value and
            // returned it in the CompletionRecord. Put it back so the
            // caller (JitFn::call or jit_call) can find it.
            ctx.vm.set_return_value(val);
            1
        }
        ControlFlow::Break(_) => 1,
    }
}
