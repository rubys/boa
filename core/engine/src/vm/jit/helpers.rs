//! Runtime helper functions callable from JIT-compiled code.
//!
//! These are `extern "C"` functions that the JIT emits calls to. They bridge
//! between the native code and Boa's runtime, operating on `Context` and its
//! register file. Each helper corresponds to one or more bytecode opcodes.
//!
//! As the JIT matures, many of these will be inlined into the generated code.
//! For now, calling back into Rust is the simplest correct approach.

use crate::{Context, JsValue, value::JsVariant};

/// Fixed offsets from GC pointer to object internals.
/// These are verified by the `ic_layout_offsets` test.
pub(super) const SHAPE_OFFSET: i32 = 32;     // GcPtr → PropertyMap.shape (enum start)
pub(super) const SHAPE_PTR_OFFSET: i32 = 40; // GcPtr → Shape inner Gc pointer (discriminant + ptr)
pub(super) const STORAGE_OFFSET: i32 = 64;  // GcPtr → PropertyMap.storage (Vec struct start)
pub(super) const STORAGE_PTR_OFFSET: i32 = 72; // GcPtr → storage Vec data pointer (Vec layout: cap, ptr, len)

/// Perform an inline-cache property lookup using raw pointer arithmetic.
/// This bypasses GcRefCell::borrow() for maximum speed.
///
/// Returns `Some(value)` if the IC hits, `None` if it misses.
///
/// # Safety
/// The `nan_boxed_obj` must be a NaN-boxed object pointer (tag == MASK_OBJECT).
/// The `cached_shape_ptr` must be a valid shape GcBox pointer from `to_addr_usize() - 16`.
pub(super) unsafe fn ic_fast_get(
    nan_boxed_obj: u64,
    cached_shape_ptr: u64,
    slot_index: u32,
) -> Option<u64> {
    // Extract 48-bit GC pointer from NaN-boxed object.
    let gc_ptr = (nan_boxed_obj & 0x0000_FFFF_FFFF_FFFF) as *const u8;

    // Read the shape's inner Gc pointer at gc_ptr + 40.
    // This is the Gc<Inner> pointer (a NonNull<GcBox<Inner>>).
    let shape_gc_ptr = *(gc_ptr.add(SHAPE_PTR_OFFSET as usize) as *const u64);

    // Compare with the cached shape.
    if shape_gc_ptr != cached_shape_ptr {
        return None;
    }

    // IC hit! Read the storage Vec's data pointer at gc_ptr + 72.
    // Vec layout on this platform is (capacity, ptr, len), so ptr is at +8 from Vec start.
    let storage_data_ptr = *(gc_ptr.add(STORAGE_PTR_OFFSET as usize) as *const *const u64);

    // Read the property value at storage[slot_index].
    let value = *storage_data_ptr.add(slot_index as usize);

    Some(value)
}

/// Verify the IC layout offsets and test the fast path.
#[cfg(test)]
pub(super) fn verify_ic_offsets_and_fast_path() {
    use crate::{Context, JsObject, Source};

    let mut ctx = Context::default();

    // Create an object with a property, then verify the IC works.
    let obj_val = ctx.eval(Source::from_bytes("({x: 42, y: 99})")).unwrap();
    let raw_bits: u64 = unsafe { std::mem::transmute_copy(&obj_val) };

    // Verify it's an object (tag == 0x7FFC)
    let tag = raw_bits & 0x7FFF_0000_0000_0000;
    assert_eq!(tag, 0x7FFC_0000_0000_0000, "should be object tag");

    let gc_ptr = (raw_bits & 0x0000_FFFF_FFFF_FFFF) as *const u8;

    // Get the shape via the proper API for comparison.
    let obj = obj_val.as_object().unwrap();
    let borrowed = obj.borrow();

    // Verify offsets are correct.
    let shape_struct_ptr = &borrowed.properties().shape as *const _ as *const u8;
    let storage_struct_ptr = &borrowed.properties().storage as *const _ as *const u8;
    let actual_shape_offset = unsafe { shape_struct_ptr.offset_from(gc_ptr) };
    let actual_storage_offset = unsafe { storage_struct_ptr.offset_from(gc_ptr) };
    assert_eq!(actual_shape_offset, SHAPE_OFFSET as isize, "shape offset mismatch");
    assert_eq!(actual_storage_offset, STORAGE_OFFSET as isize, "storage offset mismatch");

    // Dump the raw bytes around the shape to understand the layout.
    let shape_bytes = unsafe {
        std::slice::from_raw_parts(shape_struct_ptr, 16)
    };
    eprintln!("Shape raw bytes: {:02x?}", shape_bytes);

    let shape_addr_usize = borrowed.properties().shape.to_addr_usize();
    eprintln!("shape.to_addr_usize(): 0x{shape_addr_usize:x}");

    // Find the shape pointer within the 16 bytes
    let word0 = unsafe { *(shape_struct_ptr as *const u64) };
    let word1 = unsafe { *(shape_struct_ptr.add(8) as *const u64) };
    eprintln!("shape word0: 0x{word0:x}");
    eprintln!("shape word1: 0x{word1:x}");

    // Figure out which word contains the Gc pointer
    let gc_header_size_guess = 16_u64;
    let expected_gc_ptr_from_word0 = word0.wrapping_add(gc_header_size_guess);
    let expected_gc_ptr_from_word1 = word1.wrapping_add(gc_header_size_guess);
    eprintln!("word0 + 16 = 0x{expected_gc_ptr_from_word0:x}");
    eprintln!("word1 + 16 = 0x{expected_gc_ptr_from_word1:x}");

    let raw_shape_gc_ptr = if expected_gc_ptr_from_word0 as usize == shape_addr_usize {
        eprintln!("Shape Gc pointer is at offset +0 (word0)");
        word0
    } else if expected_gc_ptr_from_word1 as usize == shape_addr_usize {
        eprintln!("Shape Gc pointer is at offset +8 (word1)");
        word1
    } else {
        panic!("Cannot find shape Gc pointer in Shape bytes");
    };

    // Verify the relationship: to_addr_usize = raw_gc_ptr + GcHeader_size
    let gc_header_size = shape_addr_usize - (raw_shape_gc_ptr as usize);
    eprintln!("GcHeader size: {gc_header_size}");
    assert!(gc_header_size > 0 && gc_header_size <= 32, "unexpected GcHeader size");

    // Verify storage Vec layout
    let storage_vec_ptr = unsafe { gc_ptr.add(STORAGE_OFFSET as usize) };
    // Vec data ptr is at offset +8 within the Vec (layout: cap, ptr, len)
    let storage_data_ptr = unsafe { *(storage_vec_ptr.add(8) as *const *const u8) };
    eprintln!("storage Vec at gc_ptr+{STORAGE_OFFSET}: {storage_vec_ptr:p}");
    eprintln!("raw at storage+0: 0x{:x}", unsafe { *(storage_vec_ptr as *const u64) });
    eprintln!("raw at storage+8: 0x{:x}", unsafe { *(storage_vec_ptr.add(8) as *const u64) });
    eprintln!("raw at storage+16: 0x{:x}", unsafe { *(storage_vec_ptr.add(16) as *const u64) });
    eprintln!("storage len: {}", borrowed.properties().storage.len());
    let actual_data_ptr = borrowed.properties().storage.as_ptr() as *const u8;
    eprintln!("actual Vec data ptr: {actual_data_ptr:p}");
    eprintln!("actual Vec capacity: {}", borrowed.properties().storage.capacity());
    assert_eq!(storage_data_ptr, actual_data_ptr, "storage data ptr mismatch");

    // Now test the IC fast path: read property "x" at slot 0.
    let proper_x = borrowed.properties().storage[0].clone();
    assert_eq!(
        proper_x.as_number().unwrap(),
        42.0,
        "storage[0] should be 42"
    );

    drop(borrowed);

    // Test ic_fast_get with the cached shape pointer.
    let result = unsafe { ic_fast_get(raw_bits, raw_shape_gc_ptr, 0) };
    assert!(result.is_some(), "IC should hit");

    // The returned u64 should be a NaN-boxed integer 42.
    let result_val: JsValue = unsafe { std::mem::transmute(result.unwrap()) };
    // Don't drop — we don't own the refcount.
    let num = result_val.as_number();
    std::mem::forget(result_val);
    assert_eq!(num.unwrap(), 42.0, "IC fast path should return 42");

    // Test with a wrong shape pointer — should miss.
    let result = unsafe { ic_fast_get(raw_bits, 0xDEAD_BEEF, 0) };
    assert!(result.is_none(), "IC should miss with wrong shape");

    eprintln!("IC fast path verification passed!");
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





// ============================================================
// Bulk helpers — written against actual Boa interpreter APIs.
// ============================================================

// --- This ---
pub(super) extern "C" fn jit_this(ctx: &mut Context, dst: u32) {
    let this = ctx.vm.stack.get_this(ctx.vm.frame()).clone();
    ctx.vm.set_register(dst as usize, this);
}

// --- Unary operations ---
pub(super) extern "C" fn jit_neg(ctx: &mut Context, value: u32) -> u64 {
    let val = ctx.vm.get_register(value as usize).clone();
    match val.neg(ctx) {
        Ok(r) => { ctx.vm.set_register(value as usize, r); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_pos(ctx: &mut Context, value: u32) -> u64 {
    let val = ctx.vm.get_register(value as usize).clone();
    match val.to_number(ctx) {
        Ok(n) => { ctx.vm.set_register(value as usize, n.into()); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_bit_not(ctx: &mut Context, value: u32) -> u64 {
    let val = ctx.vm.get_register(value as usize).clone();
    match val.to_i32(ctx) {
        Ok(n) => { ctx.vm.set_register(value as usize, JsValue::from(!n)); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_logical_not(ctx: &mut Context, value: u32) {
    let val = ctx.vm.get_register(value as usize);
    ctx.vm.set_register(value as usize, JsValue::from(!val.to_boolean()));
}

pub(super) extern "C" fn jit_type_of(ctx: &mut Context, value: u32) {
    let val = ctx.vm.get_register(value as usize);
    let s = crate::JsString::from(val.type_of());
    ctx.vm.set_register(value as usize, JsValue::from(s));
}

pub(super) extern "C" fn jit_is_object(ctx: &mut Context, value: u32) {
    let val = ctx.vm.get_register(value as usize);
    ctx.vm.set_register(value as usize, JsValue::from(val.is_object()));
}

// --- Comparison helpers ---
pub(super) extern "C" fn jit_instance_of(ctx: &mut Context, dst: u32, lhs: u32, rhs: u32) -> u64 {
    let l = ctx.vm.get_register(lhs as usize).clone();
    let r = ctx.vm.get_register(rhs as usize).clone();
    match l.instance_of(&r, ctx) {
        Ok(v) => { ctx.vm.set_register(dst as usize, v.into()); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_in(ctx: &mut Context, dst: u32, lhs: u32, rhs: u32) -> u64 {
    let rhs_val = ctx.vm.get_register(rhs as usize).clone();
    let lhs_val = ctx.vm.get_register(lhs as usize).clone();
    let result = (|| {
        let Some(rhs_obj) = rhs_val.as_object() else {
            return Err(crate::JsNativeError::typ()
                .with_message(format!("right-hand side of 'in' should be an object, got `{}`", rhs_val.type_of()))
                .into());
        };
        let key = lhs_val.to_property_key(ctx)?;
        rhs_obj.has_property(key, ctx)
    })();
    match result {
        Ok(v) => { ctx.vm.set_register(dst as usize, v.into()); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_value_not_null_or_undefined(ctx: &mut Context, src: u32) -> u64 {
    let val = ctx.vm.get_register(src as usize);
    if val.is_null_or_undefined() {
        ctx.vm.pending_exception = Some(
            crate::JsNativeError::typ().with_message("Cannot destructure undefined or null").into()
        );
        1
    } else { 0 }
}

// --- Variable / binding access ---
pub(super) extern "C" fn jit_set_name(ctx: &mut Context, src: u32, binding_index: u32) -> u64 {
    let value = ctx.vm.get_register(src as usize).clone();
    let mut locator = ctx.vm.frame().code_block.bindings[binding_index as usize].clone();
    match ctx.find_runtime_binding(&mut locator).and_then(|()| ctx.set_binding(&locator, value, ctx.vm.frame().code_block.strict())) {
        Ok(()) => 0,
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_get_name_or_undefined(ctx: &mut Context, dst: u32, binding_index: u32) -> u64 {
    let mut locator = ctx.vm.frame().code_block.bindings[binding_index as usize].clone();
    match ctx.find_runtime_binding(&mut locator) {
        Ok(()) => match ctx.get_binding(&locator) {
            Ok(v) => { ctx.vm.set_register(dst as usize, v.unwrap_or_default()); 0 }
            Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
        },
        Err(_) => { ctx.vm.set_register(dst as usize, JsValue::undefined()); 0 }
    }
}

pub(super) extern "C" fn jit_get_name_and_locator(ctx: &mut Context, dst: u32, binding_index: u32) -> u64 {
    let mut locator = ctx.vm.frame().code_block.bindings[binding_index as usize].clone();
    match ctx.find_runtime_binding(&mut locator) {
        Ok(()) => match ctx.get_binding(&locator) {
            Ok(Some(v)) => {
                ctx.vm.set_register(dst as usize, v);
                ctx.vm.frame_mut().binding_stack.push(locator);
                0
            }
            Ok(None) => {
                let name = locator.name().to_std_string_escaped();
                ctx.vm.pending_exception = Some(crate::JsNativeError::reference().with_message(format!("{name} is not defined")).into());
                1
            }
            Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
        },
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_get_locator(ctx: &mut Context, binding_index: u32) -> u64 {
    let mut locator = ctx.vm.frame().code_block.bindings[binding_index as usize].clone();
    match ctx.find_runtime_binding(&mut locator) {
        Ok(()) => { ctx.vm.frame_mut().binding_stack.push(locator); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_set_name_by_locator(ctx: &mut Context, src: u32) -> u64 {
    let value = ctx.vm.get_register(src as usize).clone();
    let locator = ctx.vm.frame_mut().binding_stack.pop().expect("locator must exist");
    let strict = ctx.vm.frame().code_block.strict();
    match ctx.set_binding(&locator, value, strict) {
        Ok(()) => 0,
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_put_lexical_value(ctx: &mut Context, src: u32, binding_index: u32) {
    let value = ctx.vm.get_register(src as usize).clone();
    let locator = ctx.vm.frame().code_block.bindings[binding_index as usize].clone();
    let scope = locator.scope();
    let bi = locator.binding_index();
    let frame = ctx.vm.frame_mut();
    let global = frame.realm.environment();
    frame.environments.put_lexical_value(scope, bi, value, global);
}

pub(super) extern "C" fn jit_def_init_var(ctx: &mut Context, src: u32, binding_index: u32) -> u64 {
    let value = ctx.vm.get_register(src as usize).clone();
    let mut locator = ctx.vm.frame().code_block.bindings[binding_index as usize].clone();
    let strict = ctx.vm.frame().code_block.strict();
    match ctx.find_runtime_binding(&mut locator).and_then(|()| ctx.set_binding(&locator, value, strict)) {
        Ok(()) => 0,
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

/* DISABLED — API mismatch
pub(super) extern "C" fn jit_def_var(ctx: &mut Context, binding_index: u32) {
    let locator = &ctx.vm.frame().code_block.bindings[binding_index as usize];
    let frame = ctx.vm.frame_mut();
    let global = frame.realm.environment();
    let scope = boa_ast::scope::BindingLocatorScope::Stack(locator.scope());
    frame.environments.put_value_if_uninitialized(scope, locator.binding_index(), JsValue::undefined(), global);
}

*/

pub(super) extern "C" fn jit_delete_name(ctx: &mut Context, dst: u32, binding_index: u32) -> u64 {
    let mut locator = ctx.vm.frame().code_block.bindings[binding_index as usize].clone();
    match ctx.find_runtime_binding(&mut locator).and_then(|()| ctx.delete_binding(&locator)) {
        Ok(v) => { ctx.vm.set_register(dst as usize, JsValue::from(v)); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

// --- Property access ---
pub(super) extern "C" fn jit_set_property_by_name(ctx: &mut Context, value: u32, object: u32, ic_index: u32) -> u64 {
    let val = ctx.vm.get_register(value as usize).clone();
    let obj_val = ctx.vm.get_register(object as usize).clone();
    let strict = ctx.vm.frame().code_block.strict();
    let result = (|| {
        let ic = &ctx.vm.frame().code_block().ic[ic_index as usize];
        let key = crate::property::PropertyKey::from(ic.name.clone());
        let obj = obj_val.to_object(ctx)?;
        obj.__set__(key, val, obj_val, &mut ctx.into())?;
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_get_property_by_name_with_this(ctx: &mut Context, dst: u32, receiver: u32, value: u32, ic_index: u32) -> u64 {
    let recv = ctx.vm.get_register(receiver as usize).clone();
    let obj_val = ctx.vm.get_register(value as usize).clone();
    let result = (|| {
        let ic = &ctx.vm.frame().code_block().ic[ic_index as usize];
        let key = crate::property::PropertyKey::from(ic.name.clone());
        let obj = obj_val.to_object(ctx)?;
        obj.__get__(&key, recv, &mut ctx.into())
    })();
    match result {
        Ok(v) => { ctx.vm.set_register(dst as usize, v); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_define_own_property_by_name(ctx: &mut Context, object: u32, value: u32, name_index: u32) -> u64 {
    let obj_val = ctx.vm.get_register(object as usize).clone();
    let val = ctx.vm.get_register(value as usize).clone();
    let result = (|| {
        let name = ctx.vm.frame().code_block().constant_string(name_index as usize);
        let key = crate::property::PropertyKey::from(name);
        let obj = obj_val.to_object(ctx)?;
        obj.__define_own_property__(
            &key,
            crate::property::PropertyDescriptor::builder().value(val).writable(true).enumerable(true).configurable(true).build(),
            &mut ctx.into(),
        )
    })();
    match result {
        Ok(_) => 0,
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_define_own_property_by_value(ctx: &mut Context, value: u32, key: u32, object: u32) -> u64 {
    let val = ctx.vm.get_register(value as usize).clone();
    let k = ctx.vm.get_register(key as usize).clone();
    let obj_val = ctx.vm.get_register(object as usize).clone();
    let result = (|| {
        let obj = obj_val.to_object(ctx)?;
        let key = k.to_property_key(ctx)?;
        obj.__define_own_property__(
            &key,
            crate::property::PropertyDescriptor::builder().value(val).writable(true).enumerable(true).configurable(true).build(),
            &mut ctx.into(),
        )
    })();
    match result {
        Ok(_) => 0,
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_delete_property_by_name(ctx: &mut Context, object: u32, name_index: u32) -> u64 {
    let obj_val = ctx.vm.take_register(object as usize);
    let result = (|| {
        let name = ctx.vm.frame().code_block().constant_string(name_index as usize);
        let key = crate::property::PropertyKey::from(name);
        let obj = obj_val.to_object(ctx)?;
        let r = obj.__delete__(&key, &mut ctx.into())?;
        if !r && ctx.vm.frame().code_block.strict() {
            return Err(crate::JsNativeError::typ().with_message("Cannot delete property").into());
        }
        Ok(JsValue::from(r))
    })();
    match result {
        Ok(v) => { ctx.vm.set_register(object as usize, v); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_delete_property_by_value(ctx: &mut Context, object: u32, key: u32) -> u64 {
    let obj_val = ctx.vm.get_register(object as usize).clone();
    let k = ctx.vm.get_register(key as usize).clone();
    let result = (|| {
        let obj = obj_val.to_object(ctx)?;
        let key = k.to_property_key(ctx)?;
        let r = obj.__delete__(&key, &mut ctx.into())?;
        if !r && ctx.vm.frame().code_block.strict() {
            return Err(crate::JsNativeError::typ().with_message("Cannot delete property").into());
        }
        Ok(JsValue::from(r))
    })();
    match result {
        Ok(v) => { ctx.vm.set_register(object as usize, v); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_to_property_key(ctx: &mut Context, src: u32, dst: u32) -> u64 {
    let val = ctx.vm.get_register(src as usize).clone();
    match val.to_property_key(ctx) {
        Ok(k) => { ctx.vm.set_register(dst as usize, k.into()); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

// --- Object operations ---
pub(super) extern "C" fn jit_get_prototype(ctx: &mut Context, object: u32) -> u64 {
    let obj_val = ctx.vm.get_register(object as usize).clone();
    let result = (|| {
        let obj = obj_val.as_object().ok_or_else(|| crate::JsNativeError::typ().with_message("not an object"))?;
        obj.__get_prototype_of__(ctx)
    })();
    match result {
        Ok(p) => { ctx.vm.set_register(object as usize, p.map_or(JsValue::null(), |p| p.into())); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_set_prototype(ctx: &mut Context, object: u32, prototype: u32) -> u64 {
    let obj_val = ctx.vm.get_register(object as usize).clone();
    let proto_val = ctx.vm.get_register(prototype as usize).clone();
    let result = (|| {
        let obj = obj_val.as_object().ok_or_else(|| crate::JsNativeError::typ().with_message("not an object"))?;
        let proto = if proto_val.is_null() { None } else { Some(proto_val.to_object(ctx)?) };
        obj.__set_prototype_of__(proto, ctx)
    })();
    match result {
        Ok(_) => 0,
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_store_literal(ctx: &mut Context, dst: u32, index: u32) {
    let constant = &ctx.vm.frame().code_block.constants[index as usize];
    let val = match constant {
        crate::vm::Constant::String(s) => JsValue::from(s.clone()),
        crate::vm::Constant::BigInt(b) => JsValue::from(b.clone()),
        _ => JsValue::undefined(),
    };
    ctx.vm.set_register(dst as usize, val);
}

pub(super) extern "C" fn jit_store_empty_object(ctx: &mut Context, dst: u32) {
    let obj = ctx.intrinsics().templates().ordinary_object().create(
        crate::builtins::OrdinaryObject,
        Vec::new(),
    );
    ctx.vm.set_register(dst as usize, obj.into());
}

pub(super) extern "C" fn jit_store_new_array(ctx: &mut Context, dst: u32) {
    let array = ctx.intrinsics().templates().array().create(
        crate::builtins::array::Array,
        Vec::from([JsValue::new(0)]),  // Initial storage with length = 0
    );
    ctx.vm.set_register(dst as usize, array.into());
}

pub(super) extern "C" fn jit_store_regexp(ctx: &mut Context, dst: u32, pattern_index: u32, flags_index: u32) -> u64 {
    let pattern = ctx.vm.frame().code_block().constant_string(pattern_index as usize);
    let flags = ctx.vm.frame().code_block().constant_string(flags_index as usize);
    match crate::builtins::regexp::RegExp::create(
        &JsValue::from(pattern),
        &JsValue::from(flags),
        ctx,
    ) {
        Ok(r) => { ctx.vm.set_register(dst as usize, r.into()); 0 }
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

pub(super) extern "C" fn jit_push_value_to_array(ctx: &mut Context, value: u32, array: u32) -> u64 {
    let val = ctx.vm.get_register(value as usize).clone();
    let o = ctx.vm.get_register(array as usize)
        .as_object().expect("should be an object").clone();

    // Fast path: push directly to dense indexed storage.
    {
        let mut o_mut = o.borrow_mut();
        let len = o_mut.properties().storage[0].as_i32();
        if let Some(len) = len {
            if o_mut.properties_mut().indexed_properties.push_dense(&val) {
                o_mut.properties_mut().storage[0] = JsValue::new(len + 1);
                return 0;
            }
        }
    }

    // Slow path
    let len = o.length_of_array_like(ctx).expect("should have length");
    o.create_data_property_or_throw(len, val, ctx).expect("should create property");
    0
}

pub(super) extern "C" fn jit_push_elision_to_array(ctx: &mut Context, array: u32) -> u64 {
    let arr_val = ctx.vm.get_register(array as usize).clone();
    let result = (|| {
        let arr = arr_val.as_object().ok_or_else(|| crate::JsNativeError::typ().with_message("not an array"))?;
        let len = arr.length_of_array_like(ctx)?;
        arr.set(crate::js_string!("length"), JsValue::from(len + 1), true, ctx)?;
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

// --- Scope ---
pub(super) extern "C" fn jit_push_scope(ctx: &mut Context, scope_index: u32) {
    let scope = ctx.vm.frame().code_block().constant_scope(scope_index as usize);
    let frame = ctx.vm.frame_mut();
    let global = frame.realm.environment();
    frame.environments.push_lexical(scope.num_bindings() as u32, global);
}

/* DISABLED — API mismatch
pub(super) extern "C" fn jit_bind_this_value(ctx: &mut Context, value: u32) -> u64 {
    let val = ctx.vm.get_register(value as usize).clone();
    let result = (|| {
        let global = ctx.vm.frame().realm.environment();
        let env = ctx.vm.frame().environments.get_this_environment(global);
        env.bind_this_value(val.clone())?;
        let val_obj = val.as_object().map(|o| o.clone());
        if let Some(obj) = val_obj {
            obj.initialize_instance_elements(ctx)?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(e) => { ctx.vm.pending_exception = Some(e); 1 }
    }
}

*/

pub(super) extern "C" fn jit_create_unmapped_arguments_object(ctx: &mut Context, dst: u32) {
    let args = ctx.vm.stack.get_arguments(ctx.vm.frame()).to_vec();
    let obj = crate::builtins::function::arguments::UnmappedArguments::new(&args, ctx);
    ctx.vm.set_register(dst as usize, obj.into());
}

/* DISABLED — API mismatch
pub(super) extern "C" fn jit_create_mapped_arguments_object(ctx: &mut Context, dst: u32) {
    let func = ctx.vm.stack.get_function(ctx.vm.frame()).clone();
    let code = ctx.vm.frame().code_block.clone();
    let args = ctx.vm.stack.get_arguments(ctx.vm.frame()).to_vec();
    let env = ctx.vm.frame().environments.current_declarative_ref();
    let obj = crate::builtins::function::arguments::MappedArguments::new(
        &func.as_object().expect("should be function"),
        &code,
        &args,
        env,
        ctx,
    );
    ctx.vm.set_register(dst as usize, obj.into());
}

*/

pub(super) extern "C" fn jit_rest_parameter_init(ctx: &mut Context, dst: u32) {
    let argument_count = ctx.vm.frame().argument_count as usize;
    let param_count = ctx.vm.frame().code_block().parameter_length as usize;
    let rest = if argument_count >= param_count {
        let start = param_count;
        let args = ctx.vm.stack.get_arguments(ctx.vm.frame());
        if start < args.len() {
            Some(args[start..].to_vec())
        } else {
            Some(Vec::new())
        }
    } else {
        None
    };
    let args = rest;
    let array = match args {
        Some(rest) => crate::builtins::Array::create_array_from_list(rest, ctx),
        None => ctx.intrinsics().templates().array().create(crate::builtins::array::Array, Vec::new()),
    };
    ctx.vm.set_register(dst as usize, array.into());
}

// --- Function ---
pub(super) extern "C" fn jit_get_function(ctx: &mut Context, dst: u32, index: u32) {
    let code = ctx.vm.frame().code_block().constant_function(index as usize);
    let func = crate::vm::create_function_object_fast(code, ctx);
    ctx.vm.set_register(dst as usize, func.into());
}

/* DISABLED — API mismatch
pub(super) extern "C" fn jit_get_function_object(ctx: &mut Context, function_object: u32) -> u64 {
    let env = ctx.vm.frame().environments.get_this_environment();
    let global = ctx.vm.frame().realm.environment();
    match env.slots(global) {
        Some(slots) => {
            ctx.vm.set_register(function_object as usize, slots.function_object().clone().into());
            0
        }
        None => {
            ctx.vm.pending_exception = Some(crate::JsNativeError::typ().with_message("no function object").into());
            1
        }
    }
}

*/

/* DISABLED — API mismatch
pub(super) extern "C" fn jit_new_target(ctx: &mut Context, dst: u32) {
    let env = ctx.vm.frame().environments.get_this_environment();
    let global = ctx.vm.frame().realm.environment();
    let val = match env.slots(global) {
        Some(slots) => slots.new_target().cloned().map_or(JsValue::undefined(), JsValue::from),
        None => JsValue::undefined(),
    };
    ctx.vm.set_register(dst as usize, val);
}

// --- Constructor ---
*/

pub(super) extern "C" fn jit_new(ctx: &mut Context, argument_count: u32, reg_base_ptr: *mut u64) -> u64 {
    let result = jit_new_inner(ctx, argument_count);
    let rp = ctx.vm.frame().rp as usize;
    let new_base = ctx.vm.stack.stack[rp..].as_mut_ptr().cast::<u64>();
    unsafe { reg_base_ptr.cast::<*mut u64>().write(new_base) };
    result
}

fn jit_new_inner(ctx: &mut Context, argument_count: u32) -> u64 {
    use crate::vm::call_frame::CallFrameFlags;
    let func = ctx.vm.stack.calling_convention_get_function(argument_count as usize);
    let Some(object) = func.as_object() else {
        ctx.vm.pending_exception = Some(crate::JsNativeError::typ().with_message("not a constructor").into());
        return 1;
    };
    let cons = object.clone();
    // Push new.target — __construct__ expects it on the stack.
    ctx.vm.stack.push(cons.clone());
    match cons.__construct__(argument_count as usize).resolve(ctx) {
        Ok(true) => return 0,
        Ok(false) => {}
        Err(e) => { ctx.vm.pending_exception = Some(e); return 1; }
    }
    ctx.vm.frame_mut().flags |= CallFrameFlags::EXIT_EARLY;
    match ctx.run() {
        crate::vm::CompletionRecord::Return(result) => {
            let frame = ctx.vm.frames.last().expect("frame");
            ctx.vm.stack.truncate_to_frame(frame);
            ctx.vm.pop_frame();
            ctx.vm.stack.push(result);
            0
        }
        crate::vm::CompletionRecord::Throw(e) => { ctx.vm.pending_exception = Some(e); 1 }
        crate::vm::CompletionRecord::Normal(_) => 0,
    }
}

// --- Error ---
pub(super) extern "C" fn jit_throw(ctx: &mut Context, src: u32) -> u64 {
    let val = ctx.vm.get_register(src as usize).clone();
    ctx.vm.pending_exception = Some(crate::JsError::from_opaque(val));
    1
}

pub(super) extern "C" fn jit_throw_new_type_error(ctx: &mut Context, message: u32) -> u64 {
    let msg = ctx.vm.frame().code_block().constant_string(message as usize);
    ctx.vm.pending_exception = Some(crate::JsNativeError::typ().with_message(msg.to_std_string_escaped()).into());
    1
}

pub(super) extern "C" fn jit_throw_new_reference_error(ctx: &mut Context, message: u32) -> u64 {
    let msg = ctx.vm.frame().code_block().constant_string(message as usize);
    ctx.vm.pending_exception = Some(crate::JsNativeError::reference().with_message(msg.to_std_string_escaped()).into());
    1
}

pub(super) extern "C" fn jit_throw_mutate_immutable(ctx: &mut Context, index: u32) -> u64 {
    let name = ctx.vm.frame().code_block().constant_string(index as usize);
    ctx.vm.pending_exception = Some(
        crate::JsNativeError::typ().with_message(format!("Cannot assign to read only variable '{}'", name.to_std_string_escaped())).into()
    );
    1
}

// --- Misc ---
pub(super) extern "C" fn jit_set_register_from_accumulator(ctx: &mut Context, dst: u32) {
    let val = ctx.vm.get_return_value();
    ctx.vm.set_register(dst as usize, val);
}
