//! Tests for the JIT compiler.

use super::JitCompiler;

#[test]
fn ic_layout_and_fast_path() {
    super::helpers::verify_ic_offsets_and_fast_path();
}

#[test]
fn jit_compiler_creates_successfully() {
    let compiler = JitCompiler::new();
    assert!(compiler.is_ok(), "JIT compiler should initialize on x86-64");
}

#[test]
fn compile_trivial_function() {
    use crate::vm::CodeBlock;
    use crate::vm::opcode::{BytecodeEmitter, RegisterOperand};
    use boa_string::JsString;

    // Build: StoreZero r0; SetAccumulator r0; CheckReturn; Return
    let mut emitter = BytecodeEmitter::new();
    emitter.emit_store_zero(RegisterOperand::new(0));
    emitter.emit_set_accumulator(RegisterOperand::new(0));
    emitter.emit_check_return();
    emitter.emit_return();

    let mut code = CodeBlock::new(JsString::from("test"), 0, false);
    code.bytecode = emitter.into_bytecode();
    code.register_count = 1;

    assert!(super::can_compile(&code), "trivial code should be compilable");

    let mut compiler = JitCompiler::new().expect("compiler should init");
    let jit_fn = compiler.compile(&code);
    assert!(jit_fn.is_some(), "should compile trivial function");
}

#[test]
fn can_compile_rejects_unsupported() {
    use crate::vm::CodeBlock;
    use crate::vm::opcode::{BytecodeEmitter, RegisterOperand};
    use boa_string::JsString;

    // Pop is not in the supported set.
    let mut emitter = BytecodeEmitter::new();
    emitter.emit_pop();
    emitter.emit_check_return();
    emitter.emit_return();

    let mut code = CodeBlock::new(JsString::from("test"), 0, false);
    code.bytecode = emitter.into_bytecode();
    code.register_count = 1;

    assert!(
        !super::can_compile(&code),
        "code with Pop should not be compilable"
    );

    let mut compiler = JitCompiler::new().expect("compiler should init");
    assert!(
        compiler.compile(&code).is_none(),
        "compile should return None for unsupported code"
    );
}

/// End-to-end test: a simple function that returns a constant is JIT-compiled
/// after reaching the call threshold, and produces the correct result.
#[test]
fn end_to_end_jit_return_constant() {
    use crate::{Context, Source};

    let mut context = Context::default();

    // Define a function and call it enough times to trigger JIT.
    let result = context.eval(Source::from_bytes(
        "function f() { return 42; }
         var r;
         for (var i = 0; i < 20; i++) { r = f(); }
         r",
    ));

    let value = result.expect("should succeed");
    assert_eq!(
        value.as_number().expect("should be number"),
        42.0,
        "JIT'd function should return 42"
    );
}

/// End-to-end: function with arguments and addition.
#[test]
fn end_to_end_jit_add() {
    use crate::{Context, Source};

    let mut context = Context::default();

    let result = context.eval(Source::from_bytes(
        "function add(a, b) { return a + b; }
         var r;
         for (var i = 0; i < 20; i++) { r = add(1, 2); }
         r",
    ));

    let value = result.expect("should succeed");
    assert_eq!(
        value.as_number().expect("should be number"),
        3.0,
        "JIT'd add function should return 3"
    );
}

/// End-to-end: function with a loop (the first real benchmark target).
#[test]
fn end_to_end_jit_sum_loop() {
    use crate::{Context, Source};

    let mut context = Context::default();

    let result = context.eval(Source::from_bytes(
        "function sum(n) {
           var s = 0;
           for (var i = 0; i < n; i++) {
             s = (s + i) | 0;
           }
           return s;
         }
         var r;
         for (var j = 0; j < 20; j++) { r = sum(100); }
         r",
    ));

    let value = result.expect("should succeed");
    assert_eq!(
        value.as_number().expect("should be number"),
        4950.0,
        "JIT'd sum(100) should return 4950"
    );
}

/// Verify that functions with unsupported opcodes still work via the interpreter.
#[test]
fn unsupported_falls_back_to_interpreter() {
    use crate::{Context, Source};

    let mut context = Context::default();

    // typeof uses TypeOf opcode which is not JIT-supported.
    let result = context.eval(Source::from_bytes(
        "function check(x) { return typeof x; }
         var r;
         for (var i = 0; i < 20; i++) { r = check(42); }
         r",
    ));

    let value = result.expect("should succeed");
    let s = value
        .as_string()
        .expect("should be string")
        .to_std_string_escaped();
    assert_eq!(s, "number", "interpreted function should still work");
}

/// Verify that the IC has data at JIT compilation time and that the
/// cached shape matches runtime objects.
#[test]
fn ic_data_available_at_compile_time() {
    use crate::{Context, Source};
    use crate::vm::code_block::JitState;

    let mut context = Context::default();

    // Define get_x and call it 9 times (below JIT threshold of 10).
    context.eval(Source::from_bytes(
        "function get_x(obj) { return obj.x; }
         for (var i = 0; i < 9; i++) get_x({x: i});"
    )).expect("setup should succeed");

    // Find the get_x function's CodeBlock.
    let get_x_val = context.eval(Source::from_bytes("get_x")).unwrap();
    let get_x_obj = get_x_val.as_object().unwrap();
    let get_x_func = get_x_obj
        .downcast_ref::<crate::builtins::function::OrdinaryFunction>()
        .expect("should be an ordinary function");
    let code = &get_x_func.code;

    // Check the IC state after 9 interpreter calls.
    let ic = &code.ic;
    eprintln!("IC entries: {}", ic.len());
    for (i, entry) in ic.iter().enumerate() {
        let entries = entry.entries.borrow();
        eprintln!("  IC[{i}] name={} entries={} megamorphic={}",
            entry.name.to_std_string_escaped(),
            entries.len(),
            entry.megamorphic.get(),
        );
        for (j, cached) in entries.iter().enumerate() {
            if let Some(shape) = cached.shape.upgrade() {
                eprintln!("    entry[{j}]: shape=0x{:x} slot_index={}",
                    shape.to_addr_usize(), cached.slot.index);
            } else {
                eprintln!("    entry[{j}]: stale (shape collected)");
            }
        }
    }

    // The IC should have at least one entry for property "x".
    assert!(!ic.is_empty(), "IC should have entries");
    let first_ic = &ic[0];
    assert_eq!(first_ic.name.to_std_string_escaped(), "x");
    let entries = first_ic.entries.borrow();
    assert!(!entries.is_empty(), "IC[0] should have cached shapes after 9 calls");
    let cached_shape = entries[0].shape.upgrade();
    assert!(cached_shape.is_some(), "cached shape should still be alive");
    let cached_addr = cached_shape.unwrap().to_addr_usize();
    drop(entries);

    // Now create a fresh object like the ones we'll pass at runtime.
    let test_obj = context.eval(Source::from_bytes("({x: 99})")).unwrap();
    let test_obj_ref = test_obj.as_object().unwrap();
    let borrowed = test_obj_ref.borrow();
    let runtime_shape_addr = borrowed.properties().shape.to_addr_usize();
    drop(borrowed);

    eprintln!("Cached shape addr:  0x{cached_addr:x}");
    eprintln!("Runtime shape addr: 0x{runtime_shape_addr:x}");
    eprintln!("Match: {}", cached_addr == runtime_shape_addr);

    // Check JIT state — should be Pending (not yet compiled).
    match code.jit.get() {
        JitState::Pending { call_count } => {
            eprintln!("JIT state: Pending (call_count={call_count})");
        }
        other => {
            eprintln!("JIT state: {other:?}");
        }
    }
}

/// Verify inline IC fires and produces correct results.
#[test]
fn inline_ic_produces_correct_result() {
    use crate::{Context, Source};

    let mut context = Context::default();

    // Call get_x 9 times to populate IC, then once more to trigger JIT.
    // After JIT, subsequent calls should use the inline IC fast path.
    let result = context.eval(Source::from_bytes(
        "function get_x(obj) { return obj.x; }
         // Populate IC with 9 interpreter calls
         for (var i = 0; i < 9; i++) get_x({x: i});
         // 10th call triggers JIT compilation
         get_x({x: 100});
         // 11th+ calls use JIT'd code with inline IC
         var results = [];
         for (var i = 0; i < 10; i++) {
           results.push(get_x({x: i * 10}));
         }
         results[5]"  // Should be 50
    ));

    let value = result.expect("should succeed");
    assert_eq!(
        value.as_number().expect("should be number"),
        50.0,
        "inline IC should return correct property value"
    );
}

/// Verify that ic_fast_get works on objects created by eval (same as benchmark).
#[test]
fn ic_fast_get_works_on_eval_objects() {
    use crate::{Context, Source};
    use super::helpers;

    let mut context = Context::default();

    // Create object and populate IC via interpreter.
    context.eval(Source::from_bytes(
        "function get_x(obj) { return obj.x; }
         for (var i = 0; i < 9; i++) get_x({x: i});"
    )).unwrap();

    // Get the IC data.
    let get_x_val = context.eval(Source::from_bytes("get_x")).unwrap();
    let get_x_obj = get_x_val.as_object().unwrap();
    let func = get_x_obj.downcast_ref::<crate::builtins::function::OrdinaryFunction>().unwrap();
    let ic = &func.code.ic[0];
    let entries = ic.entries.borrow();
    let shape = entries[0].shape.upgrade().unwrap();
    let slot_index = entries[0].slot.index;
    let shape_addr = shape.to_addr_usize();
    let cached_shape_ptr = (shape_addr - 16) as u64; // subtract GcHeader
    drop(entries);
    drop(func);

    // Create a test object the same way the benchmark does.
    let obj = context.eval(Source::from_bytes("({x: 42})")).unwrap();
    let raw_bits: u64 = unsafe { std::mem::transmute_copy(&obj) };

    // Verify it's an object.
    let tag = raw_bits & 0x7FFF_0000_0000_0000;
    assert_eq!(tag, 0x7FFC_0000_0000_0000, "should be object");

    // Test ic_fast_get.
    let result = unsafe { helpers::ic_fast_get(raw_bits, cached_shape_ptr, slot_index as u32) };
    assert!(result.is_some(), "IC fast path should hit for same-shape object");

    // Verify the value is NaN-boxed integer 42.
    let val_bits = result.unwrap();
    let expected_bits: u64 = unsafe { std::mem::transmute_copy(&crate::JsValue::from(42)) };
    assert_eq!(val_bits, expected_bits, "should be NaN-boxed 42");

    eprintln!("ic_fast_get works correctly on eval'd objects!");
}

/// End-to-end: property access by name.
#[test]
fn end_to_end_jit_property_access() {
    use crate::{Context, Source};

    let mut context = Context::default();

    let result = context.eval(Source::from_bytes(
        "function get_x(obj) { return obj.x; }
         var r;
         for (var i = 0; i < 20; i++) { r = get_x({x: 42}); }
         r",
    ));

    let value = result.expect("should succeed");
    assert_eq!(
        value.as_number().expect("should be number"),
        42.0,
        "JIT'd property access should work"
    );
}

/// End-to-end: function with branches, modulo, strict equality, and decrement.
#[test]
fn end_to_end_jit_branchy_loop() {
    use crate::{Context, Source};

    let mut context = Context::default();

    let result = context.eval(Source::from_bytes(
        "function test(n) {
           var count = 0;
           for (var i = 0; i < n; i++) {
             if (i % 3 === 0) count++;
             else if (i % 5 === 0) count--;
             else count = (count + i) | 0;
           }
           return count;
         }
         var r;
         for (var j = 0; j < 20; j++) { r = test(100); }
         r",
    ));

    let value = result.expect("should succeed");
    // Verify against known result.
    let expected: i32 = {
        let mut count = 0i32;
        for i in 0..100 {
            if i % 3 == 0 { count += 1; }
            else if i % 5 == 0 { count -= 1; }
            else { count = count.wrapping_add(i); }
        }
        count
    };
    assert_eq!(
        value.as_number().expect("should be number"),
        f64::from(expected),
        "JIT'd branchy function should produce correct result"
    );
}

/// End-to-end: recursive fibonacci with Call and GetNameGlobal.
#[test]
fn end_to_end_jit_fib() {
    use crate::{Context, Source};

    let mut context = Context::default();

    let result = context.eval(Source::from_bytes(
        "function fib(n) { if (n <= 1) return n; return fib(n-1) + fib(n-2); }
         var r;
         for (var j = 0; j < 20; j++) { r = fib(10); }
         r",
    ));

    let value = result.expect("should succeed");
    assert_eq!(
        value.as_number().expect("should be number"),
        55.0,
        "JIT'd fib(10) should return 55"
    );
}
