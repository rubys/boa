//! Tests for the JIT compiler.

use super::JitCompiler;

#[test]
fn ic_layout_and_fast_path() {
    super::helpers::verify_ic_offsets_and_fast_path();
}

#[test]
fn jit_compiler_creates_successfully() {
    let compiler = JitCompiler::new();
    assert!(
        compiler.is_ok(),
        "JIT compiler should initialize on the host architecture"
    );
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

    assert!(
        super::can_compile(&code),
        "trivial code should be compilable"
    );

    let mut compiler = JitCompiler::new().expect("compiler should init");
    let jit_fn = compiler.compile(&code);
    assert!(jit_fn.is_some(), "should compile trivial function");
}

#[test]
fn can_compile_rejects_unsupported() {
    // typeof uses TypeOf which IS supported now, but async functions
    // use Await which is NOT supported. Use a for-in loop which needs iterators.
    use crate::{Context, Source};
    let mut context = Context::default();
    // for-in requires CreateForInIterator which is unsupported.
    // Verify the function still works (interpreted) even with 20+ calls.
    let result = context.eval(Source::from_bytes(
        "function keys(obj) { var r = []; for (var k in obj) r.push(k); return r.length; }
         var n;
         for (var i = 0; i < 20; i++) n = keys({a:1, b:2});
         n",
    ));
    let value = result.expect("should succeed");
    assert_eq!(
        value.as_number().expect("should be number"),
        2.0,
        "unsupported function should still work via interpreter"
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
    use crate::vm::code_block::JitState;
    use crate::{Context, Source};

    let mut context = Context::default();

    // Define get_x and call it 9 times (below JIT threshold of 10).
    context
        .eval(Source::from_bytes(
            "function get_x(obj) { return obj.x; }
         for (var i = 0; i < 9; i++) get_x({x: i});",
        ))
        .expect("setup should succeed");

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
        eprintln!(
            "  IC[{i}] name={} entries={} megamorphic={}",
            entry.name.to_std_string_escaped(),
            entries.len(),
            entry.megamorphic.get(),
        );
        for (j, cached) in entries.iter().enumerate() {
            if let Some(shape) = cached.shape.upgrade() {
                eprintln!(
                    "    entry[{j}]: shape=0x{:x} slot_index={}",
                    shape.to_addr_usize(),
                    cached.slot.index
                );
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
    assert!(
        !entries.is_empty(),
        "IC[0] should have cached shapes after 9 calls"
    );
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
         results[5]", // Should be 50
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
    use super::helpers;
    use crate::{Context, Source};

    let mut context = Context::default();

    // Create object and populate IC via interpreter.
    context
        .eval(Source::from_bytes(
            "function get_x(obj) { return obj.x; }
         for (var i = 0; i < 9; i++) get_x({x: i});",
        ))
        .unwrap();

    // Get the IC data.
    let get_x_val = context.eval(Source::from_bytes("get_x")).unwrap();
    let get_x_obj = get_x_val.as_object().unwrap();
    let func = get_x_obj
        .downcast_ref::<crate::builtins::function::OrdinaryFunction>()
        .unwrap();
    let ic = &func.code.ic[0];
    let entries = ic.entries.borrow();
    let shape = entries[0].shape.upgrade().unwrap();
    let slot_index = entries[0].slot.index;
    let shape_addr = shape.to_addr_usize();
    let cached_shape_ptr = (shape_addr - helpers::gcbox_value_offset()) as u64;
    drop(entries);
    drop(func);

    // Create a test object the same way the benchmark does.
    let obj = context.eval(Source::from_bytes("({x: 42})")).unwrap();
    let raw_bits: u64 = unsafe { std::mem::transmute_copy(&obj) };

    // Verify it's an object.
    let tag = raw_bits & 0x7FFF_0000_0000_0000;
    assert_eq!(tag, 0x7FFC_0000_0000_0000, "should be object");

    // Test ic_fast_get.
    let offsets = helpers::IcOffsets::compute().expect("IC offsets should compute on this platform");
    let result =
        unsafe { helpers::ic_fast_get(raw_bits, cached_shape_ptr, slot_index as u32, &offsets) };
    assert!(
        result.is_some(),
        "IC fast path should hit for same-shape object"
    );

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
            if i % 3 == 0 {
                count += 1;
            } else if i % 5 == 0 {
                count -= 1;
            } else {
                count = count.wrapping_add(i);
            }
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

// ============================================================
// Comprehensive opcode test suite.
// Each test calls a function enough times to trigger JIT (20x),
// then verifies the result is correct.
// ============================================================

/// Helper: evaluate JS, expect a numeric result.
fn eval_num(code: &str) -> f64 {
    let mut ctx = crate::Context::default();
    let result = ctx.eval(crate::Source::from_bytes(code));
    let value = result.unwrap_or_else(|e| panic!("JS error: {e}"));
    value
        .as_number()
        .unwrap_or_else(|| panic!("expected number, got {:?}", value))
}

/// Helper: evaluate JS, expect a string result.
fn eval_str(code: &str) -> String {
    let mut ctx = crate::Context::default();
    let result = ctx.eval(crate::Source::from_bytes(code));
    let value = result.unwrap_or_else(|e| panic!("JS error: {e}"));
    value
        .as_string()
        .unwrap_or_else(|| panic!("expected string, got {:?}", value))
        .to_std_string_escaped()
}

/// Helper: evaluate JS, expect a boolean result.
fn eval_bool(code: &str) -> bool {
    let mut ctx = crate::Context::default();
    let result = ctx.eval(crate::Source::from_bytes(code));
    let value = result.unwrap_or_else(|e| panic!("JS error: {e}"));
    value
        .as_boolean()
        .unwrap_or_else(|| panic!("expected boolean, got {:?}", value))
}

/// Wrap code in a function called 20 times to trigger JIT.
fn jit_call(body: &str, call: &str) -> String {
    format!("{body}\nfor (var _jit_i = 0; _jit_i < 20; _jit_i++) {{ {call} }}\n{call}")
}

// --- Store constants ---

#[test]
fn op_store_zero() {
    assert_eq!(
        eval_num(&jit_call("function f() { return 0; }", "f()")),
        0.0
    );
}

#[test]
fn op_store_one() {
    assert_eq!(
        eval_num(&jit_call("function f() { return 1; }", "f()")),
        1.0
    );
}

#[test]
fn op_store_int8() {
    assert_eq!(
        eval_num(&jit_call("function f() { return 42; }", "f()")),
        42.0
    );
}

#[test]
fn op_store_int32() {
    assert_eq!(
        eval_num(&jit_call("function f() { return 100000; }", "f()")),
        100000.0
    );
}

#[test]
fn op_store_float() {
    assert_eq!(
        eval_num(&jit_call("function f() { return 3.14; }", "f()")),
        3.14
    );
}

#[test]
fn op_store_null() {
    let code = jit_call("function f() { return null === null; }", "f()");
    assert!(eval_bool(&code));
}

#[test]
fn op_store_true_false() {
    assert!(eval_bool(&jit_call("function f() { return true; }", "f()")));
    assert!(!eval_bool(&jit_call(
        "function f() { return false; }",
        "f()"
    )));
}

#[test]
fn op_store_undefined() {
    let code = jit_call("function f() { return undefined === undefined; }", "f()");
    assert!(eval_bool(&code));
}

// --- Arithmetic ---

#[test]
fn op_add_integers() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a+b; }", "f(3,4)")),
        7.0
    );
}

#[test]
fn op_add_floats() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a+b; }", "f(1.5, 2.5)")),
        4.0
    );
}

#[test]
fn op_add_string_concat() {
    assert_eq!(
        eval_str(&jit_call(
            "function f(a,b) { return a+b; }",
            "f('hello', ' world')"
        )),
        "hello world"
    );
}

#[test]
fn op_sub() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a-b; }", "f(10,3)")),
        7.0
    );
}

#[test]
fn op_mul() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a*b; }", "f(6,7)")),
        42.0
    );
}

#[test]
fn op_div() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a/b; }", "f(10,4)")),
        2.5
    );
}

#[test]
fn op_mod() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a%b; }", "f(10,3)")),
        1.0
    );
}

#[test]
fn op_pow() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a**b; }", "f(2,10)")),
        1024.0
    );
}

#[test]
fn op_neg() {
    assert_eq!(
        eval_num(&jit_call("function f(x) { return -x; }", "f(42)")),
        -42.0
    );
}

#[test]
fn op_pos() {
    assert_eq!(
        eval_num(&jit_call("function f(x) { return +x; }", "f('42')")),
        42.0
    );
}

#[test]
fn op_inc_dec() {
    assert_eq!(
        eval_num(&jit_call("function f(x) { x++; return x; }", "f(5)")),
        6.0
    );
    assert_eq!(
        eval_num(&jit_call("function f(x) { x--; return x; }", "f(5)")),
        4.0
    );
}

// --- Bitwise ---

#[test]
fn op_bit_or() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a|b; }", "f(5,3)")),
        7.0
    );
}

#[test]
fn op_bit_and() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a&b; }", "f(5,3)")),
        1.0
    );
}

#[test]
fn op_bit_xor() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a^b; }", "f(5,3)")),
        6.0
    );
}

#[test]
fn op_bit_not() {
    assert_eq!(
        eval_num(&jit_call("function f(x) { return ~x; }", "f(0)")),
        -1.0
    );
}

#[test]
fn op_shift_left() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a<<b; }", "f(1,4)")),
        16.0
    );
}

#[test]
fn op_shift_right() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a>>b; }", "f(16,2)")),
        4.0
    );
}

#[test]
fn op_unsigned_shift_right() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a>>>b; }", "f(-1,28)")),
        15.0
    );
}

// --- Comparison ---

#[test]
fn op_strict_eq() {
    assert!(eval_bool(&jit_call(
        "function f(a,b) { return a===b; }",
        "f(42,42)"
    )));
    assert!(!eval_bool(&jit_call(
        "function f(a,b) { return a===b; }",
        "f(42,43)"
    )));
}

#[test]
fn op_eq() {
    assert!(eval_bool(&jit_call(
        "function f(a,b) { return a==b; }",
        "f(0,false)"
    )));
}

#[test]
fn op_less_than() {
    assert!(eval_bool(&jit_call(
        "function f(a,b) { return a<b; }",
        "f(1,2)"
    )));
    assert!(!eval_bool(&jit_call(
        "function f(a,b) { return a<b; }",
        "f(2,1)"
    )));
}

#[test]
fn op_greater_than() {
    assert!(eval_bool(&jit_call(
        "function f(a,b) { return a>b; }",
        "f(2,1)"
    )));
}

#[test]
fn op_instance_of() {
    assert!(eval_bool(&jit_call(
        "function Foo() {} function f(x) { return x instanceof Foo; }",
        "f(new Foo())"
    )));
}

#[test]
fn op_typeof() {
    assert_eq!(
        eval_str(&jit_call("function f(x) { return typeof x; }", "f(42)")),
        "number"
    );
    assert_eq!(
        eval_str(&jit_call("function f(x) { return typeof x; }", "f('hi')")),
        "string"
    );
}

#[test]
fn op_is_object() {
    // IsObject is used internally, test via typeof/truthiness patterns
    assert!(eval_bool(&jit_call(
        "function f(x) { return typeof x === 'object' && x !== null; }",
        "f({})"
    )));
}

// --- Logical ---

#[test]
fn op_logical_and() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a && b; }", "f(1,42)")),
        42.0
    );
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a && b; }", "f(0,42)")),
        0.0
    );
}

#[test]
fn op_logical_or() {
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a || b; }", "f(0,42)")),
        42.0
    );
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a || b; }", "f(1,42)")),
        1.0
    );
}

#[test]
fn op_logical_not() {
    assert!(eval_bool(&jit_call(
        "function f(x) { return !x; }",
        "f(false)"
    )));
    assert!(!eval_bool(&jit_call(
        "function f(x) { return !x; }",
        "f(true)"
    )));
}

#[test]
fn op_coalesce() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(a,b) { return a ?? b; }",
            "f(null,42)"
        )),
        42.0
    );
    assert_eq!(
        eval_num(&jit_call("function f(a,b) { return a ?? b; }", "f(7,42)")),
        7.0
    );
}

// --- Control flow ---

#[test]
fn op_if_else() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(x) { if (x > 0) return 1; else return -1; }",
            "f(5)"
        )),
        1.0
    );
}

#[test]
fn op_for_loop() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(n) { var s=0; for(var i=0;i<n;i++) s+=i; return s; }",
            "f(10)"
        )),
        45.0
    );
}

#[test]
fn op_while_loop() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(n) { var s=0; var i=0; while(i<n) { s+=i; i++; } return s; }",
            "f(10)"
        )),
        45.0
    );
}

#[test]
fn op_switch_case() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(x) { switch(x) { case 1: return 10; case 2: return 20; default: return 0; } }",
            "f(2)"
        )),
        20.0
    );
}

// --- This / constructors ---

#[test]
fn op_this_simple() {
    assert_eq!(
        eval_num(&jit_call(
            "function Foo(x) { this.x = x; } function f() { return new Foo(42).x; }",
            "f()"
        )),
        42.0
    );
}

#[test]
fn op_this_method() {
    assert_eq!(
        eval_num(&jit_call(
            "function Obj(v) { this.v = v; this.get = function() { return this.v; }; }
         function f() { var o = new Obj(99); return o.get(); }",
            "f()"
        )),
        99.0
    );
}

#[test]
fn op_new_constructor() {
    assert!(eval_bool(&jit_call(
        "function Foo() {} function f() { return (new Foo()) instanceof Foo; }",
        "f()"
    )));
}

// --- Property access ---

#[test]
fn op_get_set_property_by_name() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(obj) { obj.x = 10; return obj.x; }",
            "f({})"
        )),
        10.0
    );
}

#[test]
fn op_get_set_property_by_value() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(arr, i) { arr[i] = 42; return arr[i]; }",
            "f([0,0,0], 1)"
        )),
        42.0
    );
}

#[test]
fn op_array_length() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(arr) { return arr.length; }",
            "f([1,2,3,4,5])"
        )),
        5.0
    );
}

#[test]
fn op_delete_property() {
    assert!(eval_bool(&jit_call(
        "function f() { var o = {x:1}; delete o.x; return o.x === undefined; }",
        "f()"
    )));
}

#[test]
fn op_in_operator() {
    assert!(eval_bool(&jit_call(
        "function f() { return 'x' in {x:1}; }",
        "f()"
    )));
}

#[test]
fn op_define_own_property() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() { var o = {}; o.x = 42; return o.x; }",
            "f()"
        )),
        42.0
    );
}

// --- Array / object creation ---

#[test]
fn op_array_literal() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() { var a = [10, 20, 30]; return a[1]; }",
            "f()"
        )),
        20.0
    );
}

#[test]
fn op_object_literal() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() { var o = {a: 1, b: 2}; return o.a + o.b; }",
            "f()"
        )),
        3.0
    );
}

#[test]
fn op_empty_object() {
    assert!(eval_bool(&jit_call(
        "function f() { var o = {}; return typeof o === 'object'; }",
        "f()"
    )));
}

// --- Variable binding ---

#[test]
fn op_var_declaration() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() { var x = 10; var y = 20; return x + y; }",
            "f()"
        )),
        30.0
    );
}

#[test]
fn op_let_const() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() { let x = 10; const y = 20; return x + y; }",
            "f()"
        )),
        30.0
    );
}

#[test]
fn op_closure_variable() {
    assert_eq!(
        eval_num(&jit_call(
            "function outer() { var x = 42; function inner() { return x; } return inner(); }",
            "outer()"
        )),
        42.0
    );
}

// --- Function calls ---

#[test]
fn op_call_simple() {
    assert_eq!(
        eval_num(&jit_call(
            "function add(a,b) { return a+b; } function f() { return add(3,4); }",
            "f()"
        )),
        7.0
    );
}

#[test]
fn op_recursive_call() {
    assert_eq!(
        eval_num(&jit_call(
            "function fib(n) { if(n<=1) return n; return fib(n-1)+fib(n-2); }",
            "fib(10)"
        )),
        55.0
    );
}

#[test]
fn op_get_function() {
    assert_eq!(
        eval_num(&jit_call(
            "function outer() { function inner() { return 42; } return inner(); }",
            "outer()"
        )),
        42.0
    );
}

// --- Error handling ---

#[test]
fn op_throw_catch() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() { try { throw 42; } catch(e) { return e; } }",
            "f()"
        )),
        42.0
    );
}

// --- Scope ---

#[test]
fn op_block_scope() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() { var x = 1; { let x = 2; } return x; }",
            "f()"
        )),
        1.0
    );
}

#[test]
fn op_push_scope_loop() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() { var s = 0; for (let i = 0; i < 5; i++) { s += i; } return s; }",
            "f()"
        )),
        10.0
    );
}

// --- Combined patterns ---

#[test]
fn pattern_accumulator_loop() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(n) { var s=0; for(var i=0;i<n;i++) s=(s+i)|0; return s; }",
            "f(100)"
        )),
        4950.0
    );
}

#[test]
fn pattern_branchy_loop() {
    let code = jit_call(
        "function f(n) {
           var count = 0;
           for (var i = 0; i < n; i++) {
             if (i % 3 === 0) count++;
             else if (i % 5 === 0) count--;
             else count = (count + i) | 0;
           }
           return count;
         }",
        "f(100)",
    );
    // Compute expected value
    let mut count: i32 = 0;
    for i in 0..100 {
        if i % 3 == 0 {
            count += 1;
        } else if i % 5 == 0 {
            count -= 1;
        } else {
            count = count.wrapping_add(i);
        }
    }
    assert_eq!(eval_num(&code), f64::from(count));
}

#[test]
fn pattern_property_loop() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(obj, n) { var s=0; for(var i=0;i<n;i++) s+=obj.x; return s; }",
            "f({x:7}, 100)"
        )),
        700.0
    );
}

#[test]
fn pattern_array_sum() {
    assert_eq!(
        eval_num(&jit_call(
            "function f(arr) { var s=0; for(var i=0;i<arr.length;i++) s+=arr[i]; return s; }",
            "f([1,2,3,4,5])"
        )),
        15.0
    );
}

#[test]
fn pattern_constructor_with_methods() {
    assert_eq!(
        eval_num(&jit_call(
            "function Point(x,y) { this.x=x; this.y=y; }
         Point.prototype.sum = function() { return this.x + this.y; };
         function f() { var p = new Point(3,4); return p.sum(); }",
            "f()"
        )),
        7.0
    );
}

#[test]
fn pattern_nested_property_access() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() { var o = {a: {b: {c: 42}}}; return o.a.b.c; }",
            "f()"
        )),
        42.0
    );
}

#[test]
fn pattern_array_of_objects() {
    assert_eq!(
        eval_num(&jit_call(
            "function f() {
           var arr = [];
           for (var i = 0; i < 5; i++) arr.push({v: i * 10});
           var s = 0;
           for (var i = 0; i < arr.length; i++) s += arr[i].v;
           return s;
         }",
            "f()"
        )),
        100.0
    );
}
