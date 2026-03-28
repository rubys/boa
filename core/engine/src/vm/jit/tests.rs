//! Tests for the JIT compiler.

use super::JitCompiler;

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

    // Property access is not JIT-supported.
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
        "interpreted function should still work"
    );
}
