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
    use crate::vm::opcode::{BytecodeEmitter, IndexOperand, RegisterOperand};
    use boa_string::JsString;

    let mut emitter = BytecodeEmitter::new();
    emitter.emit_get_argument(IndexOperand::new(0), RegisterOperand::new(1));
    emitter.emit_check_return();
    emitter.emit_return();

    let mut code = CodeBlock::new(JsString::from("test"), 0, false);
    code.bytecode = emitter.into_bytecode();
    code.register_count = 2;

    assert!(
        !super::can_compile(&code),
        "code with GetArgument should not be compilable yet"
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

/// Verify that functions with unsupported opcodes still work via the interpreter.
#[test]
fn unsupported_falls_back_to_interpreter() {
    use crate::{Context, Source};

    let mut context = Context::default();

    // This function uses GetArgument + Add, which aren't JIT-supported yet.
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
        "interpreted function should still work"
    );
}
