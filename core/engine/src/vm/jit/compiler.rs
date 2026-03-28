//! Cranelift-based JIT compiler for Boa bytecode.
//!
//! The compiler translates a [`CodeBlock`] into a native function that operates
//! on the same `Context` and stack that the interpreter uses. This first version
//! eliminates dispatch overhead but still calls back into Rust helpers for all
//! value operations.

use cranelift_codegen::{
    ir::{types, AbiParam, Function, InstBuilder, Type, UserFuncName, Value},
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use std::ops::ControlFlow;

use crate::{
    Context,
    vm::{
        CodeBlock, CompletionRecord,
        opcode::{Instruction, InstructionIterator},
    },
};

use super::helpers;

/// A compiled JIT function that can be called in place of interpreting bytecode.
///
/// The function signature matches what the interpreter loop does: it takes a
/// `&mut Context` and returns a `ControlFlow<CompletionRecord>`.
pub(crate) type JitFn = fn(&mut Context) -> ControlFlow<CompletionRecord>;

/// Holds the [`FuncId`]s for the runtime helper functions that JIT code calls.
struct HelperFuncs {
    store_zero: FuncId,
    store_one: FuncId,
    store_int8: FuncId,
    store_int16: FuncId,
    store_int32: FuncId,
    move_: FuncId,
    set_accumulator: FuncId,
    push_from_register: FuncId,
    pop_into_register: FuncId,
    check_return_and_return: FuncId,
}

/// The JIT compiler. Holds the Cranelift module and compiled function cache.
// JITModule doesn't implement Debug, so we implement it manually.
pub(crate) struct JitCompiler {
    /// The Cranelift JIT module that owns the generated code memory.
    module: JITModule,
    /// ISA-specific pointer type (i64 on x86-64).
    ptr_type: Type,
    /// Pre-declared helper function IDs.
    helpers: HelperFuncs,
    /// Counter for unique function names.
    func_counter: u32,
}

impl std::fmt::Debug for JitCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompiler")
            .field("func_counter", &self.func_counter)
            .finish_non_exhaustive()
    }
}

impl JitCompiler {
    /// Create a new JIT compiler for the host architecture.
    pub(crate) fn new() -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| e.to_string())?;

        let isa_builder =
            cranelift_native::builder().map_err(|e| format!("host ISA not supported: {e}"))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        let ptr_type = isa.pointer_type();

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register helper function symbols so Cranelift can link calls to them.
        builder.symbol("jit_store_zero", helpers::jit_store_zero as *const u8);
        builder.symbol("jit_store_one", helpers::jit_store_one as *const u8);
        builder.symbol("jit_store_int8", helpers::jit_store_int8 as *const u8);
        builder.symbol("jit_store_int16", helpers::jit_store_int16 as *const u8);
        builder.symbol("jit_store_int32", helpers::jit_store_int32 as *const u8);
        builder.symbol("jit_move", helpers::jit_move as *const u8);
        builder.symbol(
            "jit_set_accumulator",
            helpers::jit_set_accumulator as *const u8,
        );
        builder.symbol(
            "jit_push_from_register",
            helpers::jit_push_from_register as *const u8,
        );
        builder.symbol(
            "jit_pop_into_register",
            helpers::jit_pop_into_register as *const u8,
        );
        builder.symbol(
            "jit_check_return_and_return",
            helpers::jit_check_return_and_return as *const u8,
        );

        let mut module = JITModule::new(builder);

        // Declare all helper functions in the module.
        let helpers = Self::declare_helpers(&mut module, ptr_type)?;

        Ok(Self {
            module,
            ptr_type,
            helpers,
            func_counter: 0,
        })
    }

    /// Declare the runtime helper functions in the Cranelift module.
    fn declare_helpers(module: &mut JITModule, ptr: Type) -> Result<HelperFuncs, String> {
        // Helper: (ctx: ptr, dst: i32) -> void
        let sig_ctx_u32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(ptr));
            sig.params.push(AbiParam::new(types::I32));
            sig
        };

        // Helper: (ctx: ptr, dst: i32, value: i32) -> void
        let sig_ctx_u32_i32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(ptr));
            sig.params.push(AbiParam::new(types::I32));
            sig.params.push(AbiParam::new(types::I32));
            sig
        };

        // Helper: (ctx: ptr, dst: i32, src: i32) -> void
        let sig_ctx_u32_u32 = &sig_ctx_u32_i32;

        // Helper: (ctx: ptr) -> i64
        let sig_ctx_ret64 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(ptr));
            sig.returns.push(AbiParam::new(types::I64));
            sig
        };

        let d = |module: &mut JITModule, name, sig: &cranelift_codegen::ir::Signature| {
            module
                .declare_function(name, Linkage::Import, sig)
                .map_err(|e| e.to_string())
        };

        Ok(HelperFuncs {
            store_zero: d(module, "jit_store_zero", &sig_ctx_u32)?,
            store_one: d(module, "jit_store_one", &sig_ctx_u32)?,
            store_int8: d(module, "jit_store_int8", &sig_ctx_u32_i32)?,
            store_int16: d(module, "jit_store_int16", &sig_ctx_u32_i32)?,
            store_int32: d(module, "jit_store_int32", &sig_ctx_u32_i32)?,
            move_: d(module, "jit_move", sig_ctx_u32_u32)?,
            set_accumulator: d(module, "jit_set_accumulator", &sig_ctx_u32)?,
            push_from_register: d(module, "jit_push_from_register", &sig_ctx_u32)?,
            pop_into_register: d(module, "jit_pop_into_register", &sig_ctx_u32)?,
            check_return_and_return: d(module, "jit_check_return_and_return", &sig_ctx_ret64)?,
        })
    }

    /// Compile a [`CodeBlock`] into a native function.
    ///
    /// Returns `None` if the code block contains unsupported opcodes.
    pub(crate) fn compile(&mut self, code: &CodeBlock) -> Option<JitFn> {
        if !super::can_compile(code) {
            return None;
        }

        // Each compiled function needs a unique name.
        let name = format!("jit_fn_{}", self.func_counter);
        self.func_counter += 1;

        // Signature: fn(ctx: *mut Context) -> i64
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(self.ptr_type)); // &mut Context
        sig.returns.push(AbiParam::new(types::I64)); // control flow tag

        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .expect("declare function");

        let mut func = Function::with_name_signature(
            UserFuncName::testcase(&name),
            sig.clone(),
        );

        let mut func_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let ctx_ptr = builder.block_params(entry_block)[0];

            self.translate_body(&mut builder, ctx_ptr, code);

            builder.finalize();
        }

        let mut ctx = cranelift_codegen::Context::for_function(func);
        self.module
            .define_function(func_id, &mut ctx)
            .expect("define function");
        self.module.clear_context(&mut ctx);
        self.module.finalize_definitions().expect("finalize");

        let code_ptr = self.module.get_finalized_function(func_id);

        // SAFETY: The generated code matches the JitFn signature — it takes
        // a pointer-sized argument and returns an i64. The Cranelift module
        // owns the executable memory and keeps it valid for the module's lifetime.
        let jit_fn: JitFn = unsafe { std::mem::transmute(code_ptr) };
        Some(jit_fn)
    }

    /// Translate the bytecode body into Cranelift IR.
    fn translate_body(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        ctx_ptr: Value,
        code: &CodeBlock,
    ) {
        // Import helper function references into this function.
        let store_zero_ref = self
            .module
            .declare_func_in_func(self.helpers.store_zero, builder.func);
        let store_one_ref = self
            .module
            .declare_func_in_func(self.helpers.store_one, builder.func);
        let store_int8_ref = self
            .module
            .declare_func_in_func(self.helpers.store_int8, builder.func);
        let store_int16_ref = self
            .module
            .declare_func_in_func(self.helpers.store_int16, builder.func);
        let store_int32_ref = self
            .module
            .declare_func_in_func(self.helpers.store_int32, builder.func);
        let move_ref = self
            .module
            .declare_func_in_func(self.helpers.move_, builder.func);
        let set_acc_ref = self
            .module
            .declare_func_in_func(self.helpers.set_accumulator, builder.func);
        let push_reg_ref = self
            .module
            .declare_func_in_func(self.helpers.push_from_register, builder.func);
        let pop_reg_ref = self
            .module
            .declare_func_in_func(self.helpers.pop_into_register, builder.func);
        let ret_ref = self
            .module
            .declare_func_in_func(self.helpers.check_return_and_return, builder.func);

        let iter = InstructionIterator::new(&code.bytecode);
        let mut terminated = false;
        for (_pc, _opcode, instruction) in iter {
            if terminated {
                // After a Return, remaining bytecode is unreachable.
                break;
            }
            match instruction {
                Instruction::StoreZero { dst } => {
                    let dst_val = builder.ins().iconst(types::I32, i64::from(u32::from(dst)));
                    builder.ins().call(store_zero_ref, &[ctx_ptr, dst_val]);
                }
                Instruction::StoreOne { dst } => {
                    let dst_val = builder.ins().iconst(types::I32, i64::from(u32::from(dst)));
                    builder.ins().call(store_one_ref, &[ctx_ptr, dst_val]);
                }
                Instruction::StoreInt8 { dst, value } => {
                    let dst_val = builder.ins().iconst(types::I32, i64::from(u32::from(dst)));
                    let val = builder.ins().iconst(types::I32, i64::from(value));
                    builder.ins().call(store_int8_ref, &[ctx_ptr, dst_val, val]);
                }
                Instruction::StoreInt16 { dst, value } => {
                    let dst_val = builder.ins().iconst(types::I32, i64::from(u32::from(dst)));
                    let val = builder.ins().iconst(types::I32, i64::from(value));
                    builder
                        .ins()
                        .call(store_int16_ref, &[ctx_ptr, dst_val, val]);
                }
                Instruction::StoreInt32 { dst, value } => {
                    let dst_val = builder.ins().iconst(types::I32, i64::from(u32::from(dst)));
                    let val = builder.ins().iconst(types::I32, i64::from(value));
                    builder
                        .ins()
                        .call(store_int32_ref, &[ctx_ptr, dst_val, val]);
                }
                Instruction::Move { dst, src } => {
                    let dst_val = builder.ins().iconst(types::I32, i64::from(u32::from(dst)));
                    let src_val = builder.ins().iconst(types::I32, i64::from(u32::from(src)));
                    builder
                        .ins()
                        .call(move_ref, &[ctx_ptr, dst_val, src_val]);
                }
                Instruction::SetAccumulator { src } => {
                    let src_val = builder.ins().iconst(types::I32, i64::from(u32::from(src)));
                    builder.ins().call(set_acc_ref, &[ctx_ptr, src_val]);
                }
                Instruction::PushFromRegister { src } => {
                    let src_val = builder.ins().iconst(types::I32, i64::from(u32::from(src)));
                    builder.ins().call(push_reg_ref, &[ctx_ptr, src_val]);
                }
                Instruction::PopIntoRegister { dst } => {
                    let dst_val = builder.ins().iconst(types::I32, i64::from(u32::from(dst)));
                    builder.ins().call(pop_reg_ref, &[ctx_ptr, dst_val]);
                }
                Instruction::CheckReturn => {
                    // Handled together with Return below.
                }
                Instruction::Return => {
                    let result = builder.ins().call(ret_ref, &[ctx_ptr]);
                    let tag = builder.inst_results(result)[0];
                    builder.ins().return_(&[tag]);
                    terminated = true;
                }
                _ => {
                    // Unsupported opcode — should not happen since can_compile
                    // checked, but emit a "return error" as a safety net.
                    let err = builder.ins().iconst(types::I64, 2);
                    builder.ins().return_(&[err]);
                    return;
                }
            }
        }

        if !terminated {
            // If bytecode doesn't end with Return (shouldn't happen for valid code),
            // return Continue.
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().return_(&[zero]);
        }
    }
}
