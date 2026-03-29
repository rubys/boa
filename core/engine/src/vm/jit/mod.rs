//! JIT compilation via Cranelift.
//!
//! This module provides a JIT compiler that translates Boa bytecode into native
//! machine code using Cranelift. It operates at function granularity: a function
//! is either fully JIT-compiled or interpreted.
//!
//! If a [`CodeBlock`] contains any opcode that the JIT does not yet support, it
//! falls back to the interpreter.
//!
//! Requires a 64-bit target (NaN-boxing layout and pointer assumptions).

#[cfg(not(target_pointer_width = "64"))]
compile_error!("JIT compilation requires a 64-bit target");

#[cfg(feature = "jsvalue-enum")]
compile_error!("The `jit` feature is incompatible with `jsvalue-enum`; the JIT assumes a NaN-boxed JsValue layout.");

mod compiler;
// pub(crate) when jit-stats needs access from vm/mod.rs
#[cfg(feature = "jit-stats")]
pub(crate) mod helpers;
#[cfg(not(feature = "jit-stats"))]
mod helpers;

#[cfg(test)]
mod tests;

use crate::vm::{
    CodeBlock,
    opcode::{InstructionIterator, Opcode},
};

pub(crate) use compiler::{JitCompiler, JitFn};

/// Check whether a [`CodeBlock`] uses only opcodes the JIT can compile.
pub(crate) fn can_compile(code: &CodeBlock) -> bool {
    let iter = InstructionIterator::new(&code.bytecode);
    for (_, opcode, _) in iter {
        if !is_supported_opcode(opcode) {
            return false;
        }
    }
    true
}

/// The set of opcodes the JIT currently supports.
fn is_supported_opcode(opcode: Opcode) -> bool {
    matches!(
        opcode,
        // === Data movement / constants ===
        Opcode::Move
            | Opcode::StoreZero
            | Opcode::StoreOne
            | Opcode::StoreInt8
            | Opcode::StoreInt16
            | Opcode::StoreInt32
            | Opcode::StoreFloat
            | Opcode::StoreDouble
            | Opcode::StoreNan
            | Opcode::StorePositiveInfinity
            | Opcode::StoreNegativeInfinity
            | Opcode::StoreNull
            | Opcode::StoreTrue
            | Opcode::StoreFalse
            | Opcode::StoreUndefined
            | Opcode::GetArgument
            | Opcode::This
            // === Stack / accumulator ===
            | Opcode::SetAccumulator
            | Opcode::PushFromRegister
            | Opcode::PopIntoRegister
            | Opcode::Pop
            // === Arithmetic ===
            | Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Mod
            | Opcode::Pow
            | Opcode::Neg
            | Opcode::Pos
            | Opcode::Inc
            | Opcode::Dec
            // === Bitwise ===
            | Opcode::BitOr
            | Opcode::BitAnd
            | Opcode::BitXor
            | Opcode::BitNot
            | Opcode::ShiftLeft
            | Opcode::ShiftRight
            | Opcode::UnsignedShiftRight
            // === Comparison ===
            | Opcode::StrictEq
            | Opcode::StrictNotEq
            | Opcode::Eq
            | Opcode::NotEq
            | Opcode::GreaterThan
            | Opcode::GreaterThanOrEq
            | Opcode::LessThan
            | Opcode::LessThanOrEq
            | Opcode::InstanceOf
            // === Type checks ===
            | Opcode::TypeOf
            | Opcode::IsObject
            | Opcode::ValueNotNullOrUndefined
            // === Logical ===
            | Opcode::LogicalAnd
            | Opcode::LogicalOr
            | Opcode::LogicalNot
            | Opcode::Coalesce
            // === Control flow / jumps ===
            | Opcode::Jump
            | Opcode::JumpIfTrue
            | Opcode::JumpIfFalse
            | Opcode::JumpIfNotLessThan
            | Opcode::JumpIfNotLessThanOrEqual
            | Opcode::JumpIfNotGreaterThan
            | Opcode::JumpIfNotGreaterThanOrEqual
            | Opcode::JumpIfNotEqual
            | Opcode::JumpIfNullOrUndefined
            | Opcode::JumpIfNotUndefined
            | Opcode::Case
            | Opcode::IncrementLoopIteration
            // === Variable / binding access ===
            | Opcode::GetName
            | Opcode::GetNameGlobal
            | Opcode::GetNameOrUndefined
            | Opcode::GetNameAndLocator
            | Opcode::GetLocator
            | Opcode::SetName
            | Opcode::SetNameByLocator
            | Opcode::PutLexicalValue
            | Opcode::DefInitVar
            | Opcode::DeleteName
            | Opcode::In
            | Opcode::ToPropertyKey
            // === Property access ===
            | Opcode::GetPropertyByName
            | Opcode::GetPropertyByNameWithThis
            | Opcode::GetLengthProperty
            | Opcode::GetPropertyByValue
            | Opcode::GetPropertyByValuePush
            | Opcode::SetPropertyByValue
            | Opcode::SetPropertyByName
            | Opcode::DefineOwnPropertyByName
            | Opcode::DefineOwnPropertyByValue
            | Opcode::DeletePropertyByName
            | Opcode::DeletePropertyByValue
            | Opcode::GetPrototype
            | Opcode::SetPrototype
            // === Object / array creation ===
            | Opcode::StoreLiteral
            | Opcode::StoreEmptyObject
            | Opcode::StoreNewArray
            | Opcode::StoreRegexp
            | Opcode::PushValueToArray
            | Opcode::PushElisionToArray
            // === Function ===
            | Opcode::GetFunction
            | Opcode::Call
            | Opcode::New
            // === Scope ===
            | Opcode::PushScope
            | Opcode::CreateUnmappedArgumentsObject
            | Opcode::RestParameterInit
            | Opcode::SetRegisterFromAccumulator
            // === Error ===
            | Opcode::Throw
            | Opcode::ThrowNewTypeError
            | Opcode::ThrowNewReferenceError
            | Opcode::ThrowMutateImmutable
            // === Return ===
            | Opcode::CheckReturn
            | Opcode::Return
    )
}
