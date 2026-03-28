//! JIT compilation via Cranelift.
//!
//! This module provides a JIT compiler that translates Boa bytecode into native
//! machine code using Cranelift. It operates at function granularity: a function
//! is either fully JIT-compiled or interpreted. If a [`CodeBlock`] contains any
//! opcode that the JIT does not yet support, it falls back to the interpreter.

mod compiler;
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
///
/// This list grows over time. When all opcodes in a function are supported,
/// `can_compile` returns `true` and the function becomes eligible for JIT.
fn is_supported_opcode(opcode: Opcode) -> bool {
    matches!(
        opcode,
        // Data movement
        Opcode::Move
            | Opcode::StoreZero
            | Opcode::StoreOne
            | Opcode::StoreInt8
            | Opcode::StoreInt16
            | Opcode::StoreInt32
            | Opcode::GetArgument
            // Stack/accumulator
            | Opcode::SetAccumulator
            | Opcode::PushFromRegister
            | Opcode::PopIntoRegister
            // Arithmetic
            | Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Mod
            | Opcode::Pow
            | Opcode::BitOr
            | Opcode::BitAnd
            | Opcode::BitXor
            | Opcode::ShiftLeft
            | Opcode::ShiftRight
            | Opcode::UnsignedShiftRight
            | Opcode::Inc
            | Opcode::Dec
            // Comparison
            | Opcode::StrictEq
            | Opcode::StrictNotEq
            | Opcode::Eq
            | Opcode::NotEq
            | Opcode::GreaterThan
            | Opcode::GreaterThanOrEq
            | Opcode::LessThan
            | Opcode::LessThanOrEq
            // Control flow
            | Opcode::Jump
            | Opcode::JumpIfTrue
            | Opcode::JumpIfFalse
            | Opcode::JumpIfNotLessThan
            | Opcode::JumpIfNotLessThanOrEqual
            | Opcode::IncrementLoopIteration
            // Return
            | Opcode::CheckReturn
            | Opcode::Return
    )
}
