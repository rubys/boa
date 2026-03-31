//! IR optimization passes for the JIT compiler.
//!
//! These passes operate on the structured IR (`IrFunction`) before
//! Cranelift lowering:
//!
//! 1. **Type propagation** — annotates each instruction with its known
//!    output type, allowing the Cranelift lowering to skip NaN-boxing
//!    tag checks when types are known.
//!
//! Future passes (copy propagation, dead store elimination) will also
//! operate on the IR.

use std::collections::HashMap;

use super::ir::{IrFunction, ValueType};
use crate::vm::opcode::Instruction;

/// Run all optimization passes on the IR.
pub(super) fn optimize(ir: &mut IrFunction) {
    type_propagation(ir);
}

/// Build a type map (PC → ValueType) from the IR's type annotations,
/// for use by the Cranelift lowering.
pub(super) fn build_type_map(ir: &IrFunction) -> HashMap<u32, ValueType> {
    let mut map = HashMap::new();
    for block in &ir.blocks {
        for inst in &block.body {
            if inst.ty != ValueType::Unknown {
                map.insert(inst.pc, inst.ty);
            }
        }
    }
    map
}

/// Type propagation: annotate each instruction with its known output type.
///
/// Seeds types from constants and bitwise operations (which always produce
/// Int32), then propagates through arithmetic when both inputs are known.
fn type_propagation(ir: &mut IrFunction) {
    for block in &mut ir.blocks {
        let mut reg_types: HashMap<u32, ValueType> = HashMap::new();

        for inst in &mut block.body {
            let (produced_type, dst_reg) = infer_type(&inst.instruction, &reg_types);
            inst.ty = produced_type;

            if let Some(dst) = dst_reg {
                if produced_type != ValueType::Unknown {
                    reg_types.insert(dst, produced_type);
                } else {
                    reg_types.remove(&dst);
                }
            }
        }
    }
}

/// Infer the output type of an instruction and return (type, dst_register).
fn infer_type(
    instruction: &Instruction,
    reg_types: &HashMap<u32, ValueType>,
) -> (ValueType, Option<u32>) {
    match instruction {
        // Constants with known types.
        Instruction::StoreZero { dst }
        | Instruction::StoreOne { dst }
        | Instruction::StoreInt8 { dst, .. }
        | Instruction::StoreInt16 { dst, .. }
        | Instruction::StoreInt32 { dst, .. } => (ValueType::Int32, Some(u32::from(*dst))),

        Instruction::StoreFloat { dst, .. }
        | Instruction::StoreDouble { dst, .. }
        | Instruction::StoreNan { dst }
        | Instruction::StorePositiveInfinity { dst }
        | Instruction::StoreNegativeInfinity { dst } => (ValueType::Float64, Some(u32::from(*dst))),

        Instruction::StoreTrue { dst } | Instruction::StoreFalse { dst } => {
            (ValueType::Boolean, Some(u32::from(*dst)))
        }

        // Bitwise ops always produce Int32 (ToInt32 conversion per spec).
        Instruction::BitOr { dst, .. }
        | Instruction::BitAnd { dst, .. }
        | Instruction::BitXor { dst, .. }
        | Instruction::ShiftLeft { dst, .. }
        | Instruction::ShiftRight { dst, .. }
        | Instruction::UnsignedShiftRight { dst, .. } => (ValueType::Int32, Some(u32::from(*dst))),

        // Comparisons always produce Boolean.
        Instruction::StrictEq { dst, .. }
        | Instruction::StrictNotEq { dst, .. }
        | Instruction::Eq { dst, .. }
        | Instruction::NotEq { dst, .. }
        | Instruction::GreaterThan { dst, .. }
        | Instruction::GreaterThanOrEq { dst, .. }
        | Instruction::LessThan { dst, .. }
        | Instruction::LessThanOrEq { dst, .. }
        | Instruction::InstanceOf { dst, .. }
        | Instruction::In { dst, .. } => (ValueType::Boolean, Some(u32::from(*dst))),

        // Arithmetic: Int32 if both inputs are Int32.
        Instruction::Add { dst, lhs, rhs }
        | Instruction::Sub { dst, lhs, rhs }
        | Instruction::Mul { dst, lhs, rhs } => {
            let lt = reg_types
                .get(&u32::from(*lhs))
                .copied()
                .unwrap_or(ValueType::Unknown);
            let rt = reg_types
                .get(&u32::from(*rhs))
                .copied()
                .unwrap_or(ValueType::Unknown);
            let ty = if lt == ValueType::Int32 && rt == ValueType::Int32 {
                ValueType::Int32
            } else {
                ValueType::Unknown
            };
            (ty, Some(u32::from(*dst)))
        }

        // Inc/Dec: Int32 if source is Int32.
        Instruction::Inc { dst, src } | Instruction::Dec { dst, src } => {
            let st = reg_types
                .get(&u32::from(*src))
                .copied()
                .unwrap_or(ValueType::Unknown);
            let ty = if st == ValueType::Int32 {
                ValueType::Int32
            } else {
                ValueType::Unknown
            };
            (ty, Some(u32::from(*dst)))
        }

        // Move propagates the source type.
        Instruction::Move { dst, src } => {
            let ty = reg_types
                .get(&u32::from(*src))
                .copied()
                .unwrap_or(ValueType::Unknown);
            (ty, Some(u32::from(*dst)))
        }

        // Everything else: Unknown, and track dst if it has one.
        _ => {
            let dst = instruction_dst(instruction);
            (ValueType::Unknown, dst)
        }
    }
}

/// Return the destination register of an instruction, if any.
fn instruction_dst(instruction: &Instruction) -> Option<u32> {
    match instruction {
        Instruction::StoreZero { dst }
        | Instruction::StoreOne { dst }
        | Instruction::StoreInt8 { dst, .. }
        | Instruction::StoreInt16 { dst, .. }
        | Instruction::StoreInt32 { dst, .. }
        | Instruction::StoreFloat { dst, .. }
        | Instruction::StoreDouble { dst, .. }
        | Instruction::StoreNan { dst }
        | Instruction::StorePositiveInfinity { dst }
        | Instruction::StoreNegativeInfinity { dst }
        | Instruction::StoreNull { dst }
        | Instruction::StoreTrue { dst }
        | Instruction::StoreFalse { dst }
        | Instruction::StoreUndefined { dst }
        | Instruction::This { dst }
        | Instruction::PopIntoRegister { dst }
        | Instruction::Add { dst, .. }
        | Instruction::Sub { dst, .. }
        | Instruction::Mul { dst, .. }
        | Instruction::Div { dst, .. }
        | Instruction::Mod { dst, .. }
        | Instruction::Pow { dst, .. }
        | Instruction::Inc { dst, .. }
        | Instruction::Dec { dst, .. }
        | Instruction::BitOr { dst, .. }
        | Instruction::BitAnd { dst, .. }
        | Instruction::BitXor { dst, .. }
        | Instruction::ShiftLeft { dst, .. }
        | Instruction::ShiftRight { dst, .. }
        | Instruction::UnsignedShiftRight { dst, .. }
        | Instruction::StrictEq { dst, .. }
        | Instruction::StrictNotEq { dst, .. }
        | Instruction::Eq { dst, .. }
        | Instruction::NotEq { dst, .. }
        | Instruction::GreaterThan { dst, .. }
        | Instruction::GreaterThanOrEq { dst, .. }
        | Instruction::LessThan { dst, .. }
        | Instruction::LessThanOrEq { dst, .. }
        | Instruction::InstanceOf { dst, .. }
        | Instruction::GetName { dst, .. }
        | Instruction::GetNameGlobal { dst, .. }
        | Instruction::GetNameOrUndefined { dst, .. }
        | Instruction::GetNameAndLocator { dst, .. }
        | Instruction::In { dst, .. }
        | Instruction::GetPropertyByName { dst, .. }
        | Instruction::GetPropertyByNameWithThis { dst, .. }
        | Instruction::GetLengthProperty { dst, .. }
        | Instruction::GetPropertyByValue { dst, .. }
        | Instruction::GetPropertyByValuePush { dst, .. }
        | Instruction::StoreLiteral { dst, .. }
        | Instruction::StoreEmptyObject { dst }
        | Instruction::StoreNewArray { dst }
        | Instruction::StoreRegexp { dst, .. }
        | Instruction::GetFunction { dst, .. }
        | Instruction::CreateUnmappedArgumentsObject { dst }
        | Instruction::RestParameterInit { dst }
        | Instruction::SetRegisterFromAccumulator { dst }
        | Instruction::Move { dst, .. } => Some(u32::from(*dst)),

        Instruction::GetArgument { dst, .. } => Some(u32::from(*dst)),
        Instruction::ToPropertyKey { dst, .. } => Some(u32::from(*dst)),
        Instruction::DeleteName { dst, .. } => Some(u32::from(*dst)),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::ir::{IrInst, build_ir};
    use crate::vm::opcode::{Bytecode, Opcode};

    fn emit(bytes: &mut Vec<u8>, opcode: Opcode, operands: &[u32]) {
        bytes.push(opcode as u8);
        for &op in operands {
            bytes.extend_from_slice(&op.to_le_bytes());
        }
    }

    #[test]
    fn type_int32_constants() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreInt32, &[1, 42]);
        emit(&mut bytes, Opcode::StoreZero, &[2]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let mut ir = build_ir(&bc);
        optimize(&mut ir);

        assert_eq!(ir.blocks[0].body[0].ty, ValueType::Int32);
        assert_eq!(ir.blocks[0].body[1].ty, ValueType::Int32);
    }

    #[test]
    fn type_bitwise_always_int32() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::BitOr, &[3, 1, 2]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let mut ir = build_ir(&bc);
        optimize(&mut ir);

        assert_eq!(ir.blocks[0].body[0].ty, ValueType::Int32);
    }

    #[test]
    fn type_add_int32_inputs() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreInt32, &[1, 10]);
        emit(&mut bytes, Opcode::StoreInt32, &[2, 20]);
        emit(&mut bytes, Opcode::Add, &[3, 1, 2]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let mut ir = build_ir(&bc);
        optimize(&mut ir);

        assert_eq!(ir.blocks[0].body[2].ty, ValueType::Int32);
    }

    #[test]
    fn type_add_unknown_input() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::GetArgument, &[0, 1]);
        emit(&mut bytes, Opcode::StoreInt32, &[2, 20]);
        emit(&mut bytes, Opcode::Add, &[3, 1, 2]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let mut ir = build_ir(&bc);
        optimize(&mut ir);

        assert_eq!(ir.blocks[0].body[2].ty, ValueType::Unknown);
    }

    #[test]
    fn type_comparison_boolean() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StrictEq, &[3, 1, 2]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let mut ir = build_ir(&bc);
        optimize(&mut ir);

        assert_eq!(ir.blocks[0].body[0].ty, ValueType::Boolean);
    }

    #[test]
    fn type_map_built_from_ir() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreInt32, &[1, 10]); // PC 0
        emit(&mut bytes, Opcode::StoreInt32, &[2, 20]); // PC 9
        emit(&mut bytes, Opcode::Add, &[3, 1, 2]); // PC 18

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let mut ir = build_ir(&bc);
        optimize(&mut ir);
        let tm = build_type_map(&ir);

        assert_eq!(tm.get(&0), Some(&ValueType::Int32));
        assert_eq!(tm.get(&9), Some(&ValueType::Int32));
        assert_eq!(tm.get(&18), Some(&ValueType::Int32));
    }
}
