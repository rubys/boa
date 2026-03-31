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
    copy_propagation(ir);
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

/// Copy propagation: eliminate redundant Move instructions.
///
/// For each pair of adjacent instructions in a block, if instruction A writes
/// to register X and instruction B is `Move { dst: Y, src: X }`, and register
/// X is not read by any subsequent instruction in the block (or the terminator),
/// rewrite A's destination to Y and remove the Move.
fn copy_propagation(ir: &mut IrFunction) {
    for block in &mut ir.blocks {
        let mut i = 0;
        while i + 1 < block.body.len() {
            // Check if instruction i+1 is a Move.
            let dominated = if let Instruction::Move {
                dst: move_dst,
                src: move_src,
            } = &block.body[i + 1].instruction
            {
                // Check if instruction i writes to move_src.
                if let Some(prev_dst) = instruction_dst(&block.body[i].instruction) {
                    if prev_dst == u32::from(*move_src) {
                        // Check that move_src is not read by any later instruction.
                        let src_reg = u32::from(*move_src);
                        // The register is safe to eliminate if:
                        // 1. It's not read by any later instruction in this block
                        // 2. It's not read by the terminator
                        // 3. It's overwritten before the block ends (so it can't
                        //    be live across block boundaries)
                        let not_read = !register_read_after(&block.body, i + 2, src_reg)
                            && !terminator_reads_reg(&block.terminator, src_reg);
                        let overwritten = register_overwritten_after(&block.body, i + 2, src_reg);
                        let is_dead = not_read && overwritten;
                        if is_dead {
                            Some(u32::from(*move_dst))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(new_dst) = dominated {
                // Rewrite instruction i's dst to the Move's dst.
                set_instruction_dst(&mut block.body[i].instruction, new_dst);
                // Remove the Move.
                block.body.remove(i + 1);
                // Don't advance — the new i+1 might also be a Move.
            } else {
                i += 1;
            }
        }
    }
}

/// Check if `reg` is overwritten (appears as dst) by any instruction in `body[start..]`.
fn register_overwritten_after(body: &[super::ir::IrInst], start: usize, reg: u32) -> bool {
    for inst in &body[start..] {
        if instruction_dst(&inst.instruction) == Some(reg) {
            return true;
        }
    }
    false
}

/// Check if `reg` is read by any instruction in `body[start..]`.
fn register_read_after(body: &[super::ir::IrInst], start: usize, reg: u32) -> bool {
    for inst in &body[start..] {
        if instruction_reads_reg(&inst.instruction, reg) {
            return true;
        }
    }
    false
}

/// Check if an instruction reads the given register.
fn instruction_reads_reg(instruction: &Instruction, reg: u32) -> bool {
    let reads = instruction_src_regs(instruction);
    reads.contains(&reg)
}

/// Check if a terminator reads the given register.
fn terminator_reads_reg(terminator: &super::ir::Terminator, reg: u32) -> bool {
    match terminator {
        super::ir::Terminator::Instruction(_, instruction) => {
            instruction_reads_reg(instruction, reg)
        }
        super::ir::Terminator::Fallthrough | super::ir::Terminator::End => false,
    }
}

/// Return all source registers read by an instruction.
fn instruction_src_regs(instruction: &Instruction) -> Vec<u32> {
    match instruction {
        // Binary ops: read lhs, rhs
        Instruction::Add { lhs, rhs, .. }
        | Instruction::Sub { lhs, rhs, .. }
        | Instruction::Mul { lhs, rhs, .. }
        | Instruction::Div { lhs, rhs, .. }
        | Instruction::Mod { lhs, rhs, .. }
        | Instruction::Pow { lhs, rhs, .. }
        | Instruction::BitOr { lhs, rhs, .. }
        | Instruction::BitAnd { lhs, rhs, .. }
        | Instruction::BitXor { lhs, rhs, .. }
        | Instruction::ShiftLeft { lhs, rhs, .. }
        | Instruction::ShiftRight { lhs, rhs, .. }
        | Instruction::UnsignedShiftRight { lhs, rhs, .. }
        | Instruction::StrictEq { lhs, rhs, .. }
        | Instruction::StrictNotEq { lhs, rhs, .. }
        | Instruction::Eq { lhs, rhs, .. }
        | Instruction::NotEq { lhs, rhs, .. }
        | Instruction::GreaterThan { lhs, rhs, .. }
        | Instruction::GreaterThanOrEq { lhs, rhs, .. }
        | Instruction::LessThan { lhs, rhs, .. }
        | Instruction::LessThanOrEq { lhs, rhs, .. }
        | Instruction::InstanceOf { lhs, rhs, .. }
        | Instruction::In { lhs, rhs, .. } => vec![u32::from(*lhs), u32::from(*rhs)],

        // Unary ops that read a register
        Instruction::Inc { src, .. } | Instruction::Dec { src, .. } => vec![u32::from(*src)],
        Instruction::Move { src, .. } => vec![u32::from(*src)],
        Instruction::SetAccumulator { src } => vec![u32::from(*src)],
        Instruction::PushFromRegister { src } => vec![u32::from(*src)],
        Instruction::Neg { value }
        | Instruction::Pos { value }
        | Instruction::BitNot { value }
        | Instruction::LogicalNot { value }
        | Instruction::TypeOf { value }
        | Instruction::IsObject { value } => vec![u32::from(*value)],
        Instruction::ValueNotNullOrUndefined { src } => vec![u32::from(*src)],
        Instruction::Throw { src } => vec![u32::from(*src)],
        Instruction::SetName { src, .. } | Instruction::SetNameByLocator { src } => {
            vec![u32::from(*src)]
        }
        Instruction::PutLexicalValue { src, .. } | Instruction::DefInitVar { src, .. } => {
            vec![u32::from(*src)]
        }
        Instruction::ToPropertyKey { src, .. } => vec![u32::from(*src)],
        Instruction::CheckReturn => vec![],

        // Property access: read object/value/key registers
        Instruction::GetPropertyByName { value, .. } => vec![u32::from(*value)],
        Instruction::GetPropertyByNameWithThis {
            receiver, value, ..
        } => vec![u32::from(*receiver), u32::from(*value)],
        Instruction::GetLengthProperty { value, .. } => vec![u32::from(*value)],
        Instruction::GetPropertyByValue {
            key,
            receiver,
            object,
            ..
        } => vec![u32::from(*key), u32::from(*receiver), u32::from(*object)],
        Instruction::GetPropertyByValuePush {
            key,
            receiver,
            object,
            ..
        } => vec![u32::from(*key), u32::from(*receiver), u32::from(*object)],
        Instruction::SetPropertyByValue {
            value,
            key,
            receiver,
            object,
        } => vec![
            u32::from(*value),
            u32::from(*key),
            u32::from(*receiver),
            u32::from(*object),
        ],
        Instruction::SetPropertyByName { value, object, .. } => {
            vec![u32::from(*value), u32::from(*object)]
        }
        Instruction::DefineOwnPropertyByName { object, value, .. } => {
            vec![u32::from(*object), u32::from(*value)]
        }
        Instruction::DefineOwnPropertyByValue {
            value, key, object, ..
        } => vec![u32::from(*value), u32::from(*key), u32::from(*object)],
        Instruction::DeletePropertyByName { object, .. } => vec![u32::from(*object)],
        Instruction::DeletePropertyByValue { object, key } => {
            vec![u32::from(*object), u32::from(*key)]
        }
        Instruction::GetPrototype { object } => vec![u32::from(*object)],
        Instruction::SetPrototype { object, prototype } => {
            vec![u32::from(*object), u32::from(*prototype)]
        }
        Instruction::PushValueToArray { value, array } => {
            vec![u32::from(*value), u32::from(*array)]
        }
        Instruction::PushElisionToArray { array } => vec![u32::from(*array)],

        // Jumps that read registers
        Instruction::JumpIfTrue { value, .. } | Instruction::JumpIfFalse { value, .. } => {
            vec![u32::from(*value)]
        }
        Instruction::JumpIfNullOrUndefined { value, .. }
        | Instruction::JumpIfNotUndefined { value, .. } => vec![u32::from(*value)],
        Instruction::JumpIfNotLessThan { lhs, rhs, .. }
        | Instruction::JumpIfNotLessThanOrEqual { lhs, rhs, .. }
        | Instruction::JumpIfNotGreaterThan { lhs, rhs, .. }
        | Instruction::JumpIfNotGreaterThanOrEqual { lhs, rhs, .. }
        | Instruction::JumpIfNotEqual { lhs, rhs, .. } => {
            vec![u32::from(*lhs), u32::from(*rhs)]
        }
        Instruction::Case {
            value, condition, ..
        } => vec![u32::from(*value), u32::from(*condition)],
        Instruction::LogicalAnd { value, .. }
        | Instruction::LogicalOr { value, .. }
        | Instruction::Coalesce { value, .. } => vec![u32::from(*value)],

        // Instructions that don't read registers (constants, jumps without values, etc.)
        _ => vec![],
    }
}

/// Set the destination register of an instruction.
fn set_instruction_dst(instruction: &mut Instruction, new_dst: u32) {
    use crate::vm::opcode::RegisterOperand;
    let new_reg = RegisterOperand::new(new_dst);
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
        | Instruction::Move { dst, .. }
        | Instruction::GetArgument { dst, .. }
        | Instruction::ToPropertyKey { dst, .. }
        | Instruction::DeleteName { dst, .. } => *dst = new_reg,
        _ => panic!("set_instruction_dst called on instruction without dst"),
    }
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

    // === Copy propagation tests ===

    #[test]
    fn copy_prop_simple_move_eliminated() {
        // Add r5, r1, r2; Move r3, r5; StoreZero r5  →  Add r3, r1, r2; StoreZero r5
        // The StoreZero overwrites r5, proving it's dead after the Move.
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::Add, &[5, 1, 2]);
        emit(&mut bytes, Opcode::Move, &[3, 5]);
        emit(&mut bytes, Opcode::StoreZero, &[5]); // overwrites r5

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let mut ir = build_ir(&bc);
        optimize(&mut ir);

        assert_eq!(ir.blocks[0].body.len(), 2, "Move should be eliminated");
        let dst = instruction_dst(&ir.blocks[0].body[0].instruction);
        assert_eq!(dst, Some(3), "Add dst should be patched to r3");
    }

    #[test]
    fn copy_prop_not_eliminated_when_src_still_live() {
        // Add r5, r1, r2; Move r3, r5; PushFromRegister r5
        // r5 is still read after the Move — don't eliminate.
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::Add, &[5, 1, 2]);
        emit(&mut bytes, Opcode::Move, &[3, 5]);
        emit(&mut bytes, Opcode::PushFromRegister, &[5]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let mut ir = build_ir(&bc);
        optimize(&mut ir);

        assert_eq!(
            ir.blocks[0].body.len(),
            3,
            "Move should NOT be eliminated when src is still live"
        );
    }

    #[test]
    fn copy_prop_two_consecutive() {
        // Inc r5, r4; Move r3, r5; BitOr r6, r3, r7; Move r1, r6; StoreZero r5; StoreZero r6
        // The StoreZeros prove r5 and r6 are dead.
        // → Inc r3, r4; BitOr r1, r3, r7; StoreZero r5; StoreZero r6
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::Inc, &[5, 4]);
        emit(&mut bytes, Opcode::Move, &[3, 5]);
        emit(&mut bytes, Opcode::BitOr, &[6, 3, 7]);
        emit(&mut bytes, Opcode::Move, &[1, 6]);
        emit(&mut bytes, Opcode::StoreZero, &[5]); // overwrites r5
        emit(&mut bytes, Opcode::StoreZero, &[6]); // overwrites r6

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let mut ir = build_ir(&bc);
        optimize(&mut ir);

        assert_eq!(
            ir.blocks[0].body.len(),
            4,
            "both Moves should be eliminated"
        );
        assert_eq!(instruction_dst(&ir.blocks[0].body[0].instruction), Some(3));
        assert_eq!(instruction_dst(&ir.blocks[0].body[1].instruction), Some(1));
    }

    #[test]
    fn copy_prop_real_bytecode() {
        // Compile real JS and verify copy propagation runs without panic.
        let mut ctx = crate::Context::default();
        ctx.eval(crate::Source::from_bytes(
            "function f(n) { var s = 0; for (var i = 0; i < n; i++) s = s + i; return s; }",
        ))
        .unwrap();

        let value = ctx
            .global_object()
            .get(crate::js_string!("f"), &mut ctx)
            .unwrap();
        let obj = value.as_object().unwrap();
        let func = obj
            .downcast_ref::<crate::builtins::function::OrdinaryFunction>()
            .unwrap();
        let code = func.codeblock();

        let mut ir = build_ir(&code.bytecode);
        optimize(&mut ir);

        // Should have fewer instructions than unoptimized.
        let total_insts: usize = ir.blocks.iter().map(|b| b.body.len()).sum();
        assert!(total_insts > 0, "optimized IR should have instructions");
    }
}
