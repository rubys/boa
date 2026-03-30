//! Bytecode optimization passes for the JIT compiler.
//!
//! These passes run on Boa bytecode before Cranelift lowering:
//!
//! 1. **Copy propagation** — eliminates redundant `Move` instructions by
//!    rewriting the destination of the preceding instruction.
//! 2. **Type propagation** — produces a map from bytecode PC to the known
//!    value type, allowing the Cranelift lowering to skip NaN-boxing guards.

use std::collections::HashMap;

use crate::vm::opcode::{Instruction, InstructionIterator, Opcode};

/// Known value type produced by an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueType {
    /// No type information — emit full NaN-boxing guards.
    Unknown,
    /// Known 32-bit integer.
    Int32,
    /// Known 64-bit float.
    Float64,
    /// Known boolean.
    Boolean,
}

/// Result of the optimization passes.
pub(super) struct OptimizedBytecode {
    /// The optimized bytecode bytes.
    pub bytes: Vec<u8>,
    /// Map from PC in the optimized bytecode to known value type.
    pub type_map: HashMap<u32, ValueType>,
}

/// Run all optimization passes on the given bytecode.
pub(super) fn optimize(bytecode: &crate::vm::opcode::Bytecode) -> OptimizedBytecode {
    // Copy propagation is disabled pending register liveness annotations
    // from the bytecompiler. Without liveness data, the pass can incorrectly
    // eliminate Moves whose source register is still read later.
    let bytes = bytecode.bytes.to_vec();

    // Build a temporary Bytecode for type propagation iteration.
    let optimized_bytecode = crate::vm::opcode::Bytecode {
        bytes: bytes.clone().into_boxed_slice(),
    };
    let type_map = type_propagation(&optimized_bytecode);

    OptimizedBytecode { bytes, type_map }
}

/// Pass 1: Copy propagation.
///
/// Pattern: if instruction I writes to register X and the very next
/// instruction is `Move { dst: Y, src: X }`, rewrite I's destination
/// to Y and eliminate the Move.
///
/// Returns new bytecode bytes with Moves removed and addresses remapped.
fn copy_propagation(bytecode: &crate::vm::opcode::Bytecode) -> Vec<u8> {
    let raw = &bytecode.bytes;

    // Phase 1: Identify which Moves to eliminate and what dst to patch.
    // Collect: (move_pc, preceding_inst_pc, preceding_inst_dst_offset, new_dst_value)
    struct Elimination {
        move_pc: usize,
        move_end: usize,
        prev_dst_byte_offset: usize, // byte offset of the dst field in the previous instruction
        new_dst: u32,                // the Move's dst register
    }

    let mut eliminations: Vec<Elimination> = Vec::new();
    let mut prev_info: Option<(usize, usize)> = None; // (inst_pc, dst_byte_offset)

    let iter = InstructionIterator::new(bytecode);
    for (pc, opcode, instruction) in iter {
        // Check if this is a Move that can be eliminated.
        if let Instruction::Move { dst, src } = &instruction {
            if let Some((prev_pc, prev_dst_offset)) = prev_info {
                if u32::from(*src) == read_u32_at(raw, prev_dst_offset) {
                    // The previous instruction wrote to the same register the Move reads.
                    // Calculate the end of this Move instruction: opcode(1) + dst(4) + src(4) = 9
                    eliminations.push(Elimination {
                        move_pc: pc,
                        move_end: pc + 9,
                        prev_dst_byte_offset: prev_dst_offset,
                        new_dst: u32::from(*dst),
                    });
                    prev_info = None;
                    continue;
                }
            }
        }

        // Track the dst field offset of the current instruction.
        prev_info = dst_byte_offset(pc, &instruction).map(|off| (pc, off));
    }

    if eliminations.is_empty() {
        // No optimization needed — return a copy of the original bytes.
        return raw.to_vec();
    }

    // Phase 2: Build new bytecode, skipping eliminated Moves and patching dst fields.
    // Also build an offset map: old_pc -> new_pc.
    let mut new_bytes: Vec<u8> = Vec::with_capacity(raw.len());
    let mut offset_map: HashMap<u32, u32> = HashMap::new();
    let mut elim_idx = 0;
    let mut skip_until: usize = 0;

    // We need to iterate byte-by-byte through instructions. Use InstructionIterator
    // to get instruction boundaries.
    let iter = InstructionIterator::new(bytecode);
    let mut inst_ranges: Vec<(usize, usize)> = Vec::new(); // (start_pc, next_pc)
    let mut last_end = 0;
    for (pc, _opcode, _instruction) in iter {
        if pc > 0 && !inst_ranges.is_empty() {
            inst_ranges.last_mut().unwrap().1 = pc;
        }
        inst_ranges.push((pc, raw.len()));
        last_end = pc;
    }
    // Fix the last instruction's end.
    if let Some(last) = inst_ranges.last_mut() {
        last.1 = raw.len();
    }

    // Build a set of Move PCs for quick lookup.
    let move_pcs: HashMap<usize, usize> = eliminations
        .iter()
        .map(|e| (e.move_pc, e.prev_dst_byte_offset))
        .collect();

    // Build a set of instructions whose dst needs patching.
    // Value is (offset of dst field relative to instruction start, new dst value).
    let dst_patches: HashMap<usize, (usize, u32)> = eliminations
        .iter()
        .filter_map(|e| {
            inst_ranges
                .iter()
                .find(|(start, end)| {
                    e.prev_dst_byte_offset >= *start && e.prev_dst_byte_offset < *end
                })
                .map(|(start, _)| (*start, (e.prev_dst_byte_offset - start, e.new_dst)))
        })
        .collect();

    for &(start, end) in &inst_ranges {
        let old_pc = start as u32;
        let new_pc = new_bytes.len() as u32;
        offset_map.insert(old_pc, new_pc);

        if move_pcs.contains_key(&start) {
            // Skip this Move instruction entirely.
            continue;
        }

        if let Some(&(dst_rel_offset, new_dst)) = dst_patches.get(&start) {
            // Copy instruction but patch the dst field at its correct offset.
            let inst_bytes = &raw[start..end];
            new_bytes.extend_from_slice(inst_bytes);
            let dst_offset_in_new = new_pc as usize + dst_rel_offset;
            let dst_le = new_dst.to_le_bytes();
            new_bytes[dst_offset_in_new..dst_offset_in_new + 4].copy_from_slice(&dst_le);
        } else {
            // Copy instruction verbatim.
            new_bytes.extend_from_slice(&raw[start..end]);
        }
    }

    // Phase 3: Fix all Address operands using the offset map.
    // Re-iterate the NEW bytecode and patch any Address fields.
    let temp_bytecode = crate::vm::opcode::Bytecode {
        bytes: new_bytes.clone().into_boxed_slice(),
    };
    let iter = InstructionIterator::new(&temp_bytecode);
    for (pc, _opcode, instruction) in iter {
        if let Some(addr_offsets) = address_byte_offsets(pc, &instruction) {
            for addr_off in addr_offsets {
                let old_addr = read_u32_at(&new_bytes, addr_off);
                if let Some(&new_addr) = offset_map.get(&old_addr) {
                    let le = new_addr.to_le_bytes();
                    new_bytes[addr_off..addr_off + 4].copy_from_slice(&le);
                }
                // If not in the map, the address points to an unchanged location
                // (shouldn't happen in valid bytecode, but be safe).
            }
        }
    }

    new_bytes
}

/// Pass 2: Type propagation.
///
/// Produces a map from bytecode PC to the known [`ValueType`] the instruction
/// at that PC produces. The Cranelift lowering can use this to skip tag checks.
fn type_propagation(bytecode: &crate::vm::opcode::Bytecode) -> HashMap<u32, ValueType> {
    let mut type_map: HashMap<u32, ValueType> = HashMap::new();
    let mut reg_types: HashMap<u32, ValueType> = HashMap::new();

    let iter = InstructionIterator::new(bytecode);
    for (pc, _opcode, instruction) in iter {
        let (produced_type, dst_reg) = infer_type(&instruction, &reg_types);

        if produced_type != ValueType::Unknown {
            type_map.insert(pc as u32, produced_type);
        }

        // Update register type tracking.
        if let Some(dst) = dst_reg {
            if produced_type != ValueType::Unknown {
                reg_types.insert(dst, produced_type);
            } else {
                reg_types.remove(&dst);
            }
        }
    }

    type_map
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

// ============================================================================
// Bytecode field offset helpers
// ============================================================================

/// Read a little-endian u32 from a byte slice at the given offset.
fn read_u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Return the byte offset of the `dst` register field in the raw bytecode
/// for instructions that have a destination register as their first operand.
///
/// The dst field is always the first operand after the opcode byte.
fn dst_byte_offset(pc: usize, instruction: &Instruction) -> Option<usize> {
    let offset = pc + 1; // skip opcode byte
    match instruction {
        Instruction::Move { .. }
        | Instruction::StoreZero { .. }
        | Instruction::StoreOne { .. }
        | Instruction::StoreInt8 { .. }
        | Instruction::StoreInt16 { .. }
        | Instruction::StoreInt32 { .. }
        | Instruction::StoreFloat { .. }
        | Instruction::StoreDouble { .. }
        | Instruction::StoreNan { .. }
        | Instruction::StorePositiveInfinity { .. }
        | Instruction::StoreNegativeInfinity { .. }
        | Instruction::StoreNull { .. }
        | Instruction::StoreTrue { .. }
        | Instruction::StoreFalse { .. }
        | Instruction::StoreUndefined { .. }
        | Instruction::This { .. }
        | Instruction::PopIntoRegister { .. }
        | Instruction::Add { .. }
        | Instruction::Sub { .. }
        | Instruction::Mul { .. }
        | Instruction::Div { .. }
        | Instruction::Mod { .. }
        | Instruction::Pow { .. }
        | Instruction::Inc { .. }
        | Instruction::Dec { .. }
        | Instruction::BitOr { .. }
        | Instruction::BitAnd { .. }
        | Instruction::BitXor { .. }
        | Instruction::ShiftLeft { .. }
        | Instruction::ShiftRight { .. }
        | Instruction::UnsignedShiftRight { .. }
        | Instruction::StrictEq { .. }
        | Instruction::StrictNotEq { .. }
        | Instruction::Eq { .. }
        | Instruction::NotEq { .. }
        | Instruction::GreaterThan { .. }
        | Instruction::GreaterThanOrEq { .. }
        | Instruction::LessThan { .. }
        | Instruction::LessThanOrEq { .. }
        | Instruction::InstanceOf { .. }
        | Instruction::GetName { .. }
        | Instruction::GetNameGlobal { .. }
        | Instruction::GetNameOrUndefined { .. }
        | Instruction::GetNameAndLocator { .. }
        | Instruction::DeleteName { .. }
        | Instruction::In { .. }
        | Instruction::GetPropertyByName { .. }
        | Instruction::GetPropertyByNameWithThis { .. }
        | Instruction::GetLengthProperty { .. }
        | Instruction::GetPropertyByValue { .. }
        | Instruction::GetPropertyByValuePush { .. }
        | Instruction::StoreLiteral { .. }
        | Instruction::StoreEmptyObject { .. }
        | Instruction::StoreNewArray { .. }
        | Instruction::StoreRegexp { .. }
        | Instruction::GetFunction { .. }
        | Instruction::CreateUnmappedArgumentsObject { .. }
        | Instruction::RestParameterInit { .. }
        | Instruction::SetRegisterFromAccumulator { .. } => Some(offset),

        // GetArgument has (index, dst) — dst is the second operand.
        Instruction::GetArgument { .. } => Some(offset + 4),

        // ToPropertyKey has (src, dst) — dst is the second operand.
        Instruction::ToPropertyKey { .. } => Some(offset + 4),

        // These don't have a dst register as first operand.
        _ => None,
    }
}

/// Return the destination register of an instruction, if any.
fn instruction_dst(instruction: &Instruction) -> Option<u32> {
    match instruction {
        Instruction::Move { dst, .. }
        | Instruction::StoreZero { dst }
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
        | Instruction::GetArgument { dst, .. } => Some(u32::from(*dst)),

        Instruction::ToPropertyKey { dst, .. } => Some(u32::from(*dst)),

        _ => None,
    }
}

/// Return the byte offsets of Address fields in the raw bytecode for an
/// instruction. These need to be remapped when bytecode is rewritten.
fn address_byte_offsets(pc: usize, instruction: &Instruction) -> Option<Vec<usize>> {
    // Address is always a u32 at a fixed position in the instruction.
    // For jump instructions, the address is the first operand after the opcode.
    let base = pc + 1; // skip opcode byte
    match instruction {
        Instruction::Jump { .. } => Some(vec![base]),

        // These have Address as first operand.
        Instruction::JumpIfTrue { .. }
        | Instruction::JumpIfFalse { .. }
        | Instruction::JumpIfNullOrUndefined { .. }
        | Instruction::JumpIfNotUndefined { .. }
        | Instruction::JumpIfNotLessThan { .. }
        | Instruction::JumpIfNotLessThanOrEqual { .. }
        | Instruction::JumpIfNotGreaterThan { .. }
        | Instruction::JumpIfNotGreaterThanOrEqual { .. }
        | Instruction::JumpIfNotEqual { .. }
        | Instruction::LogicalAnd { .. }
        | Instruction::LogicalOr { .. }
        | Instruction::Coalesce { .. }
        | Instruction::Case { .. } => Some(vec![base]),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::opcode::{Bytecode, Instruction, Opcode};

    /// Helper: encode an opcode with u32 operands into a byte buffer.
    fn emit(bytes: &mut Vec<u8>, opcode: Opcode, operands: &[u32]) {
        bytes.push(opcode as u8);
        for &op in operands {
            bytes.extend_from_slice(&op.to_le_bytes());
        }
    }

    /// Helper: emit a Jump instruction (opcode + address).
    fn emit_jump(bytes: &mut Vec<u8>, opcode: Opcode, address: u32) {
        bytes.push(opcode as u8);
        bytes.extend_from_slice(&address.to_le_bytes());
    }

    /// Helper: emit a conditional jump (opcode + address + register(s)).
    fn emit_cond_jump(bytes: &mut Vec<u8>, opcode: Opcode, address: u32, regs: &[u32]) {
        bytes.push(opcode as u8);
        bytes.extend_from_slice(&address.to_le_bytes());
        for &r in regs {
            bytes.extend_from_slice(&r.to_le_bytes());
        }
    }

    /// Decode all instructions from raw bytes.
    fn decode_instructions(bytes: &[u8]) -> Vec<(usize, Opcode, Instruction)> {
        let bc = Bytecode {
            bytes: bytes.to_vec().into_boxed_slice(),
        };
        InstructionIterator::new(&bc).collect()
    }

    #[test]
    fn no_moves_unchanged() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreInt32, &[1, 42]);
        emit(&mut bytes, Opcode::StoreInt32, &[2, 7]);

        let bc = Bytecode {
            bytes: bytes.clone().into_boxed_slice(),
        };
        let result = copy_propagation(&bc);
        assert_eq!(result, bytes, "bytecode without Moves should be unchanged");
    }

    #[test]
    fn simple_move_eliminated() {
        // Add r5, r1, r2; Move r3, r5  →  Add r3, r1, r2
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::Add, &[5, 1, 2]);
        emit(&mut bytes, Opcode::Move, &[3, 5]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let result = copy_propagation(&bc);

        let insts = decode_instructions(&result);
        assert_eq!(insts.len(), 1, "Move should be eliminated");

        if let Instruction::Add { dst, lhs, rhs } = &insts[0].2 {
            assert_eq!(u32::from(*dst), 3, "dst should be patched to r3");
            assert_eq!(u32::from(*lhs), 1);
            assert_eq!(u32::from(*rhs), 2);
        } else {
            panic!("expected Add instruction");
        }
    }

    #[test]
    fn move_not_eliminated_when_src_differs() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::Add, &[5, 1, 2]);
        emit(&mut bytes, Opcode::Move, &[3, 4]); // src=r4, not r5

        let bc = Bytecode {
            bytes: bytes.clone().into_boxed_slice(),
        };
        let result = copy_propagation(&bc);
        let insts = decode_instructions(&result);
        assert_eq!(insts.len(), 2, "Move should NOT be eliminated");
    }

    #[test]
    fn two_consecutive_eliminations() {
        // Inc r5, r4; Move r3, r5; BitOr r6, r3, r7; Move r1, r6
        // → Inc r3, r4; BitOr r1, r3, r7
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::Inc, &[5, 4]);
        emit(&mut bytes, Opcode::Move, &[3, 5]);
        emit(&mut bytes, Opcode::BitOr, &[6, 3, 7]);
        emit(&mut bytes, Opcode::Move, &[1, 6]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let result = copy_propagation(&bc);
        let insts = decode_instructions(&result);
        assert_eq!(insts.len(), 2, "both Moves should be eliminated");

        if let Instruction::Inc { dst, src } = &insts[0].2 {
            assert_eq!(u32::from(*dst), 3);
            assert_eq!(u32::from(*src), 4);
        } else {
            panic!("expected Inc");
        }

        if let Instruction::BitOr { dst, lhs, rhs } = &insts[1].2 {
            assert_eq!(u32::from(*dst), 1);
            assert_eq!(u32::from(*lhs), 3);
            assert_eq!(u32::from(*rhs), 7);
        } else {
            panic!("expected BitOr");
        }
    }

    #[test]
    fn jump_address_remapped() {
        // Add r5, r1, r2; Move r3, r5; StoreInt32 r2, 7; Jump <StoreInt32>
        // After: Add r3, r1, r2; StoreInt32 r2, 7; Jump <new addr>
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::Add, &[5, 1, 2]); // 13 bytes (offset 0)
        emit(&mut bytes, Opcode::Move, &[3, 5]); // 9 bytes (offset 13)
        let store_pc = bytes.len(); // offset 22
        emit(&mut bytes, Opcode::StoreInt32, &[2, 7]); // 9 bytes
        emit_jump(&mut bytes, Opcode::Jump, store_pc as u32);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let result = copy_propagation(&bc);
        let insts = decode_instructions(&result);
        assert_eq!(insts.len(), 3, "Move eliminated");
        assert_eq!(insts[1].0, 13, "StoreInt32 now at offset 13");

        if let Instruction::Jump { address } = &insts[2].2 {
            assert_eq!(address.as_u32(), 13, "Jump target remapped");
        } else {
            panic!("expected Jump");
        }
    }

    #[test]
    fn conditional_jump_address_remapped() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::Add, &[5, 1, 2]); // 13 bytes
        emit(&mut bytes, Opcode::Move, &[3, 5]); // 9 bytes
        let store_pc = bytes.len(); // 22
        emit(&mut bytes, Opcode::StoreInt32, &[2, 0]);
        emit_cond_jump(&mut bytes, Opcode::JumpIfTrue, store_pc as u32, &[3]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let result = copy_propagation(&bc);
        let insts = decode_instructions(&result);
        assert_eq!(insts.len(), 3);

        if let Instruction::JumpIfTrue { address, .. } = &insts[2].2 {
            assert_eq!(address.as_u32(), 13);
        } else {
            panic!("expected JumpIfTrue");
        }
    }

    #[test]
    fn store_empty_object_not_corrupted() {
        // StoreEmptyObject r3; DefineOwnPropertyByName r3, r5, <name_idx>; Move should not be here
        // but test that StoreEmptyObject followed by a non-Move doesn't cause issues.
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreEmptyObject, &[3]); // 5 bytes: opcode + dst(4)
        // DefineOwnPropertyByName { object: r3, value: r5, name_index: 0 }
        emit(&mut bytes, Opcode::DefineOwnPropertyByName, &[3, 5, 0]);

        let bc = Bytecode {
            bytes: bytes.clone().into_boxed_slice(),
        };
        let result = copy_propagation(&bc);
        // No Move present, so bytecode should be unchanged.
        assert_eq!(result, bytes);
    }

    #[test]
    fn object_literal_pattern() {
        // Simulate: StoreEmptyObject r5; Move r3, r5; DefineOwnPropertyByName r3, ...
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreEmptyObject, &[5]); // 5 bytes
        emit(&mut bytes, Opcode::Move, &[3, 5]); // 9 bytes
        emit(&mut bytes, Opcode::DefineOwnPropertyByName, &[3, 7, 0]); // object=r3, value=r7, name=0

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let result = copy_propagation(&bc);
        let insts = decode_instructions(&result);
        assert_eq!(insts.len(), 2, "Move should be eliminated");

        // StoreEmptyObject should now write to r3 directly.
        if let Instruction::StoreEmptyObject { dst } = &insts[0].2 {
            assert_eq!(u32::from(*dst), 3);
        } else {
            panic!("expected StoreEmptyObject");
        }

        // DefineOwnPropertyByName should be unchanged.
        if let Instruction::DefineOwnPropertyByName {
            object,
            value,
            name_index,
        } = &insts[1].2
        {
            assert_eq!(u32::from(*object), 3);
            assert_eq!(u32::from(*value), 7);
        } else {
            panic!("expected DefineOwnPropertyByName");
        }
    }

    // === Type propagation tests ===

    #[test]
    fn type_int32_constants() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreInt32, &[1, 42]);
        emit(&mut bytes, Opcode::StoreZero, &[2]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let tm = type_propagation(&bc);
        assert_eq!(tm.get(&0), Some(&ValueType::Int32));
        assert_eq!(tm.get(&9), Some(&ValueType::Int32));
    }

    #[test]
    fn type_bitwise_always_int32() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::BitOr, &[3, 1, 2]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let tm = type_propagation(&bc);
        assert_eq!(tm.get(&0), Some(&ValueType::Int32));
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
        let tm = type_propagation(&bc);
        assert_eq!(tm.get(&18), Some(&ValueType::Int32));
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
        let tm = type_propagation(&bc);
        assert_eq!(tm.get(&18), None, "Add with unknown input is Unknown");
    }

    #[test]
    fn get_argument_move_eliminated() {
        // GetArgument { index: 0, dst: 3 }; Move { dst: 1, src: 3 }
        // → GetArgument { index: 0, dst: 1 }
        // Regression: dst is the SECOND operand in GetArgument, not the first.
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::GetArgument, &[0, 3]); // index=0, dst=3
        emit(&mut bytes, Opcode::Move, &[1, 3]); // dst=1, src=3

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let result = copy_propagation(&bc);
        let insts = decode_instructions(&result);
        assert_eq!(insts.len(), 1, "Move should be eliminated");

        if let Instruction::GetArgument { index, dst } = &insts[0].2 {
            assert_eq!(u32::from(*index), 0, "index must be unchanged");
            assert_eq!(u32::from(*dst), 1, "dst should be patched to r1");
        } else {
            panic!("expected GetArgument");
        }
    }

    #[test]
    fn type_comparison_boolean() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StrictEq, &[3, 1, 2]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let tm = type_propagation(&bc);
        assert_eq!(tm.get(&0), Some(&ValueType::Boolean));
    }
}
