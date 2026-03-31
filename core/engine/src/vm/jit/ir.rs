//! Structured intermediate representation for JIT compilation.
//!
//! The IR deserializes Boa's flat bytecode into basic blocks, providing the
//! structured access needed for optimization passes and Cranelift lowering.
//!
//! The IR reuses Boa's [`Instruction`] enum directly — no parallel opcode
//! type. This means the lowering methods can match on the same `Instruction`
//! variants whether called from the bytecode path or the IR path.

use crate::vm::opcode::{Instruction, InstructionIterator};
use std::collections::HashSet;

/// Block identifier, indexing into [`IrFunction::blocks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct BlockId(pub usize);

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

/// An instruction in the IR: bytecode PC, the instruction, and its known output type.
pub(super) struct IrInst {
    pub pc: u32,
    pub instruction: Instruction,
    pub ty: ValueType,
}

/// A basic block: a sequence of non-branching instructions followed by a terminator.
pub(super) struct BasicBlock {
    /// The bytecode PC where this block starts.
    pub start_pc: u32,
    /// Non-terminator instructions with their PCs and type annotations.
    pub body: Vec<IrInst>,
    /// How this block ends.
    pub terminator: Terminator,
}

/// How a basic block transfers control.
pub(super) enum Terminator {
    /// Falls through to the next block (no explicit jump in bytecode).
    Fallthrough,
    /// The block's last instruction is a terminator (jump, branch, return).
    /// Stored with its PC so the lowering can access it.
    Instruction(u32, Instruction),
    /// End of bytecode (implicit return).
    End,
}

/// The complete IR for one function.
pub(super) struct IrFunction {
    pub blocks: Vec<BasicBlock>,
}

/// Build the IR from bytecode.
///
/// Splits the bytecode into basic blocks at jump targets. Instructions that
/// are terminators (jumps, branches, return) end their block.
pub(super) fn build_ir(bytecode: &crate::vm::opcode::Bytecode) -> IrFunction {
    // Phase 1: Find all jump targets.
    let mut jump_targets: HashSet<u32> = HashSet::new();
    let iter = InstructionIterator::new(bytecode);
    for (_pc, _opcode, instruction) in iter {
        if let Some(target) = jump_target(&instruction) {
            jump_targets.insert(target);
        }
    }

    // Phase 2: Build blocks.
    let mut blocks: Vec<BasicBlock> = Vec::new();
    let mut body: Vec<IrInst> = Vec::new();
    let mut block_start_pc: u32 = 0;

    let iter = InstructionIterator::new(bytecode);
    for (pc, _opcode, instruction) in iter {
        let pc = pc as u32;

        // Start a new block if this PC is a jump target or after a terminator.
        if block_start_pc == u32::MAX {
            // Previous instruction was a terminator. Start fresh.
            block_start_pc = pc;
        } else if jump_targets.contains(&pc) && (pc != block_start_pc || !body.is_empty()) {
            // This PC is a jump target and we have a pending block — close it.
            blocks.push(BasicBlock {
                start_pc: block_start_pc,
                body: std::mem::take(&mut body),
                terminator: Terminator::Fallthrough,
            });
            block_start_pc = pc;
        }

        // Check if this instruction is a terminator.
        if is_terminator(&instruction) {
            blocks.push(BasicBlock {
                start_pc: block_start_pc,
                body: std::mem::take(&mut body),
                terminator: Terminator::Instruction(pc, instruction),
            });
            block_start_pc = u32::MAX;
            continue;
        }

        body.push(IrInst {
            pc,
            instruction,
            ty: ValueType::Unknown,
        });
    }

    // Close the last block if there are remaining instructions.
    if !body.is_empty() {
        blocks.push(BasicBlock {
            start_pc: block_start_pc,
            body,
            terminator: Terminator::End,
        });
    }

    IrFunction { blocks }
}

/// Extract the jump target PC from an instruction, if it has one.
fn jump_target(instruction: &Instruction) -> Option<u32> {
    match instruction {
        Instruction::Jump { address }
        | Instruction::JumpIfTrue { address, .. }
        | Instruction::JumpIfFalse { address, .. }
        | Instruction::JumpIfNotLessThan { address, .. }
        | Instruction::JumpIfNotLessThanOrEqual { address, .. }
        | Instruction::JumpIfNotGreaterThan { address, .. }
        | Instruction::JumpIfNotGreaterThanOrEqual { address, .. }
        | Instruction::JumpIfNotEqual { address, .. }
        | Instruction::JumpIfNullOrUndefined { address, .. }
        | Instruction::JumpIfNotUndefined { address, .. }
        | Instruction::Case { address, .. }
        | Instruction::LogicalAnd { address, .. }
        | Instruction::LogicalOr { address, .. }
        | Instruction::Coalesce { address, .. } => Some(address.as_u32()),
        _ => None,
    }
}

/// Check if an instruction is a block terminator.
fn is_terminator(instruction: &Instruction) -> bool {
    matches!(
        instruction,
        Instruction::Jump { .. }
            | Instruction::JumpIfTrue { .. }
            | Instruction::JumpIfFalse { .. }
            | Instruction::JumpIfNotLessThan { .. }
            | Instruction::JumpIfNotLessThanOrEqual { .. }
            | Instruction::JumpIfNotGreaterThan { .. }
            | Instruction::JumpIfNotGreaterThanOrEqual { .. }
            | Instruction::JumpIfNotEqual { .. }
            | Instruction::JumpIfNullOrUndefined { .. }
            | Instruction::JumpIfNotUndefined { .. }
            | Instruction::Case { .. }
            | Instruction::LogicalAnd { .. }
            | Instruction::LogicalOr { .. }
            | Instruction::Coalesce { .. }
            | Instruction::Return
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::opcode::{Bytecode, Opcode};

    fn emit(bytes: &mut Vec<u8>, opcode: Opcode, operands: &[u32]) {
        bytes.push(opcode as u8);
        for &op in operands {
            bytes.extend_from_slice(&op.to_le_bytes());
        }
    }

    fn emit_jump(bytes: &mut Vec<u8>, opcode: Opcode, address: u32) {
        bytes.push(opcode as u8);
        bytes.extend_from_slice(&address.to_le_bytes());
    }

    fn emit_cond_jump(bytes: &mut Vec<u8>, opcode: Opcode, address: u32, regs: &[u32]) {
        bytes.push(opcode as u8);
        bytes.extend_from_slice(&address.to_le_bytes());
        for &r in regs {
            bytes.extend_from_slice(&r.to_le_bytes());
        }
    }

    #[test]
    fn straight_line_single_block() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreInt32, &[1, 42]);
        emit(&mut bytes, Opcode::StoreInt32, &[2, 7]);
        emit(&mut bytes, Opcode::Add, &[3, 1, 2]);

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let ir = build_ir(&bc);
        assert_eq!(ir.blocks.len(), 1);
        assert_eq!(ir.blocks[0].body.len(), 3);
        assert!(matches!(ir.blocks[0].terminator, Terminator::End));
    }

    #[test]
    fn jump_splits_blocks() {
        // StoreInt32 r1, 0; Jump <after>; StoreInt32 r2, 1
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreInt32, &[1, 0]); // 9 bytes, PC 0
        let target_pc = 9 + 5; // Jump is 5 bytes, target is PC 14
        emit_jump(&mut bytes, Opcode::Jump, target_pc as u32); // 5 bytes, PC 9
        emit(&mut bytes, Opcode::StoreInt32, &[2, 1]); // 9 bytes, PC 14

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let ir = build_ir(&bc);

        // Block 0: StoreInt32 + Jump terminator
        // Block 1: empty (unreachable fallthrough after Jump, before target)
        // Block 2: StoreInt32 (at jump target PC 14)
        // Or blocks may be: Block 0 (body + Jump), Block 1 (target).
        // The exact count depends on whether PC 14 == the instruction after Jump.
        // In this case: Jump at PC 9, size 5, so next PC is 14 == target. So 2 blocks.
        // But the block after the terminator also starts, making 3 if they don't merge.
        assert!(ir.blocks.len() >= 2, "got {} blocks", ir.blocks.len());
        // First block has StoreInt32 in body and Jump as terminator.
        assert_eq!(ir.blocks[0].body.len(), 1);
        // Last block has the target StoreInt32.
        let last = ir.blocks.last().unwrap();
        assert_eq!(last.start_pc, 14);
    }

    #[test]
    fn backward_jump_creates_block_at_target() {
        // loop: StoreInt32 r1, 0; Jump <loop>
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreInt32, &[1, 0]); // 9 bytes, PC 0
        emit_jump(&mut bytes, Opcode::Jump, 0); // Jump back to PC 0

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let ir = build_ir(&bc);

        // Single block: StoreInt32 body + Jump terminator
        // (PC 0 is a jump target but it's also the block start, so no split)
        assert_eq!(ir.blocks.len(), 1);
        assert_eq!(ir.blocks[0].body.len(), 1);
        assert_eq!(ir.blocks[0].start_pc, 0);
    }

    #[test]
    fn conditional_branch_splits() {
        let mut bytes = Vec::new();
        emit(&mut bytes, Opcode::StoreInt32, &[1, 0]); // 9 bytes, PC 0
        let target_pc = 9 + 9 + 9; // after the JumpIfTrue and one more instruction
        emit_cond_jump(&mut bytes, Opcode::JumpIfTrue, target_pc as u32, &[1]); // 9 bytes, PC 9
        emit(&mut bytes, Opcode::StoreInt32, &[2, 1]); // 9 bytes, PC 18
        emit(&mut bytes, Opcode::StoreInt32, &[3, 2]); // 9 bytes, PC 27

        let bc = Bytecode {
            bytes: bytes.into_boxed_slice(),
        };
        let ir = build_ir(&bc);

        // Block 0: StoreInt32 + JumpIfTrue terminator
        // Block 1: StoreInt32 (fallthrough, PC 18)
        // Block 2: StoreInt32 (jump target, PC 27)
        assert!(ir.blocks.len() >= 2);
        assert!(matches!(
            ir.blocks[0].terminator,
            Terminator::Instruction(9, Instruction::JumpIfTrue { .. })
        ));
    }

    #[test]
    fn real_bytecode_builds_ir() {
        // Compile actual JS and build IR from it.
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

        let ir = build_ir(&code.bytecode);

        // A for-loop function should have multiple blocks.
        assert!(
            ir.blocks.len() >= 3,
            "for-loop function should have at least 3 blocks, got {}",
            ir.blocks.len()
        );

        // Every non-empty block should have a valid start_pc.
        let bytecode_len = code.bytecode.bytes.len() as u32;
        for block in &ir.blocks {
            if !block.body.is_empty() || matches!(block.terminator, Terminator::Instruction(..)) {
                assert!(
                    block.start_pc < bytecode_len,
                    "block start_pc {} >= bytecode_len {}",
                    block.start_pc,
                    bytecode_len
                );
            }
        }
    }
}
