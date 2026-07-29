#!/usr/bin/env python3
"""Map direct X-A5 adjustment opcodes to a recovered case address.

This emulates only the dispatcher at 0xc01c96c8. External calls are skipped;
the three table helpers used for case selection are modeled explicitly. It
never accesses a camera or modifies a firmware image.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from unicorn import Uc, UC_ARCH_ARM, UC_HOOK_CODE, UC_MODE_ARM
from unicorn.arm_const import (
    UC_ARM_REG_LR,
    UC_ARM_REG_PC,
    UC_ARM_REG_R0,
    UC_ARM_REG_R1,
    UC_ARM_REG_R2,
    UC_ARM_REG_R6,
    UC_ARM_REG_R10,
    UC_ARM_REG_SP,
)

BASE = 0xC0000000
ENTRY = 0xC01C96C8
DISPATCH_END = 0xC01CA430
STACK = 0xD0000000


def branch_target(address: int, instruction: int) -> int:
    displacement = instruction & 0xFFFFFF
    if displacement & 0x800000:
        displacement -= 1 << 24
    return (address + 8 + (displacement << 2)) & 0xFFFFFFFF


def visits_target(
    image: bytes, opcode: int, argument: int, target: int
) -> bool:
    emulator = Uc(UC_ARCH_ARM, UC_MODE_ARM)
    emulator.mem_map(BASE, 0x10000000)
    emulator.mem_write(BASE, image)
    emulator.mem_map(STACK, 0x10000)
    emulator.reg_write(UC_ARM_REG_SP, STACK + 0x8000)
    emulator.reg_write(UC_ARM_REG_LR, DISPATCH_END)
    emulator.reg_write(UC_ARM_REG_R0, 0)
    emulator.reg_write(UC_ARM_REG_R6, argument)
    emulator.reg_write(UC_ARM_REG_R10, opcode)
    reached = False

    def table_search(uc: Uc, table: int, high: int, key: int) -> int:
        low = -1
        result = struct.unpack("<H", uc.mem_read(table - 2, 2))[0]
        while low + 1 != high:
            middle = (high + low) >> 1
            candidate = uc.mem_read(table + middle, 1)[0]
            if key < candidate:
                high = middle
            elif key > candidate:
                low = middle
            else:
                result = struct.unpack(
                    "<H", uc.mem_read(table - (middle * 2 + 4), 2)
                )[0]
                break
        return result

    def hook(uc: Uc, address: int, size: int, _: object) -> None:
        nonlocal reached
        if address == target:
            reached = True
            uc.emu_stop()
            return
        if address >= DISPATCH_END:
            uc.emu_stop()
            return

        instruction = struct.unpack("<I", uc.mem_read(address, 4))[0]
        if (instruction >> 24) & 0xF != 0xB:
            return
        destination = branch_target(address, instruction)

        if destination == 0xC0599B64:
            index = uc.reg_read(UC_ARM_REG_R0)
            table = uc.reg_read(UC_ARM_REG_R1)
            value = struct.unpack("<H", uc.mem_read(table + 2 * index, 2))[0]
            uc.reg_write(UC_ARM_REG_R1, value)
            uc.reg_write(UC_ARM_REG_PC, address + 4)
            return
        if destination == 0xC02124E8:
            index = uc.reg_read(UC_ARM_REG_R1)
            table = uc.reg_read(UC_ARM_REG_R2)
            value = struct.unpack("<H", uc.mem_read(table + 2 * index, 2))[0]
            uc.reg_write(UC_ARM_REG_R2, value)
            uc.reg_write(UC_ARM_REG_PC, address + 4)
            return
        if destination == 0xC01E5BF4:
            table = uc.reg_read(UC_ARM_REG_R0)
            high = uc.reg_read(UC_ARM_REG_R1)
            key = uc.reg_read(UC_ARM_REG_R2) & 0xFF
            result = table_search(uc, table, high, key)
            uc.reg_write(UC_ARM_REG_R0, result)
            uc.reg_write(UC_ARM_REG_PC, address + 4)
            return
        if destination == 0xC01CA5C0:
            table = uc.reg_read(UC_ARM_REG_R0) + 0x16
            key = uc.reg_read(UC_ARM_REG_R2) & 0xFF
            result = table_search(uc, table, 10, key)
            uc.reg_write(UC_ARM_REG_R0, result)
            uc.reg_write(UC_ARM_REG_PC, address + 4)
            return

        if not (ENTRY <= destination < DISPATCH_END):
            uc.reg_write(UC_ARM_REG_PC, address + 4)

    emulator.hook_add(UC_HOOK_CODE, hook)
    try:
        emulator.emu_start(ENTRY, DISPATCH_END, count=20_000)
    except Exception:
        pass
    return reached


def integer(text: str) -> int:
    return int(text, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("target", type=integer)
    parser.add_argument("--argument", type=integer, default=1)
    parser.add_argument("--limit", type=integer, default=900)
    args = parser.parse_args()

    image = args.image.read_bytes()
    for opcode in range(args.limit):
        if visits_target(image, opcode, args.argument, args.target):
            print(f"opcode={opcode} target=0x{args.target:08x}")


if __name__ == "__main__":
    main()
