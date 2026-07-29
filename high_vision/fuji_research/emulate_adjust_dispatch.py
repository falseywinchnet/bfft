#!/usr/bin/env python3
"""Brute-force X-A5 adjustment opcodes against the extracted ARM dispatcher.

This is a research aid, not a camera/card writer. It needs the third-party
``unicorn`` module and executes only the extracted firmware in memory.
"""

from __future__ import annotations

import argparse
import struct
import sys
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
RETURN = 0xC01CA42C
PARAMETER_HANDLER = 0xC02D59E8
TARGET_BRANCH = 0xC01C9F68
STACK = 0xD0000000


def branch_target(address: int, instruction: int) -> int:
    displacement = instruction & 0xFFFFFF
    if displacement & 0x800000:
        displacement -= 1 << 24
    return (address + 8 + (displacement << 2)) & 0xFFFFFFFF


def reaches_target(image: bytes, opcode: int, parameter: int) -> tuple[bool, list[int]]:
    emulator = Uc(UC_ARCH_ARM, UC_MODE_ARM)
    emulator.mem_map(BASE, 0x10000000)
    emulator.mem_write(BASE, image)
    emulator.mem_map(STACK, 0x10000)
    emulator.reg_write(UC_ARM_REG_SP, STACK + 0x8000)
    emulator.reg_write(UC_ARM_REG_LR, RETURN)
    emulator.reg_write(UC_ARM_REG_R0, 0)
    emulator.reg_write(UC_ARM_REG_R6, parameter)
    emulator.reg_write(UC_ARM_REG_R10, opcode)
    trace: list[int] = []
    reached = False

    def hook(uc: Uc, address: int, size: int, _: object) -> None:
        nonlocal reached
        if address in (TARGET_BRANCH, PARAMETER_HANDLER):
            reached = True
            uc.emu_stop()
            return
        if address == RETURN:
            uc.emu_stop()
            return
        instruction = struct.unpack("<I", uc.mem_read(address, 4))[0]
        if (instruction >> 24) & 0xF != 0xB:
            return
        target = branch_target(address, instruction)
        if target == 0xC0599B64:
            index = uc.reg_read(UC_ARM_REG_R0)
            table = uc.reg_read(UC_ARM_REG_R1)
            value = struct.unpack("<H", uc.mem_read(table + 2 * index, 2))[0]
            uc.reg_write(UC_ARM_REG_R1, value)
            uc.reg_write(UC_ARM_REG_PC, address + 4)
            return
        if target == 0xC02124E8:
            index = uc.reg_read(UC_ARM_REG_R1)
            table = uc.reg_read(UC_ARM_REG_R2)
            value = struct.unpack("<H", uc.mem_read(table + 2 * index, 2))[0]
            uc.reg_write(UC_ARM_REG_R2, value)
            uc.reg_write(UC_ARM_REG_PC, address + 4)
            return
        if target == 0xC01E5BF4:
            table = uc.reg_read(UC_ARM_REG_R0)
            high = uc.reg_read(UC_ARM_REG_R1)
            key = uc.reg_read(UC_ARM_REG_R2) & 0xFF
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
            uc.reg_write(UC_ARM_REG_R0, result)
            uc.reg_write(UC_ARM_REG_PC, address + 4)
            return
        # Preserve the two dispatcher regions. Other calls perform I/O,
        # formatting, or state changes which are irrelevant to case selection.
        if (
            0xC01C8F94 <= target < 0xC01C915C
            or 0xC01C961C <= target < 0xC01CA430
        ):
            trace.append(target)
            return
        uc.reg_write(UC_ARM_REG_PC, address + 4)

    emulator.hook_add(UC_HOOK_CODE, hook)
    try:
        emulator.emu_start(ENTRY, RETURN, count=20_000)
    except Exception:
        pass
    return reached, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--parameter", type=lambda value: int(value, 0), default=1100)
    args = parser.parse_args()
    image = args.image.read_bytes()
    found = []
    for opcode in range(1024):
        reached, trace = reaches_target(image, opcode, args.parameter)
        if reached:
            found.append(opcode)
            print(f"opcode={opcode} parameter={args.parameter} trace={trace}")
    if not found:
        print("No candidate reached the target with the conservative call policy.")
        sys.exit(1)


if __name__ == "__main__":
    main()
