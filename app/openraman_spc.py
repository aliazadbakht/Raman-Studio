# SPDX-License-Identifier: CERN-OHL-W-2.0
# Copyright (c) 2026 Wfront Principle B.V. (https://wfront.nl)
# Copyright (c) 2026 Precisometer B.V. (https://precisometer.com)
# Portions Copyright (c) Luc Boussemaere (The Pulsar) — the .spc container
# format (storage objects, string table, RLE0 encoding) is derived from the
# OpenRAMAN Spectrum Analyzer (https://www.open-raman.org/), CERN-OHL-W-2.0.
"""Read and write OpenRAMAN Spectrum Analyzer .spc files."""
from __future__ import annotations

import os
import random
import struct
from dataclasses import dataclass, field

import numpy as np


SPC_IDENT = 0x53504330  # C++ multi-character literal 'SPC0'
ENCRYPTION_KEY = 0xFEEFBEEF
ENCRYPTION_BLOCK_SIZE = 4

TYPE_SPECTRE_FILE = ".?AVSpectreFile@@"
TYPE_STORAGE_VECTOR_DOUBLE = ".?AV?$storage_vector@N@@"
TYPE_STORAGE_STRING = ".?AVstorage_string@@"
TYPE_DOUBLE = "double"
TYPE_CHAR = "char"
TYPE_SIZE_T = "size_t"

OBJ_HEADER = struct.Struct("<QQQQQQQQQQQQQ")
OBJ_VAR = struct.Struct("<QQQQQ")
OBJ_CHILD = struct.Struct("<QQQQ")
SPC_HEADER = struct.Struct("<IIQQI4xQ10I")
SPC_BUFFER = struct.Struct("<B7xQQ")


@dataclass
class Var:
    owner: str
    name: str
    type_name: str
    data: bytes


@dataclass
class StorageObject:
    owner: str = ""
    name: str = ""
    type_name: str = ""
    vars: list[Var] = field(default_factory=list)
    children: list["StorageObject"] = field(default_factory=list)

    def child(self, name: str) -> "StorageObject | None":
        for item in self.children:
            if item.name == name:
                return item
        return None


def _cstring(data: bytes, offset: int) -> str:
    end = data.index(b"\0", offset)
    return data[offset:end].decode("utf-8", errors="replace")


def _append_cstring(strings: bytearray, value: str) -> int:
    offset = len(strings)
    strings.extend(value.encode("utf-8"))
    strings.append(0)
    return offset


def _pack_object(obj: StorageObject) -> bytes:
    strings = bytearray()
    vars_buf = bytearray()
    children_buf = bytearray()
    data = bytearray()

    type_ofs = _append_cstring(strings, obj.type_name)
    owner_ofs = _append_cstring(strings, obj.owner)
    name_ofs = _append_cstring(strings, obj.name)

    for var in obj.vars:
        var_owner = _append_cstring(strings, var.owner)
        var_name = _append_cstring(strings, var.name)
        var_type = _append_cstring(strings, var.type_name)
        data_ofs = len(data)
        data.extend(var.data)
        vars_buf.extend(OBJ_VAR.pack(var_owner, var_name, var_type, data_ofs, len(var.data)))

    for child in obj.children:
        child_owner = _append_cstring(strings, child.owner)
        child_name = _append_cstring(strings, child.name)
        packed = _pack_object(child)
        data_ofs = len(data)
        data.extend(packed)
        children_buf.extend(OBJ_CHILD.pack(child_owner, child_name, data_ofs, len(packed)))

    out = bytearray(OBJ_HEADER.size)
    strings_ofs = len(out)
    out.extend(strings)
    vars_ofs = len(out)
    out.extend(vars_buf)
    children_ofs = len(out)
    out.extend(children_buf)
    data_ofs = len(out)
    out.extend(data)

    out[:OBJ_HEADER.size] = OBJ_HEADER.pack(
        type_ofs,
        owner_ofs,
        name_ofs,
        len(obj.vars),
        len(obj.children),
        strings_ofs,
        len(strings),
        vars_ofs,
        len(vars_buf),
        children_ofs,
        len(children_buf),
        data_ofs,
        len(data),
    )
    return bytes(out)


def _unpack_object(buffer: bytes) -> StorageObject:
    (
        type_ofs,
        owner_ofs,
        name_ofs,
        n_vars,
        n_children,
        strings_ofs,
        strings_size,
        vars_ofs,
        vars_size,
        children_ofs,
        children_size,
        data_ofs,
        data_size,
    ) = OBJ_HEADER.unpack_from(buffer, 0)

    if vars_size != n_vars * OBJ_VAR.size or children_size != n_children * OBJ_CHILD.size:
        raise ValueError("Invalid OpenRAMAN storage object")

    strings = buffer[strings_ofs : strings_ofs + strings_size]
    data = buffer[data_ofs : data_ofs + data_size]
    obj = StorageObject(
        owner=_cstring(strings, owner_ofs),
        name=_cstring(strings, name_ofs),
        type_name=_cstring(strings, type_ofs),
    )

    for i in range(n_vars):
        var_owner, var_name, var_type, var_data_ofs, var_size = OBJ_VAR.unpack_from(
            buffer, vars_ofs + i * OBJ_VAR.size
        )
        obj.vars.append(
            Var(
                _cstring(strings, var_owner),
                _cstring(strings, var_name),
                _cstring(strings, var_type),
                bytes(data[var_data_ofs : var_data_ofs + var_size]),
            )
        )

    for i in range(n_children):
        child_owner, child_name, child_data_ofs, child_size = OBJ_CHILD.unpack_from(
            buffer, children_ofs + i * OBJ_CHILD.size
        )
        child = _unpack_object(data[child_data_ofs : child_data_ofs + child_size])
        child.owner = _cstring(strings, child_owner)
        child.name = _cstring(strings, child_name)
        obj.children.append(child)

    return obj


def _rle8_decode(buffer: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(buffer), 2):
        out.extend(buffer[i + 1 : i + 2] * buffer[i])
    return bytes(out)


def _rle0_decode(buffer: bytes) -> bytes:
    out = bytearray()
    block_size = 1
    i = 0
    while i < len(buffer):
        occurrence = buffer[i]
        i += 1
        if occurrence == 0:
            block_size = buffer[i]
            i += 1
            occurrence = buffer[i]
            i += 1
        block = buffer[i : i + block_size]
        out.extend(block * occurrence)
        i += block_size
    return bytes(out)


def _decode(buffer: bytes, encoding: int) -> bytes:
    if encoding == 0:
        return buffer
    if encoding == 1:
        return _rle8_decode(buffer)
    if encoding == 2:
        return _rle0_decode(buffer)
    raise ValueError(f"Unsupported OpenRAMAN encoding: {encoding}")


def _repmat(value: int) -> int:
    value &= 0xFF
    value |= value << 8
    value |= value << 16
    return (~value) & 0xFFFFFFFF


def _crypt_words(words: list[int], key: int, forward: bool) -> list[int]:
    key = (key + _repmat(len(words))) & 0xFFFFFFFF
    words = words[:]
    for j in range(ENCRYPTION_BLOCK_SIZE):
        for i in range(j, len(words), ENCRYPTION_BLOCK_SIZE):
            temp = (~words[i]) & 0xFFFFFFFF
            words[i] = (words[i] ^ key) & 0xFFFFFFFF
            key = (key + (((~words[i]) & 0xFFFFFFFF) if forward else temp)) & 0xFFFFFFFF
            key = (key + _repmat(i)) & 0xFFFFFFFF
    return words


def _crypt_tail(buffer: bytearray, seed: int, forward: bool) -> None:
    tail = bytes(buffer[SPC_HEADER.size :])
    words = list(struct.unpack(f"<{len(tail) // 4}I", tail))
    words = _crypt_words(words, ENCRYPTION_KEY ^ seed, forward)
    buffer[SPC_HEADER.size :] = struct.pack(f"<{len(words)}I", *words)


def _checksum(buffer: bytes) -> int:
    total = 0
    i = 0
    n = len(buffer)
    while n - i >= 4:
        total = (total + (~struct.unpack_from("<I", buffer, i)[0] & 0xFFFFFFFF)) & 0xFFFFFFFF
        i += 4
    while n - i >= 2:
        total = (total + (~struct.unpack_from("<H", buffer, i)[0] & 0xFFFF)) & 0xFFFFFFFF
        i += 2
    while n - i >= 1:
        total = (total + (~buffer[i] & 0xFF)) & 0xFFFFFFFF
        i += 1
    return (~total) & 0xFFFFFFFF


def _pack_container(objects: list[StorageObject]) -> bytes:
    table = bytearray()
    data = bytearray()
    for obj in objects:
        packed = _pack_object(obj)
        offset = len(data)
        data.extend(packed)
        table.extend(SPC_BUFFER.pack(0, offset, len(packed)))

    seed = random.randint(0, 0xFFFFFFFF)
    out = bytearray(SPC_HEADER.size)
    table_ofs = len(out)
    out.extend(table)
    data_ofs = len(out)
    out.extend(data)
    padding = (len(out) - SPC_HEADER.size) % 4
    if padding:
        out.extend(b"\0" * (4 - padding))

    out[:SPC_HEADER.size] = SPC_HEADER.pack(
        SPC_IDENT,
        0,
        len(objects),
        table_ofs,
        seed,
        data_ofs,
        *([0] * 10),
    )
    checksum = _checksum(bytes(out))
    out[:SPC_HEADER.size] = SPC_HEADER.pack(
        SPC_IDENT,
        checksum,
        len(objects),
        table_ofs,
        seed,
        data_ofs,
        *([0] * 10),
    )
    _crypt_tail(out, seed, True)
    return bytes(out)


def _unpack_container(buffer: bytes) -> list[StorageObject]:
    mutable = bytearray(buffer)
    ident, old_checksum, n_buffers, table_ofs, seed, data_ofs, *_ = SPC_HEADER.unpack_from(mutable, 0)
    if ident != SPC_IDENT:
        raise ValueError("Not an OpenRAMAN .spc file")
    mutable[4:8] = b"\0\0\0\0"
    _crypt_tail(mutable, seed, False)
    if _checksum(bytes(mutable)) != old_checksum:
        raise ValueError("OpenRAMAN .spc checksum failed")

    objects = []
    for i in range(n_buffers):
        encoding, offset, size = SPC_BUFFER.unpack_from(mutable, table_ofs + i * SPC_BUFFER.size)
        raw = bytes(mutable[data_ofs + offset : data_ofs + offset + size])
        objects.append(_unpack_object(_decode(raw, encoding)))
    return objects


def _vector_object(owner: str, name: str, values) -> StorageObject:
    obj = StorageObject(owner=owner, name=name, type_name=TYPE_STORAGE_VECTOR_DOUBLE)
    for value in np.asarray(values, dtype=float):
        obj.vars.append(Var(TYPE_STORAGE_VECTOR_DOUBLE, "", TYPE_DOUBLE, struct.pack("<d", float(value))))
    return obj


def _string_object(owner: str, name: str, value: str) -> StorageObject:
    encoded = value.encode("utf-8") + b"\0"
    obj = StorageObject(owner=owner, name=name, type_name=TYPE_STORAGE_STRING)
    obj.vars.append(Var(TYPE_STORAGE_STRING, "size", TYPE_SIZE_T, struct.pack("<Q", len(encoded))))
    obj.vars.append(Var(TYPE_STORAGE_STRING, "data", TYPE_CHAR, encoded))
    return obj


def _read_vector(obj: StorageObject | None) -> np.ndarray:
    if obj is None:
        return np.array([], dtype=float)
    values = [struct.unpack("<d", var.data)[0] for var in obj.vars if len(var.data) == 8]
    return np.asarray(values, dtype=float)


def load(path: str):
    with open(path, "rb") as f:
        objects = _unpack_container(f.read())

    data_obj = next((obj for obj in objects if obj.owner == "" and obj.name == "data"), None)
    if data_obj is None:
        raise ValueError("OpenRAMAN .spc file does not contain spectrum data")

    signal = _read_vector(data_obj.child("m_data"))
    blank = _read_vector(data_obj.child("m_blank"))
    calibration_obj = next((obj for obj in objects if obj.owner == "" and obj.name == "calibration"), None)
    calibration = _read_vector(calibration_obj).tolist() if calibration_obj is not None else None
    return signal, calibration, blank if blank.size else None, {}


def save(path: str, signal, calibration=None, blank=None, uid: str = "") -> None:
    data_obj = StorageObject(owner="", name="data", type_name=TYPE_SPECTRE_FILE)
    data_obj.children.append(_vector_object(TYPE_SPECTRE_FILE, "m_data", signal))
    data_obj.children.append(_vector_object(TYPE_SPECTRE_FILE, "m_blank", [] if blank is None else blank))
    data_obj.children.append(_string_object(TYPE_SPECTRE_FILE, "m_uid", uid))

    objects = [data_obj]
    if calibration is not None:
        objects.append(_vector_object("", "calibration", calibration))

    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(_pack_container(objects))
    os.replace(tmp, path)
