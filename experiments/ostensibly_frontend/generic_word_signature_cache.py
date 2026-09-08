"""Persistent compiled conditional surfaces for generic pronunciations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FORMAT = "ostensibly_generic_word_signature_cache_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(phones: tuple[str, ...]) -> str:
    if not phones or any(" " in phone for phone in phones):
        raise ValueError("signature cache requires nonempty atomic phone labels")
    return " ".join(phones)


class GenericWordSignatureCache:
    """A mutable pronunciation map with a fixed, validated storage ABI."""

    def __init__(
        self,
        bank_fingerprint: str,
        realization_count: int,
        time_bins: int,
        row_quantiles: int,
        storage_dtype: str = "float16",
        realization_policy: str = "unconditional",
    ) -> None:
        if (
            not bank_fingerprint
            or realization_count < 1
            or time_bins < 4
            or row_quantiles < 4
            or storage_dtype not in ("float16", "float32")
            or realization_policy not in (
                "unconditional",
                "contextual",
                "contextual_mixture",
            )
        ):
            raise ValueError("generic signature cache configuration is invalid")
        self.bank_fingerprint = bank_fingerprint
        self.realization_count = realization_count
        self.time_bins = time_bins
        self.row_quantiles = row_quantiles
        self.storage_dtype = storage_dtype
        self.realization_policy = realization_policy
        self.entries: dict[
            str, tuple[tuple[np.ndarray, np.ndarray], ...]
        ] = {}
        self.dirty = False

    @property
    def dtype(self):
        return np.float16 if self.storage_dtype == "float16" else np.float32

    def get(
        self, phones: tuple[str, ...]
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...] | None:
        return self.entries.get(_key(phones))

    def put(
        self,
        phones: tuple[str, ...],
        signatures: tuple[tuple[np.ndarray, np.ndarray], ...],
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        if len(signatures) != self.realization_count:
            raise ValueError("signature cache realization count disagrees")
        stored = []
        for surface, mass in signatures:
            surface_array = np.asarray(surface, dtype=self.dtype)
            mass_array = np.asarray(mass, dtype=self.dtype)
            if surface_array.shape != (self.time_bins, self.row_quantiles):
                raise ValueError("signature cache surface resolution disagrees")
            if mass_array.shape != (self.time_bins,):
                raise ValueError("signature cache mass resolution disagrees")
            stored.append((surface_array, mass_array))
        value = tuple(stored)
        self.entries[_key(phones)] = value
        self.dirty = True
        return value

    def save(self, path: Path) -> None:
        keys = sorted(self.entries)
        if keys:
            surfaces = np.stack(
                [
                    np.stack([signature[0] for signature in self.entries[key]])
                    for key in keys
                ]
            ).astype(self.dtype, copy=False)
            masses = np.stack(
                [
                    np.stack([signature[1] for signature in self.entries[key]])
                    for key in keys
                ]
            ).astype(self.dtype, copy=False)
        else:
            surfaces = np.empty(
                (
                    0,
                    self.realization_count,
                    self.time_bins,
                    self.row_quantiles,
                ),
                dtype=self.dtype,
            )
            masses = np.empty(
                (0, self.realization_count, self.time_bins), dtype=self.dtype
            )
        metadata = json.dumps(
            {
                "format": FORMAT,
                "bank_fingerprint": self.bank_fingerprint,
                "realization_count": self.realization_count,
                "time_bins": self.time_bins,
                "row_quantiles": self.row_quantiles,
                "storage_dtype": self.storage_dtype,
                "realization_policy": self.realization_policy,
            },
            sort_keys=True,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            metadata=np.asarray(metadata),
            keys=np.asarray(keys),
            surfaces=surfaces,
            masses=masses,
        )
        temporary.replace(path)
        self.dirty = False

    @classmethod
    def load(cls, path: Path) -> "GenericWordSignatureCache":
        with np.load(path, allow_pickle=False) as document:
            metadata = json.loads(str(document["metadata"]))
            if metadata.get("format") != FORMAT:
                raise ValueError("unsupported generic signature cache format")
            cache = cls(
                bank_fingerprint=str(metadata["bank_fingerprint"]),
                realization_count=int(metadata["realization_count"]),
                time_bins=int(metadata["time_bins"]),
                row_quantiles=int(metadata["row_quantiles"]),
                storage_dtype=str(metadata["storage_dtype"]),
                realization_policy=str(
                    metadata.get("realization_policy", "unconditional")
                ),
            )
            keys = np.asarray(document["keys"]).astype(str)
            surfaces = np.asarray(document["surfaces"], dtype=cache.dtype)
            masses = np.asarray(document["masses"], dtype=cache.dtype)
        expected_surfaces = (
            keys.size,
            cache.realization_count,
            cache.time_bins,
            cache.row_quantiles,
        )
        expected_masses = (
            keys.size,
            cache.realization_count,
            cache.time_bins,
        )
        if surfaces.shape != expected_surfaces or masses.shape != expected_masses:
            raise ValueError("generic signature cache tensor shape disagrees")
        for index, key in enumerate(keys):
            cache.entries[str(key)] = tuple(
                (surfaces[index, realization], masses[index, realization])
                for realization in range(cache.realization_count)
            )
        cache.dirty = False
        return cache

    def require_configuration(
        self,
        bank_fingerprint: str,
        realization_count: int,
        time_bins: int,
        row_quantiles: int,
        storage_dtype: str,
        realization_policy: str = "unconditional",
    ) -> None:
        observed = (
            self.bank_fingerprint,
            self.realization_count,
            self.time_bins,
            self.row_quantiles,
            self.storage_dtype,
            self.realization_policy,
        )
        expected = (
            bank_fingerprint,
            realization_count,
            time_bins,
            row_quantiles,
            storage_dtype,
            realization_policy,
        )
        if observed != expected:
            raise ValueError("generic signature cache belongs to another bank/configuration")
