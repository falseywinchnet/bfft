"""Label-blind coarse catalog for whole-span pronunciation proposals."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .generic_word_template_bank import GenericWordTemplateBank


FORMAT = "ostensibly_generic_pronunciation_catalog_v1"


@dataclass(frozen=True)
class GenericPronunciationCatalog:
    """Vectorized coarse signatures indexed only by pronunciation length."""

    phone_keys: np.ndarray
    lengths: np.ndarray
    surfaces: np.ndarray
    masses: np.ndarray
    context_policy: str

    def __post_init__(self) -> None:
        count, realizations, time_bins, row_quantiles = self.surfaces.shape
        if (
            count < 1
            or realizations < 1
            or time_bins < 4
            or row_quantiles < 4
            or self.phone_keys.shape != (count,)
            or self.lengths.shape != (count,)
            or self.masses.shape != (count, realizations, time_bins)
            or self.context_policy not in ("unconditional", "maximum", "mixed")
        ):
            raise ValueError("generic pronunciation catalog has invalid shape")

    @property
    def time_bins(self) -> int:
        return int(self.surfaces.shape[2])

    @property
    def row_quantiles(self) -> int:
        return int(self.surfaces.shape[3])

    @property
    def realization_count(self) -> int:
        return int(self.surfaces.shape[1])

    def rank(
        self,
        query_signature: tuple[np.ndarray, np.ndarray],
        allowed_lengths: Iterable[int],
        count: int = 128,
        maximum_shift: int = 1,
        batch_size: int = 1024,
    ) -> tuple[tuple[tuple[str, ...], float], ...]:
        """Return exact coarse-signature ranks without using phone labels."""

        query_surface = np.asarray(query_signature[0], dtype=np.float32)
        query_mass = np.asarray(query_signature[1], dtype=np.float32)
        if (
            query_surface.shape != (self.time_bins, self.row_quantiles)
            or query_mass.shape != (self.time_bins,)
            or count < 1
            or maximum_shift < 0
            or batch_size < 1
        ):
            raise ValueError("generic catalog query configuration is invalid")
        allowed = np.asarray(sorted(set(int(value) for value in allowed_lengths)))
        indices = np.flatnonzero(np.isin(self.lengths, allowed))
        if not indices.size:
            return ()
        output_distances = np.full(indices.size, np.inf, dtype=np.float32)
        time_bins = self.time_bins
        for batch0 in range(0, indices.size, batch_size):
            batch1 = min(indices.size, batch0 + batch_size)
            selected = indices[batch0:batch1]
            surfaces = np.asarray(self.surfaces[selected], dtype=np.float32)
            masses = np.asarray(self.masses[selected], dtype=np.float32)
            realization_distances = np.full(
                surfaces.shape[:2], np.inf, dtype=np.float32
            )
            for shift in range(-maximum_shift, maximum_shift + 1):
                if shift < 0:
                    query_slice = slice(-shift, time_bins)
                    reference_slice = slice(0, time_bins + shift)
                elif shift > 0:
                    query_slice = slice(0, time_bins - shift)
                    reference_slice = slice(shift, time_bins)
                else:
                    query_slice = reference_slice = slice(None)
                left = query_surface[query_slice]
                right = surfaces[:, :, reference_slice]
                base = np.sqrt(
                    np.mean((right - left[None, None]) ** 2, axis=(2, 3))
                )
                derivative = np.sqrt(
                    np.mean(
                        (
                            np.diff(right, axis=2)
                            - np.diff(left, axis=0)[None, None]
                        )
                        ** 2,
                        axis=(2, 3),
                    )
                )
                mass = np.sqrt(
                    np.mean(
                        (
                            masses[:, :, reference_slice]
                            - query_mass[query_slice][None, None]
                        )
                        ** 2,
                        axis=2,
                    )
                )
                realization_distances = np.minimum(
                    realization_distances,
                    base + 0.5 * derivative + 0.25 * mass,
                )
            output_distances[batch0:batch1] = np.mean(
                realization_distances, axis=1
            )
        order = np.lexsort((self.phone_keys[indices], output_distances))[:count]
        return tuple(
            (
                tuple(str(self.phone_keys[indices[position]]).split()),
                float(output_distances[position]),
            )
            for position in order
        )

    def save(self, path: Path) -> None:
        metadata = json.dumps(
            {
                "format": FORMAT,
                "context_policy": self.context_policy,
            },
            sort_keys=True,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            metadata=np.asarray(metadata),
            phone_keys=self.phone_keys,
            lengths=self.lengths,
            surfaces=self.surfaces,
            masses=self.masses,
        )

    @classmethod
    def load(cls, path: Path) -> "GenericPronunciationCatalog":
        with np.load(path, allow_pickle=False) as document:
            metadata = json.loads(str(document["metadata"]))
            if metadata.get("format") != FORMAT:
                raise ValueError("unsupported generic pronunciation catalog")
            return cls(
                phone_keys=np.asarray(document["phone_keys"]).astype(str),
                lengths=np.asarray(document["lengths"], dtype=np.int16),
                surfaces=np.asarray(document["surfaces"]),
                masses=np.asarray(document["masses"]),
                context_policy=str(metadata["context_policy"]),
            )


def compile_generic_pronunciation_catalog(
    pronunciations: Iterable[tuple[str, ...]],
    bank: GenericWordTemplateBank,
    realization_count: int = 4,
    time_bins: int = 16,
    row_quantiles: int = 8,
    context_policy: str = "unconditional",
    storage_dtype: str = "float16",
) -> GenericPronunciationCatalog:
    """Compile unique generic word signatures into a compact search tensor."""

    if storage_dtype not in ("float16", "float32"):
        raise ValueError("generic catalog storage dtype is invalid")
    phones = sorted(set(tuple(value) for value in pronunciations))
    if not phones:
        raise ValueError("generic pronunciation catalog requires pronunciations")
    signatures = [
        bank.word_signatures(
            value,
            realization_count,
            time_bins,
            row_quantiles,
            context_policy=context_policy,
        )
        for value in phones
    ]
    dtype = np.float16 if storage_dtype == "float16" else np.float32
    return GenericPronunciationCatalog(
        phone_keys=np.asarray([" ".join(value) for value in phones]),
        lengths=np.asarray([len(value) for value in phones], dtype=np.int16),
        surfaces=np.asarray(
            [[signature[0] for signature in value] for value in signatures],
            dtype=dtype,
        ),
        masses=np.asarray(
            [[signature[1] for signature in value] for value in signatures],
            dtype=dtype,
        ),
        context_policy=context_policy,
    )
