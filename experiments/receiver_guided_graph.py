#!/usr/bin/env python3
"""Receiver-driven graph Newton experiment for BFFT transport cells.

This file deliberately does not modify the live viewer or implement SAD.
It tests the narrower hypothesis suggested by SAD, MegaLights, and the
Alpha/Sigma measurements:

1. BFFT fixes the support geometry.
2. The current partition-of-unity renderer supplies its own cell graph.
3. Reconstruction residual is receiver demand.
4. One sparse damped Newton solve transports additive site reach on that
   measured graph.

For a pixel jointly explained by owner i and runner j,

    y = w p_i + (1-w) p_j
    w = sigmoid(beta * (d_j - d_i))

and an additive reach bias changes the distance gap in direction e_i-e_j.
The Gauss-Newton curvature contributed by that pixel is therefore a weighted
edge Laplacian.  Summing pixels gives a sparse positive-semidefinite system
whose adjacency is exactly current co-ownership, not a guessed distance
graph.

Run a quick paired trial:

    PYTHONPATH=.:viewer .venv/bin/python \
        experiments/receiver_guided_graph.py --max-side 128 --cells 700

The established normalized-gradient reach descent is included as a control.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402
import gallery  # noqa: E402
from bfft.effects import lab_to_srgb  # noqa: E402
from bfft.vision import SingleStageDecompositionObjective  # noqa: E402
from claude_trial_sigma import LAB_WEIGHTS, SigmaVoronoi  # noqa: E402
from transport_voronoi import Config  # noqa: E402


class ReceiverGuidedVoronoi(SigmaVoronoi):
    """Sigma model with a sparse graph-Newton site-reach update."""

    def decomposition_objective(self, cfg):
        key = (
            float(cfg.lam), float(cfg.mu), int(cfg.passes), self.rgb.shape)
        cached = getattr(self, "_decomposition_objective", None)
        if cached is None or cached[0] != key:
            cached = (
                key,
                SingleStageDecompositionObjective(
                    self.rgb, lam=cfg.lam, mu=cfg.mu, passes=cfg.passes,
                    threads=4))
            self._decomposition_objective = cached
        return cached[1]

    def _reach_gradient_and_laplacian(
            self, fields, forest, cartoon_softness, texture_softness):
        precision = float(np.clip(self.cfg.detail_precision, 0.0, 1.5))
        packed = []
        curvature = np.zeros(self.npix, dtype=np.float64)

        for name, scale, softness in (
                ("base", 1.0, cartoon_softness),
                ("detail", precision, texture_softness)):
            context, solved = fields[name]
            packed.append({
                "w1": context["w1"],
                "w2": context["w2"],
                "softness": softness,
                "scale": scale,
                "pred_first": solved["pred_first"],
                "pred_second": solved["pred_second"],
            })

            # J_x = d reconstruction(x) / d(reach_owner-reach_runner).
            beta = 0.5 * float(softness)
            blend_derivative = (
                scale * beta * context["w1"] * context["w2"])
            atom_gap = solved["pred_first"] - solved["pred_second"]
            atom_energy = np.sum(
                LAB_WEIGHTS[None, :] * atom_gap * atom_gap, axis=1)
            curvature += (
                2.0 * blend_derivative * blend_derivative * atom_energy)

        _, sensitivity = self.metric_gradient(packed, forest)
        gradient = self.weight_gradient(sensitivity)

        valid = (
            (self.second >= 0) &
            (self.owner >= 0) &
            (self.owner != self.second) &
            np.isfinite(curvature) &
            (curvature > 1e-16))
        owner = self.owner[valid].astype(np.int64)
        runner = self.second[valid].astype(np.int64)
        weight = curvature[valid]
        n = len(self.seeds)

        rows = np.concatenate([owner, runner, owner, runner])
        cols = np.concatenate([owner, runner, runner, owner])
        data = np.concatenate([weight, weight, -weight, -weight])
        laplacian = sparse.coo_matrix(
            (data, (rows, cols)), shape=(n, n)).tocsr()
        laplacian.sum_duplicates()
        return gradient, laplacian, {
            "active_pixels": int(np.sum(valid)),
            "graph_edges": int(sparse.triu(
                laplacian, k=1).count_nonzero()),
            "gradient_norm": float(np.linalg.norm(gradient)),
        }

    def receiver_newton_step(
            self, cartoon_softness=4.0, texture_softness=16.0,
            damping=0.05, trust=1.5, backtracks=5,
            accept_objective="composite"):
        """Take one graph-Newton reach step with exact rendered evaluations."""
        started = time.perf_counter()
        n = len(self.seeds)
        current = getattr(self, "reach_offset", None)
        offset = (
            np.zeros(n, dtype=np.float64)
            if current is None or len(current) != n
            else np.asarray(current, dtype=np.float64).copy())

        forest = self.walk_with_predecessors(
            self.scale_field, reach_offset=offset)
        fields = self._solve_direct_pair(
            cartoon_softness=cartoon_softness,
            texture_softness=texture_softness,
            enriched=False)
        self._apply_direct_fields(
            fields["base"][1]["field"],
            fields["detail"][1]["field"],
            "receiver graph baseline")
        baseline_mse = float(self.rgb_mse)
        baseline_psnr = float(self.psnr)
        baseline_score = score(self, self.cfg)
        key = (
            "objective"
            if str(accept_objective).lower() == "composite"
            else "rgb_mse")
        baseline_measure = float(baseline_score[key])

        gradient, laplacian, report = (
            self._reach_gradient_and_laplacian(
                fields, forest, cartoon_softness, texture_softness))
        diagonal = laplacian.diagonal()
        positive = diagonal[diagonal > 1e-16]
        scale = (
            float(np.median(positive)) if positive.size else 1.0)
        ridge = max(float(damping) * scale, 1e-12)
        system = laplacian + sparse.eye(n, format="csr") * ridge
        direction = np.asarray(
            spsolve(system.tocsc(), -gradient), dtype=np.float64)
        direction[~np.isfinite(direction)] = 0.0
        direction -= float(np.mean(direction))

        spacing = math.sqrt(self.npix / max(n, 1))
        span = float(np.percentile(np.abs(direction), 95.0))
        limit = max(float(trust) * spacing, 1e-9)
        if span > limit:
            direction *= limit / span

        best_offset = offset.copy()
        best_mse = baseline_mse
        best_psnr = baseline_psnr
        best_measure = baseline_measure
        best_score = baseline_score
        accepted_alpha = 0.0
        trace = []
        for attempt in range(max(1, int(backtracks))):
            alpha = 0.5 ** attempt
            candidate = offset + alpha * direction
            candidate -= float(np.mean(candidate))
            self.walk_with_predecessors(
                self.scale_field, reach_offset=candidate)
            candidate_fields = self._solve_direct_pair(
                cartoon_softness=cartoon_softness,
                texture_softness=texture_softness,
                enriched=False)
            self._apply_direct_fields(
                candidate_fields["base"][1]["field"],
                candidate_fields["detail"][1]["field"],
                "receiver graph Newton")
            candidate_score = score(self, self.cfg)
            candidate_measure = float(candidate_score[key])
            trace.append({
                "alpha": float(alpha),
                "psnr": float(self.psnr),
                "rgb_mse": float(self.rgb_mse),
                "objective": float(candidate_score["objective"]),
            })
            if candidate_measure < best_measure:
                best_measure = candidate_measure
                best_mse = float(self.rgb_mse)
                best_psnr = float(self.psnr)
                best_score = candidate_score
                best_offset = candidate.copy()
                accepted_alpha = float(alpha)

        self.reach_offset = best_offset
        self.walk_with_predecessors(
            self.scale_field, reach_offset=best_offset)
        final_fields = self._solve_direct_pair(
            cartoon_softness=cartoon_softness,
            texture_softness=texture_softness,
            enriched=False)
        self._apply_direct_fields(
            final_fields["base"][1]["field"],
            final_fields["detail"][1]["field"],
            "receiver graph Newton")
        self.last_ms = (time.perf_counter() - started) * 1000.0
        report.update({
            "baseline_psnr": baseline_psnr,
            "baseline_objective": float(baseline_score["objective"]),
            "final_psnr": best_psnr,
            "final_objective": float(best_score["objective"]),
            "gain_db": best_psnr - baseline_psnr,
            "objective_gain": (
                float(baseline_score["objective"]) -
                float(best_score["objective"])),
            "accept_objective": key,
            "accepted_alpha": accepted_alpha,
            "damping": float(damping),
            "ridge": ridge,
            "direction_p95": span,
            "trust_limit": limit,
            "offset_span": [
                float(best_offset.min()), float(best_offset.max())],
            "trace": trace,
            "elapsed_s": self.last_ms / 1000.0,
        })
        return report

    def receiver_newton(
            self, outer_steps=4, cartoon_softness=4.0,
            texture_softness=16.0, damping=0.01, trust=1.5,
            backtracks=5, accept_objective="composite"):
        """Cross a small number of ownership chambers, rebuilding each time."""
        reports = []
        for _ in range(max(1, int(outer_steps))):
            report = self.receiver_newton_step(
                cartoon_softness=cartoon_softness,
                texture_softness=texture_softness,
                damping=damping,
                trust=trust,
                backtracks=backtracks,
                accept_objective=accept_objective)
            reports.append(report)
            if report["accepted_alpha"] <= 0.0:
                break
        starting_psnr = (
            float(reports[0]["baseline_psnr"])
            if reports else float(self.psnr))
        return {
            "starting_psnr": starting_psnr,
            "final_psnr": float(self.psnr),
            "gain_db": float(self.psnr) - starting_psnr,
            "outer_steps": len(reports),
            "steps": reports,
        }

    def _softness_gauss_newton(self, fields):
        """Two-variable GN model for cartoon/detail overlap temperatures."""
        precision = float(np.clip(self.cfg.detail_precision, 0.0, 1.5))
        jacobian = []
        for name, scale in (("base", 1.0), ("detail", precision)):
            context, solved = fields[name]
            softness = float(context["softness"])
            gap = self.d2 - self.d1
            derivative = (
                0.5 * softness * gap * context["w1"] * context["w2"])
            derivative[~context["valid"]] = 0.0
            jacobian.append(
                scale * derivative[:, None] *
                (solved["pred_first"] - solved["pred_second"]))
        residual = (
            self.lab - self.reconstruction).reshape(-1, 3)
        weights = LAB_WEIGHTS[None, :]
        gradient = np.array([
            -2.0 * float(np.sum(weights * residual * column))
            for column in jacobian], dtype=np.float64)
        hessian = np.empty((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                hessian[i, j] = 2.0 * float(np.sum(
                    weights * jacobian[i] * jacobian[j]))
        scale = max(float(np.trace(hessian)) * 0.5, 1e-12)
        hessian += np.eye(2) * (1e-3 * scale)
        try:
            direction = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            direction = np.zeros(2, dtype=np.float64)
        direction = np.clip(direction, -0.35, 0.35)
        predicted = max(float(
            -(gradient @ direction +
              0.5 * direction @ hessian @ direction)), 0.0)
        return direction, predicted, gradient, hessian

    def receiver_trust_step(
            self, cartoon_softness=4.0, texture_softness=16.0,
            damping=0.05, trust=1.5, accept_objective="composite",
            joint_softness=True):
        """One stronger receiver round with one proposal and one retry.

        Reach is updated on the measured graph.  Optionally, two additional
        GN variables update the logarithms of cartoon and texture overlap.
        The nonlinear result is always judged by the requested measured
        objective; a failed full step receives one half-step retry.
        """
        for name in ("fusion_labels", "fusion_groups", "fusion_trace"):
            if hasattr(self, name):
                delattr(self, name)
        started = time.perf_counter()
        n = len(self.seeds)
        offset = np.asarray(
            getattr(self, "reach_offset", np.zeros(n)), dtype=np.float64)
        if offset.shape != (n,):
            offset = np.zeros(n, dtype=np.float64)
        offset = offset.copy()

        forest = self.walk_with_predecessors(
            self.scale_field, reach_offset=offset)
        fields = self._solve_direct_pair(
            cartoon_softness, texture_softness, enriched=False)
        base_field = fields["base"][1]["field"].copy()
        detail_field = fields["detail"][1]["field"].copy()
        self._apply_direct_fields(
            base_field, detail_field, "receiver trust baseline")
        key = (
            "objective"
            if str(accept_objective).lower() == "composite"
            else "rgb_mse")
        measure = (
            (lambda: score(self, self.cfg))
            if key == "objective"
            else (lambda: rgb_score(self)))
        baseline = measure()
        baseline_measure = float(baseline[key])

        gradient, laplacian, report = self._reach_gradient_and_laplacian(
            fields, forest, cartoon_softness, texture_softness)
        diagonal = laplacian.diagonal()
        positive = diagonal[diagonal > 1e-16]
        scale = float(np.median(positive)) if positive.size else 1.0
        ridge = max(float(damping) * scale, 1e-12)
        system = laplacian + sparse.eye(n, format="csr") * ridge
        direction = np.asarray(
            spsolve(system.tocsc(), -gradient), dtype=np.float64)
        direction[~np.isfinite(direction)] = 0.0
        direction -= float(np.mean(direction))
        spacing = math.sqrt(self.npix / max(n, 1))
        p95 = float(np.percentile(np.abs(direction), 95.0))
        limit = max(float(trust) * spacing, 1e-9)
        if p95 > limit:
            direction *= limit / p95
        predicted_reach = max(float(
            -(gradient @ direction +
              0.5 * direction @ (system @ direction))), 0.0)

        if joint_softness:
            log_temperature, predicted_softness, _, _ = (
                self._softness_gauss_newton(fields))
        else:
            log_temperature = np.zeros(2, dtype=np.float64)
            predicted_softness = 0.0

        trace = []
        accepted = None
        for alpha in (1.0, 0.5):
            candidate_offset = offset + alpha * direction
            candidate_offset -= float(np.mean(candidate_offset))
            candidate_softness = np.array(
                [cartoon_softness, texture_softness], dtype=np.float64)
            candidate_softness *= np.exp(alpha * log_temperature)
            candidate_softness = np.clip(candidate_softness, 0.5, 40.0)
            self.walk_with_predecessors(
                self.scale_field, reach_offset=candidate_offset)
            candidate_fields = self._solve_direct_pair(
                float(candidate_softness[0]),
                float(candidate_softness[1]), enriched=False)
            self._apply_direct_fields(
                candidate_fields["base"][1]["field"],
                candidate_fields["detail"][1]["field"],
                "receiver joint trust")
            measured = measure()
            value = float(measured[key])
            predicted = (
                alpha * (predicted_reach + predicted_softness) -
                0.5 * alpha * alpha *
                (predicted_reach + predicted_softness))
            actual = baseline_measure - value
            trace.append({
                "alpha": alpha,
                "objective": float(measured["objective"]),
                "psnr": float(measured["psnr"]),
                "actual_reduction": actual,
                "predicted_local_reduction": predicted,
            })
            if value < baseline_measure:
                accepted = (
                    candidate_offset.copy(), candidate_softness.copy(),
                    measured)
                break

        if accepted is None:
            self.reach_offset = offset
            self.walk_with_predecessors(
                self.scale_field, reach_offset=offset)
            self._apply_direct_fields(
                base_field, detail_field, "rejected receiver trust step")
            final = baseline
            accepted_alpha = 0.0
            final_softness = np.array(
                [cartoon_softness, texture_softness], dtype=np.float64)
        else:
            self.reach_offset, final_softness, final = accepted
            accepted_alpha = float(trace[-1]["alpha"])

        self.last_ms = (time.perf_counter() - started) * 1000.0
        report.update({
            "baseline_psnr": float(baseline["psnr"]),
            "baseline_objective": float(baseline["objective"]),
            "final_psnr": float(final["psnr"]),
            "final_objective": float(final["objective"]),
            "gain_db": float(final["psnr"] - baseline["psnr"]),
            "objective_gain": float(
                baseline["objective"] - final["objective"]),
            "accepted_alpha": accepted_alpha,
            "evaluations": len(trace),
            "trace": trace,
            "cartoon_softness": float(final_softness[0]),
            "texture_softness": float(final_softness[1]),
            "softness_log_step": log_temperature.tolist(),
            "elapsed_s": self.last_ms / 1000.0,
        })
        return report

    def receiver_trust(
            self, outer_steps=4, cartoon_softness=4.0,
            texture_softness=16.0, damping=0.05, trust=1.5,
            accept_objective="composite", joint_softness=True):
        reports = []
        csoft = float(cartoon_softness)
        tsoft = float(texture_softness)
        for _ in range(max(1, int(outer_steps))):
            report = self.receiver_trust_step(
                csoft, tsoft, damping=damping, trust=trust,
                accept_objective=accept_objective,
                joint_softness=joint_softness)
            reports.append(report)
            csoft = report["cartoon_softness"]
            tsoft = report["texture_softness"]
            if report["accepted_alpha"] <= 0.0:
                break
        start = (
            float(reports[0]["baseline_psnr"])
            if reports else float(self.psnr))
        return {
            "starting_psnr": start,
            "final_psnr": float(self.psnr),
            "gain_db": float(self.psnr) - start,
            "outer_steps": len(reports),
            "evaluations": int(sum(r["evaluations"] for r in reports)),
            "cartoon_softness": csoft,
            "texture_softness": tsoft,
            "steps": reports,
        }


def native_components(image, cfg):
    split = bfft.meyer_channels(
        image, space="oklab_lc", lam=cfg.lam, mu=cfg.mu,
        passes=cfg.passes, threads=4)
    scale = np.maximum(split.scale[None, None, :], 1e-12)
    return split.cartoon / scale, split.texture / scale


def score(model, cfg):
    reconstruction = np.clip(
        lab_to_srgb(model.reconstruction), 0.0, 1.0)
    if hasattr(model, "decomposition_objective"):
        return model.decomposition_objective(cfg).evaluate(reconstruction)
    target_cartoon, target_texture = native_components(model.rgb, cfg)
    recon_cartoon, recon_texture = native_components(reconstruction, cfg)
    rgb_mse = float(np.mean((model.rgb - reconstruction) ** 2))
    cartoon_mse = float(np.mean((target_cartoon - recon_cartoon) ** 2))
    texture_mse = float(np.mean((target_texture - recon_texture) ** 2))
    return {
        "psnr": -10.0 * math.log10(max(rgb_mse, 1e-12)),
        "rgb_mse": rgb_mse,
        "cartoon_mse": cartoon_mse,
        "texture_mse": texture_mse,
        "objective": rgb_mse + cartoon_mse + texture_mse,
    }


def rgb_score(model):
    """Cheap full-resolution acceptance score for interactive HD rounds."""
    reconstruction = np.clip(
        lab_to_srgb(model.reconstruction), 0.0, 1.0)
    rgb_mse = float(np.mean((model.rgb - reconstruction) ** 2))
    return {
        "psnr": -10.0 * math.log10(max(rgb_mse, 1e-12)),
        "rgb_mse": rgb_mse,
        "cartoon_mse": float("nan"),
        "texture_mse": float("nan"),
        "objective": rgb_mse,
    }


def grow(image, cfg):
    model = ReceiverGuidedVoronoi(image, cfg)
    while len(model.seeds) < cfg.max_cells:
        model.step()
    for _ in range(3):
        model.step()
    return model


def clone_grown(model):
    clone = copy.deepcopy(model)
    clone.scale_field = np.ones(clone.npix, dtype=np.float64)
    clone.reach_offset = np.zeros(len(clone.seeds), dtype=np.float64)
    clone.walk_with_predecessors(
        clone.scale_field, reach_offset=clone.reach_offset)
    return clone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=["pikachu"])
    parser.add_argument("--max-side", type=int, default=128)
    parser.add_argument("--cells", type=int, default=700)
    parser.add_argument("--initial-cells", type=int, default=96)
    parser.add_argument("--split-batch", type=int, default=36)
    parser.add_argument("--passes", type=int, default=6)
    parser.add_argument("--flow-sweeps", type=int, default=24)
    parser.add_argument("--gradient-steps", type=int, default=18)
    parser.add_argument("--newton-steps", type=int, default=4)
    parser.add_argument(
        "--accept", choices=["rgb", "composite"], default="composite")
    parser.add_argument(
        "--damping", nargs="+", type=float,
        default=[0.01, 0.05, 0.2, 1.0])
    args = parser.parse_args()

    output = {"protocol": vars(args), "results": []}
    for key in args.images:
        cfg = Config(
            max_side=args.max_side,
            passes=args.passes,
            flow_sweeps=args.flow_sweeps,
            initial_cells=args.initial_cells,
            max_cells=args.cells,
            split_batch=args.split_batch,
            allocation_mode="Expected affine gain",
            recursive_memory_stages=1,
            residual_memory_weight=0.0,
            composition_discrepancy_weight=0.0)
        base = grow(gallery.load(key), cfg)

        direct = clone_grown(base)
        direct.solve_direct_coupled(4.0, 16.0)
        output["results"].append({
            "image": key,
            "trial": "direct",
            **score(direct, cfg),
        })

        gradient = clone_grown(base)
        gradient_report = gradient.optimize_site_reach(
            steps=args.gradient_steps,
            learning_rate=0.35,
            cartoon_softness=4.0,
            texture_softness=16.0)
        output["results"].append({
            "image": key,
            "trial": f"gradient_{args.gradient_steps}",
            **score(gradient, cfg),
            "reach": gradient_report,
        })

        for damping in args.damping:
            newton = clone_grown(base)
            report = newton.receiver_newton(
                outer_steps=args.newton_steps,
                cartoon_softness=4.0,
                texture_softness=16.0,
                damping=damping,
                accept_objective=args.accept)
            result = {
                "image": key,
                "trial": f"newton_{damping:g}",
                **score(newton, cfg),
                "reach": report,
            }
            output["results"].append(result)
            print(
                f"{key:9s} newton {damping:5.2g}: "
                f"{result['psnr']:6.2f} dB, "
                f"gain {report['gain_db']:+.3f}, "
                f"objective {result['objective']:.3e}",
                file=sys.stderr, flush=True)

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
