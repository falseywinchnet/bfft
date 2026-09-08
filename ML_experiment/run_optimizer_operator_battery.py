"""Run selected optimizers on the Muon operator-recovery witness."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ML_experiment.anchor import Anchor, RestrainedMomentumAnchor
from ML_experiment.muon_primacy import (
    OperatorProblem,
    make_block_sweep_covariances,
    make_operator_problem,
    one_step_certificate,
)
from ML_experiment.optimizers import (
    MatrixTransport,
    Muon,
    RidgeAdamW,
    TransportedAdamW,
    TransportLepton,
    TurnAdamW,
)


OPTIMIZERS = (
    "adamw",
    "sgd",
    "anchor",
    "anchor_restrained_momentum",
    "muon",
    "matrix_transport",
    "matrix_transport_nesterov",
    "matrix_transport_longitudinal_fusion",
    "lepton_transport",
)

STANDING_LRS = {
    "adamw": 0.003,
    "sgd": 0.03,
    "anchor": 1.0,
    "anchor_restrained_momentum": 1.0,
    "muon": 0.003,
    "matrix_transport": 0.003,
    "matrix_transport_nesterov": 0.003,
    "matrix_transport_longitudinal_fusion": 0.003,
    "lepton_transport": 0.003,
}

# This second protocol exposes the matrix primitive at its natural unit
# operator scale.  Comparator rates are the declared best stable rates from
# the population assay or derived shape normalization, not rates selected
# after this comparison.
OPERATOR_NATIVE_LRS = {
    "adamw": 0.01,
    "sgd": 1.9,
    "anchor": 1.0,
    "anchor_restrained_momentum": 1.0,
    "muon": 1.0,
    "matrix_transport": 1.0 / math.sqrt(32.0),
    "matrix_transport_nesterov": 1.0 / math.sqrt(32.0),
    "matrix_transport_longitudinal_fusion": 1.0 / math.sqrt(32.0),
    "lepton_transport": 1.0,
}


def _float_problem(problem: OperatorProblem) -> OperatorProblem:
    return OperatorProblem(
        problem.target.float(),
        problem.covariance.float(),
        problem.covariance_root.float(),
    )


def _make_optimizer(name: str, parameter: torch.nn.Parameter, lr: float):
    if name == "adamw":
        return torch.optim.AdamW([parameter], lr=lr, weight_decay=0.0)
    if name == "adamw_transport":
        return TransportedAdamW(
            [parameter], lr=lr, weight_decay=0.0
        )
    if name == "adamw_ridge":
        return RidgeAdamW(
            [parameter], lr=lr, weight_decay=0.0, ridge=1.0
        )
    if name == "adamw_ridge_025":
        return RidgeAdamW(
            [parameter], lr=lr, weight_decay=0.0, ridge=.25
        )
    if name == "adamw_turn":
        return TurnAdamW([parameter], lr=lr, weight_decay=0.0)
    if name == "adamw_coherence":
        return TurnAdamW(
            [parameter], lr=lr, weight_decay=0.0, initial_scale=.03,
            coherence_threshold=.3, recovery=.2,
            target_from_coherence=True, coherence_ema_beta=.9,
        )
    if name == "adamw_coherence_latch":
        return TurnAdamW(
            [parameter], lr=lr, weight_decay=0.0, initial_scale=.03,
            coherence_threshold=.5, recovery=.5,
            target_from_coherence=True, coherence_ema_beta=.9,
            binary_coherence_target=True,
        )
    if name == "adamw_warm_turn":
        return TurnAdamW(
            [parameter], lr=lr, weight_decay=0.0, initial_scale=.03,
            hold_steps=100, recovery=.05,
        )
    if name == "adamw_turn_certified":
        return TurnAdamW(
            [parameter], lr=lr, weight_decay=0.0, gradient_trust=1.0
        )
    if name == "adamw_turn_guarded":
        return TurnAdamW(
            [parameter], lr=lr, weight_decay=0.0, gradient_trust=10.0
        )
    if name == "adamw_turn_adaptive_guard":
        return TurnAdamW(
            [parameter], lr=lr, weight_decay=0.0, gradient_trust=1.0,
            gradient_trust_max=10.0, gradient_trust_threshold=.3,
        )
    if name == "adamw_turn_global_guard":
        return TurnAdamW(
            [parameter], lr=lr, weight_decay=0.0, gradient_trust=1.0,
            gradient_trust_max=10.0, gradient_trust_threshold=.3,
            global_gradient_trust=True,
        )
    if name == "sgd":
        return torch.optim.SGD([parameter], lr=lr, weight_decay=0.0)
    if name == "anchor":
        return Anchor([parameter], lr=lr, weight_decay=0.0, trust_radius=0.02)
    if name == "anchor_restrained_momentum":
        return RestrainedMomentumAnchor(
            [parameter], lr=lr, weight_decay=0.0, trust_radius=0.02
        )
    if name == "muon":
        return Muon(
            [parameter], lr=lr, weight_decay=0.0, momentum=0.95,
            nesterov=True, ns_steps=5, adjust_lr="original",
        )
    if name == "matrix_transport":
        return MatrixTransport(
            [parameter], lr=lr, weight_decay=0.0,
        )
    if name == "matrix_transport_nesterov":
        return MatrixTransport(
            [parameter], lr=lr, weight_decay=0.0, nesterov=True,
        )
    if name == "matrix_transport_longitudinal_fusion":
        return MatrixTransport(
            [parameter], lr=lr, weight_decay=0.0,
            longitudinal_fusion=True,
        )
    if name == "lepton_transport":
        return TransportLepton(
            [parameter], lr=lr, weight_decay=0.0, momentum=0.95,
            nesterov=True, bregman_epsilon=0.05, adjust_lr="original",
        )
    if name == "lepton_transport_current":
        return TransportLepton(
            [parameter], lr=lr, weight_decay=0.0, momentum=0.95,
            nesterov=True, bregman_epsilon=0.05, adjust_lr="original",
            frame_target="current",
        )
    if name == "lepton_transport_certified":
        return TransportLepton(
            [parameter], lr=lr, weight_decay=0.0, momentum=0.95,
            nesterov=True, bregman_epsilon=0.05, adjust_lr="original",
            gradient_trust=1.0,
        )
    if name == "lepton_transport_current_certified":
        return TransportLepton(
            [parameter], lr=lr, weight_decay=0.0, momentum=0.95,
            nesterov=True, bregman_epsilon=0.05, adjust_lr="original",
            frame_target="current", gradient_trust=1.0,
        )
    if name == "lepton_transport_guarded":
        return TransportLepton(
            [parameter], lr=lr, weight_decay=0.0, momentum=0.95,
            nesterov=True, bregman_epsilon=0.05, adjust_lr="original",
            gradient_trust=100.0,
        )
    if name == "lepton_transport_current_guarded":
        return TransportLepton(
            [parameter], lr=lr, weight_decay=0.0, momentum=0.95,
            nesterov=True, bregman_epsilon=0.05, adjust_lr="original",
            frame_target="current", gradient_trust=100.0,
        )
    raise KeyError(name)


def _first(history: list[dict], key: str, threshold: float) -> int | None:
    return next(
        (point["step"] for point in history if point[key] <= threshold), None
    )


def run(
    problem: OperatorProblem,
    covariances: list[torch.Tensor],
    optimizer_name: str,
    lr: float,
) -> dict:
    parameter = torch.nn.Parameter(torch.zeros_like(problem.target))
    optimizer = _make_optimizer(optimizer_name, parameter, lr)
    initial = float(problem.loss(parameter))
    history = []
    failure = None
    for step, covariance in enumerate(covariances, 1):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = (parameter.detach() - problem.target) @ covariance
        gradient_norm = torch.nn.utils.clip_grad_norm_([parameter], 5.0)
        if not torch.isfinite(gradient_norm):
            failure = {"step": step, "stage": "gradient"}
            break
        optimizer.step()
        value = float(problem.loss(parameter))
        operator_error = problem.relative_operator_error(parameter)
        if not math.isfinite(value) or not math.isfinite(operator_error):
            failure = {"step": step, "stage": "parameters"}
            break
        history.append({
            "step": step,
            "relative_loss": value / initial,
            "operator_error": operator_error,
            **({"optimizer_turn": float(optimizer.last_turn)}
               if hasattr(optimizer, "last_turn") else {}),
            **({"optimizer_lr_scale": float(optimizer.last_lr_scale)}
               if hasattr(optimizer, "last_lr_scale") else {}),
            **({"optimizer_certificate_scale": float(optimizer.last_certificate_scale)}
               if hasattr(optimizer, "last_certificate_scale") else {}),
        })
    final = history[-1] if history else {
        "relative_loss": float("nan"), "operator_error": float("nan")
    }
    return {
        "optimizer": optimizer_name,
        "lr": lr,
        "status": "failed" if failure else "complete",
        "failure": failure,
        "steps": len(covariances),
        "final_relative_loss": final["relative_loss"],
        "final_operator_error": final["operator_error"],
        "steps_to_loss_1e-2": _first(history, "relative_loss", 1e-2),
        "steps_to_loss_1e-4": _first(history, "relative_loss", 1e-4),
        "steps_to_loss_1e-8": _first(history, "relative_loss", 1e-8),
        "steps_to_operator_1e-2": _first(history, "operator_error", 1e-2),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/operator_battery.json"))
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--condition", type=float, default=1e4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=731)
    parser.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    args = parser.parse_args()

    optimizer_names = tuple(
        name.strip() for name in args.optimizers.split(",") if name.strip()
    )
    unknown = set(optimizer_names) - set(OPTIMIZERS)
    if unknown:
        raise ValueError(f"unknown optimizers: {sorted(unknown)}")

    runs = []
    certificates = []
    operator_native_lrs = dict(OPERATOR_NATIVE_LRS)
    matrix_native_lr = 1.0 / math.sqrt(args.dimension)
    operator_native_lrs["matrix_transport"] = matrix_native_lr
    operator_native_lrs["matrix_transport_nesterov"] = matrix_native_lr
    operator_native_lrs["matrix_transport_longitudinal_fusion"] = matrix_native_lr
    protocols = {
        "standing": {name: STANDING_LRS[name] for name in optimizer_names},
        "operator_native": {
            name: operator_native_lrs[name] for name in optimizer_names
        },
    }
    for seed_index in range(args.seeds):
        seed = args.seed_base + seed_index
        exact_problem = make_operator_problem(
            args.dimension, args.condition, seed
        )
        certificates.append({"seed": seed, **one_step_certificate(exact_problem)})
        problem = _float_problem(exact_problem)
        population = [problem.covariance] * args.steps
        block_count = args.dimension // args.batch_size
        block = make_block_sweep_covariances(
            problem, args.batch_size, cycles=math.ceil(args.steps / block_count)
        )[:args.steps]
        for protocol, rates in protocols.items():
            for scenario, covariances in (
                ("population", population),
                ("block_sweep", block),
            ):
                for optimizer_name in optimizer_names:
                    row = run(
                        problem, covariances, optimizer_name,
                        rates[optimizer_name],
                    )
                    row.update({
                        "seed": seed_index,
                        "problem_seed": seed,
                        "protocol": protocol,
                        "scenario": scenario,
                    })
                    runs.append(row)
                    print(json.dumps({
                        key: value for key, value in row.items()
                        if key != "history"
                    }), flush=True)

    payload = {
        "configuration": {**vars(args), "out": str(args.out)},
        "optimizers": list(optimizer_names),
        "protocols": protocols,
        "exact_one_step_certificates": certificates,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
