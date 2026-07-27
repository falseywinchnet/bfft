import sys
from pathlib import Path

from skimage import data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

from marching_fusion import marching_fusion  # noqa: E402
from receiver_guided_graph import ReceiverGuidedVoronoi, score  # noqa: E402
from transport_voronoi import Config  # noqa: E402


def _model():
    cfg = Config(
        max_side=48, passes=4, flow_sweeps=8,
        initial_cells=30, max_cells=60, split_batch=15,
        allocation_mode="Expected affine gain")
    model = ReceiverGuidedVoronoi(data.camera(), cfg)
    while len(model.seeds) < cfg.max_cells:
        model.step_direct()
    model.solve_direct_coupled()
    return model, cfg


def test_joint_trust_and_soft_fusion_are_measured_and_bounded():
    model, cfg = _model()
    baseline = score(model, cfg)
    trust = model.receiver_trust_step(
        damping=0.05, trust=1.5, joint_softness=True)
    after_trust = score(model, cfg)
    assert trust["evaluations"] <= 2
    assert after_trust["objective"] <= baseline["objective"] + 1e-14

    tolerance = 0.01
    fusion = marching_fusion(
        model, rounds=2, fraction=0.1, quantile=0.5,
        objective_tolerance=tolerance, psnr_tolerance=0.1,
        cartoon_softness=trust["cartoon_softness"],
        texture_softness=trust["texture_softness"])
    assert fusion["groups"] <= fusion["cells"]
    assert (
        fusion["final"]["objective"] <=
        fusion["baseline"]["objective"] * (1.0 + tolerance) + 1e-14)
