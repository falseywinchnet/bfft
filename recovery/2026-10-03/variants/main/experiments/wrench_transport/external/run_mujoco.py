"""Runs the exported container scene in MuJoCo and writes the common result record.

Usage: python3 run_mujoco.py scene.json out.json --steps-per-sample 8 [--cone elliptic]
The step is 1 / (sampleRate * steps_per_sample) seconds.
"""

import argparse
import json
import time

import mujoco
import numpy as np


def build_xml(scene, timestep, cone, multiccd, solref):
    friction = scene["friction"]
    parts = []
    parts.append('<mujoco model="container">')
    parts.append('<option timestep="%.12g" gravity="0 0 -9.81" cone="%s">' % (timestep, cone))
    if multiccd:
        # Several contact points per convex pair instead of one.
        parts.append('<flag multiccd="enable"/>')
    parts.append('</option>')
    if solref > 0:
        # Contact time constant in seconds (the default is 0.02); it may not
        # be below twice the step.
        parts.append('<default><geom solref="%.9g 1"/></default>' % solref)
    parts.append('<asset>')
    for index, body in enumerate(scene["bodies"]):
        vertices = " ".join("%.9g" % value for value in body["vertices"])
        parts.append('<mesh name="m%d" vertex="%s"/>' % (index, vertices))
    parts.append('</asset>')
    parts.append('<worldbody>')
    parts.append('<geom name="floor" type="plane" size="5 5 0.1" friction="%g 0 0" condim="3"/>' % friction)
    for index, wall in enumerate(scene["walls"]):
        centre = wall["centre"]
        size = wall["size"]
        parts.append('<geom name="wall%d" type="box" pos="%g %g %g" size="%g %g %g" friction="%g 0 0" condim="3"/>' % (
            index, centre[0], centre[1], centre[2], 0.5 * size[0], 0.5 * size[1], 0.5 * size[2], friction))
    for index, body in enumerate(scene["bodies"]):
        p = body["position"]
        q = body["quaternion"]
        parts.append('<body name="b%d" pos="%.12g %.12g %.12g" quat="%.12g %.12g %.12g %.12g">' % (
            index, p[0], p[1], p[2], q[0], q[1], q[2], q[3]))
        parts.append('<freejoint/>')
        parts.append('<geom type="mesh" mesh="m%d" density="%g" friction="%g 0 0" condim="3"/>' % (
            index, scene["density"], friction))
        parts.append('</body>')
    parts.append('</worldbody>')
    parts.append('</mujoco>')
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("--steps-per-sample", type=int, default=8)
    parser.add_argument("--cone", default="elliptic")
    parser.add_argument("--multiccd", action="store_true")
    parser.add_argument("--solref", type=float, default=0.0)
    arguments = parser.parse_args()

    with open(arguments.scene) as handle:
        scene = json.load(handle)
    timestep = 1.0 / (scene["sampleRate"] * arguments.steps_per_sample)
    model = mujoco.MjModel.from_xml_string(
        build_xml(scene, timestep, arguments.cone, arguments.multiccd, arguments.solref))
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    data = mujoco.MjData(model)

    body_count = len(scene["bodies"])
    radii = np.array([body["radius"] for body in scene["bodies"]])
    static_geoms = set()
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    static_geoms.add(floor_geom)
    for index in range(len(scene["walls"])):
        static_geoms.add(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall%d" % index))

    sample_count = int(round(scene["seconds"] * scene["sampleRate"]))
    samples = []
    solver_iterations = 0
    contact_sum = 0
    step_count = 0
    wall_seconds = 0.0
    worst_step = 0.0
    for sample in range(sample_count):
        for _ in range(arguments.steps_per_sample):
            started = time.perf_counter()
            mujoco.mj_step(model, data)
            elapsed = time.perf_counter() - started
            wall_seconds += elapsed
            worst_step = max(worst_step, elapsed)
            solver_iterations += int(data.solver_niter[0])
            contact_sum += int(data.ncon)
            step_count += 1
        velocity = data.qvel.reshape(body_count, 6)
        speed = np.linalg.norm(velocity[:, 0:3], axis=1) + np.linalg.norm(velocity[:, 3:6], axis=1) * radii
        samples.append({
            "t": (sample + 1) / scene["sampleRate"],
            "maxSpeed": float(speed.max()),
            "kineticEnergy": float(data.energy[1]),
        })

    # Vertical force from the floor and the walls on the bodies.
    support = 0.0
    floor_force = 0.0
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        first_static = contact.geom1 in static_geoms
        second_static = contact.geom2 in static_geoms
        if not first_static and not second_static:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        frame = np.array(contact.frame).reshape(3, 3)
        world_force = wrench[0] * frame[0] + wrench[1] * frame[1] + wrench[2] * frame[2]
        # The force is expressed as acting from geom1 on geom2.
        vertical = world_force[2] if first_static else -world_force[2]
        support += vertical
        if contact.geom1 == floor_geom or contact.geom2 == floor_geom:
            floor_force += vertical

    final_poses = []
    for index in range(body_count):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b%d" % index)
        final_poses.append({"p": [float(v) for v in data.xpos[body_id]], "q": [float(v) for v in data.xquat[body_id]]})

    weight = float(sum(model.body_mass[1:]) * 9.81)
    result = {
        "engine": "mujoco " + mujoco.__version__,
        "config": "newton, %s cone, %d Hz%s%s" % (
            arguments.cone, scene["sampleRate"] * arguments.steps_per_sample,
            ", multiccd" if arguments.multiccd else "",
            ", solref %g s" % arguments.solref if arguments.solref > 0 else ""),
        "dt": timestep,
        "samples": samples,
        "finalPoses": final_poses,
        "supportForce": support,
        "floorForce": floor_force,
        "weight": weight,
        "wallSeconds": wall_seconds,
        "worstStepMs": worst_step * 1000.0,
        "work": {
            "stepsPerSecond": scene["sampleRate"] * arguments.steps_per_sample,
            "solverIterations": solver_iterations,
            "meanContacts": contact_sum / max(1, step_count),
        },
    }
    with open(arguments.out, "w") as handle:
        json.dump(result, handle)
    print(result["engine"], result["config"], "wall %.2f s" % wall_seconds,
          "final max speed %.2e" % samples[-1]["maxSpeed"], "support/weight %.6f" % (support / weight))


if __name__ == "__main__":
    main()
