// Container benchmark for the C++ engine. Reads the scene written by
// `container.mjs --export-text`, runs it, prints per-step timings, and writes
// the common result record that container_report.mjs scores.
//
//   phys_bench scene.txt out.json [hertz] [seconds]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../physics.hpp"

namespace {

using zc::phys::BodyId;
using zc::phys::Vec3;

struct SceneBody {
    zc::phys::Shape shape;
    zc::phys::Pose pose;
};

std::vector<Vec3> box_points(Vec3 size) {
    std::vector<Vec3> points;
    for (int i = 0; i < 8; i += 1) {
        Vec3 p{(i & 1 ? 0.5 : -0.5) * size.x, (i & 2 ? 0.5 : -0.5) * size.y, (i & 4 ? 0.5 : -0.5) * size.z};
        points.push_back(p);
    }
    return points;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    return elapsed.count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: phys_bench scene.txt out.json [hertz] [seconds]\n");
        return 2;
    }
    const int hertz = argc > 3 ? std::atoi(argv[3]) : 60;
    const double seconds = argc > 4 ? std::atof(argv[4]) : 8.0;
    const double friction = 0.5;
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    zc::phys::WorldParams params;
    params.frame_dt = 1.0 / hertz;
    params.linear_damping = 0;
    params.angular_damping = 0;
    params.ground_friction = friction;
    zc::phys::World world(params);
    zc::phys::SolverParams solver;
    solver.sleeping = false;
    solver.ground_rolling_resistance = 0;      // as in the cross-engine comparison
    world.set_solver_params(solver);

    int wall_count = 0;
    in >> wall_count;
    for (int k = 0; k < wall_count; k += 1) {
        Vec3 centre, size;
        in >> centre.x >> centre.y >> centre.z >> size.x >> size.y >> size.z;
        zc::phys::ShapeDesc desc;
        zc::phys::HullDesc hull;
        hull.points = box_points(size);
        desc.hulls.push_back(hull);
        desc.friction = friction;
        desc.restitution = 0;
        desc.rolling_resistance = 0;
        zc::phys::Pose pose;
        pose.p = centre;
        world.add_static_body(zc::phys::cook(desc), pose);
    }
    int body_count = 0;
    in >> body_count;
    std::vector<BodyId> ids;
    std::vector<double> radii;
    double weight = 0;
    for (int i = 0; i < body_count; i += 1) {
        int vertex_count = 0;
        in >> vertex_count;
        zc::phys::ShapeDesc desc;
        zc::phys::HullDesc hull;
        for (int v = 0; v < vertex_count; v += 1) {
            Vec3 p;
            in >> p.x >> p.y >> p.z;
            hull.points.push_back(p);
        }
        desc.hulls.push_back(hull);
        desc.friction = friction;
        desc.restitution = 0;
        desc.rolling_resistance = 0;
        zc::phys::Pose pose;
        in >> pose.p.x >> pose.p.y >> pose.p.z >> pose.q.w >> pose.q.x >> pose.q.y >> pose.q.z;
        const zc::phys::Shape shape = zc::phys::cook(desc);
        weight += shape.mass * 9.81;
        radii.push_back(shape.radius);
        ids.push_back(world.add_body(shape, pose));
    }

    const int frames = static_cast<int>(seconds * hertz + 0.5);
    const int every = hertz / 60;
    std::string samples;
    double total = 0;
    double worst = 0;
    double moving_total = 0;
    int moving_frames = 0;
    double quiet_total = 0;
    int quiet_frames = 0;
    for (int f = 0; f < frames; f += 1) {
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        world.step();
        const double elapsed = seconds_since(start);
        total += elapsed;
        worst = elapsed > worst ? elapsed : worst;
        const zc::phys::RestReport rest = world.rest_report();
        if (rest.max_speed < 0.004) {
            quiet_total += elapsed;
            quiet_frames += 1;
        } else {
            moving_total += elapsed;
            moving_frames += 1;
        }
        if ((f + 1) % every == 0) {
            char line[160];
            std::snprintf(line, sizeof(line), "%s{\"t\":%.6f,\"maxSpeed\":%.9e,\"kineticEnergy\":%.9e}",
                          samples.empty() ? "" : ",", (f + 1) / static_cast<double>(hertz), rest.max_speed,
                          rest.kinetic_energy);
            samples += line;
        }
    }
    // Vertical force the floor and walls return, from the last frame.
    double support = 0;
    double floor_force = 0;
    const std::vector<zc::phys::Contact> contacts = world.contacts();
    for (std::size_t k = 0; k < contacts.size(); k += 1) {
        if (contacts[k].b == zc::phys::kGround) {
            floor_force += contacts[k].normal_impulse * hertz;
        }
    }
    (void)support;
    const zc::phys::SolveStats stats = world.solve_stats();

    std::FILE* out = std::fopen(argv[2], "w");
    if (out == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 2;
    }
    std::fprintf(out, "{\"engine\":\"wrench_transport_cpp\",\"config\":\"wrench@%dHz\",\"dt\":%.12f,\"samples\":[%s],",
                 hertz, 1.0 / hertz, samples.c_str());
    std::fprintf(out, "\"finalPoses\":[");
    for (std::size_t i = 0; i < ids.size(); i += 1) {
        const zc::phys::BodyState state = world.state(ids[i]);
        std::fprintf(out, "%s{\"p\":[%.17g,%.17g,%.17g],\"q\":[%.17g,%.17g,%.17g,%.17g]}", i == 0 ? "" : ",",
                     state.pose.p.x, state.pose.p.y, state.pose.p.z, state.pose.q.w, state.pose.q.x, state.pose.q.y,
                     state.pose.q.z);
    }
    std::fprintf(out, "],\"supportForce\":null,\"floorForce\":%.9f,\"weight\":%.9f,\"wallSeconds\":%.6f,", floor_force,
                 weight, total);
    std::fprintf(out, "\"worstStepMs\":%.6f,\"work\":{\"stepsPerSecond\":%d,\"newtonIterations\":%lld,", worst * 1000,
                 hertz, static_cast<long long>(stats.newton_iterations));
    std::fprintf(out, "\"factorizations\":%lld,\"passes\":%lld}}\n", static_cast<long long>(stats.factorizations),
                 static_cast<long long>(stats.passes));
    std::fclose(out);

    std::printf("wrench_transport_cpp %d Hz: %d frames, mean step %.4f ms, worst %.4f ms\n", hertz, frames,
                1000 * total / frames, 1000 * worst);
    std::printf("  moving frames %d: mean %.4f ms; quiet frames %d: mean %.4f ms\n", moving_frames,
                moving_frames > 0 ? 1000 * moving_total / moving_frames : 0.0, quiet_frames,
                quiet_frames > 0 ? 1000 * quiet_total / quiet_frames : 0.0);
    std::printf("  newton %lld, factorizations %lld, reuses %lld, passes %lld, analyses %lld, unconverged %lld, guards %d\n",
                static_cast<long long>(stats.newton_iterations), static_cast<long long>(stats.factorizations),
                static_cast<long long>(stats.factor_reuses), static_cast<long long>(stats.passes),
                static_cast<long long>(stats.analyses), static_cast<long long>(stats.not_converged), world.guard_count());
    const zc::phys::RestReport rest = world.rest_report();
    std::printf("  final max speed %.3e m/s, kinetic energy %.3e J, floor/weight %.6f\n", rest.max_speed,
                rest.kinetic_energy, floor_force / weight);
    return 0;
}
