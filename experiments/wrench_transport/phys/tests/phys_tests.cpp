// Acceptance tests for zc::phys: the specification's section 11 (sixteen tests)
// and the feature checks proven first in the JavaScript prototype (static
// bodies, restitution, the crane hold, the rest report, the stable set).
// Each test prints its measured numbers and PASS or FAIL; the program exits
// non-zero on any failure. Run one test by name: phys_tests rest.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../physics.hpp"

namespace {

using zc::phys::BodyId;
using zc::phys::BodyState;
using zc::phys::Contact;
using zc::phys::HoldParams;
using zc::phys::HoldState;
using zc::phys::HullDesc;
using zc::phys::Pose;
using zc::phys::Quat;
using zc::phys::Shape;
using zc::phys::ShapeDesc;
using zc::phys::SolverParams;
using zc::phys::Vec3;
using zc::phys::World;
using zc::phys::WorldParams;

constexpr double kPi = 3.14159265358979323846;
constexpr double kFrame = 1.0 / 60.0;

int failures = 0;

void report(const char* name, bool pass, const std::string& detail) {
    if (!pass) {
        failures += 1;
    }
    std::printf("%s  %s  %s\n", pass ? "PASS" : "FAIL", name, detail.c_str());
    std::fflush(stdout);
}

std::string format(const char* pattern, double a) {
    char buffer[256];
    std::snprintf(buffer, sizeof buffer, pattern, a);
    return std::string(buffer);
}

std::string format(const char* pattern, double a, double b) {
    char buffer[256];
    std::snprintf(buffer, sizeof buffer, pattern, a, b);
    return std::string(buffer);
}

std::string format(const char* pattern, double a, double b, double c) {
    char buffer[256];
    std::snprintf(buffer, sizeof buffer, pattern, a, b, c);
    return std::string(buffer);
}

// ---------------------------------------------------------------- test shapes

// mulberry32, as in the prototype's shapes.mjs, so seeds give the same rocks.
struct Random {
    std::uint32_t state = 0;
};

Random make_random(std::uint32_t seed) {
    Random random;
    random.state = seed;
    return random;
}

double random_unit(Random& random) {
    random.state = random.state + 0x6d2b79f5u;
    std::uint32_t t = random.state;
    t = (t ^ (t >> 15)) * (t | 1u);
    t = t ^ (t + (t ^ (t >> 7)) * (t | 61u));
    const std::uint32_t mixed = t ^ (t >> 14);
    return static_cast<double>(mixed) / 4294967296.0;
}

double random_range(Random& random, double low, double high) {
    const double u = random_unit(random);
    return low + (high - low) * u;
}

enum class RockKind { round, flat, jagged };

std::vector<Vec3> box_points(double sx, double sy, double sz) {
    std::vector<Vec3> points;
    for (int i = 0; i < 8; i += 1) {
        const double x = ((i & 1) != 0 ? 0.5 : -0.5) * sx;
        const double y = ((i & 2) != 0 ? 0.5 : -0.5) * sy;
        const double z = ((i & 4) != 0 ? 0.5 : -0.5) * sz;
        points.push_back(Vec3{x, y, z});
    }
    return points;
}

struct Material {
    double density = 2600;
    double friction = 0.75;
    double restitution = 0.05;
    double rolling_resistance = 0.002;
};

Shape cook_points(const std::vector<std::vector<Vec3>>& hulls, const Material& material) {
    ShapeDesc desc;
    for (size_t h = 0; h < hulls.size(); h += 1) {
        HullDesc hull;
        hull.points = hulls[h];
        desc.hulls.push_back(hull);
    }
    desc.density = material.density;
    desc.friction = material.friction;
    desc.restitution = material.restitution;
    desc.rolling_resistance = material.rolling_resistance;
    return zc::phys::cook(desc);
}

Shape box_shape(double sx, double sy, double sz, const Material& material) {
    std::vector<std::vector<Vec3>> hulls;
    hulls.push_back(box_points(sx, sy, sz));
    return cook_points(hulls, material);
}

Shape box_shape(double sx, double sy, double sz) {
    const Material material;
    return box_shape(sx, sy, sz, material);
}

// 32 points on a deformed ellipsoid, as in the specification's section 11.
std::vector<Vec3> rock_points(std::uint32_t seed, RockKind kind) {
    Random random = make_random(seed * 7919u + 17u);
    double ax = 0;
    double ay = 0;
    double az = 0;
    double noise = 0.15;
    if (kind == RockKind::round) {
        ax = random_range(random, 0.06, 0.09);
        ay = random_range(random, 0.06, 0.09);
        az = random_range(random, 0.06, 0.09);
    } else if (kind == RockKind::flat) {
        ax = random_range(random, 0.10, 0.20);
        ay = random_range(random, 0.08, 0.15);
        az = random_range(random, 0.02, 0.04);
    } else {
        ax = random_range(random, 0.03, 0.06);
        ay = random_range(random, 0.03, 0.06);
        az = random_range(random, 0.03, 0.06);
        noise = 0.3;
    }
    std::vector<Vec3> points;
    const double turn = random_range(random, 0, 2 * kPi);
    const double golden = kPi * (3 - std::sqrt(5.0));
    for (int i = 0; i < 32; i += 1) {
        const double z_unit = 1 - (2.0 * i + 1) / 32.0;
        const double angle = turn + golden * i;
        const double ring = std::sqrt(std::max(0.0, 1 - z_unit * z_unit));
        const double stretch = 1 + noise * random_range(random, -1, 1);
        double z = 0.5 * az * z_unit * stretch;
        if (kind == RockKind::flat) {
            // a flat stone's top and bottom are cut level at 60% of its half thickness
            const double cut = 0.3 * az;
            z = std::min(cut, std::max(-cut, z));
        }
        const double x = 0.5 * ax * ring * std::cos(angle) * stretch;
        const double y = 0.5 * ay * ring * std::sin(angle) * stretch;
        points.push_back(Vec3{x, y, z});
    }
    return points;
}

Shape rock_shape(std::uint32_t seed, RockKind kind) {
    std::vector<std::vector<Vec3>> hulls;
    hulls.push_back(rock_points(seed, kind));
    const Material material;
    return cook_points(hulls, material);
}

Shape concave_rock_shape(std::uint32_t seed) {
    Random random = make_random(seed * 104729u + 5u);
    std::vector<std::vector<Vec3>> hulls;
    for (std::uint32_t part = 0; part < 3; part += 1) {
        std::vector<Vec3> points = rock_points(seed * 3u + part, RockKind::round);
        const double sx = random_range(random, -0.03, 0.03);
        const double sy = random_range(random, -0.03, 0.03);
        const double sz = random_range(random, -0.015, 0.015);
        for (size_t i = 0; i < points.size(); i += 1) {
            points[i].x = points[i].x + sx;
            points[i].y = points[i].y + sy;
            points[i].z = points[i].z + sz;
        }
        hulls.push_back(points);
    }
    const Material material;
    return cook_points(hulls, material);
}

// ---------------------------------------------------------------- helpers

Pose pose_at(double x, double y, double z) {
    Pose pose;
    pose.p = Vec3{x, y, z};
    return pose;
}

void run(World& world, int frames) {
    for (int f = 0; f < frames; f += 1) {
        world.step();
    }
}

double distance(Vec3 a, Vec3 b) {
    const Vec3 d = a - b;
    return zc::phys::length(d);
}

// the angle between two orientations, degrees
double angle_between(Quat a, Quat b) {
    const double overlap = std::abs(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z);
    const double clamped = std::min(1.0, overlap);
    return 2 * std::acos(clamped) * 180 / kPi;
}

// world-space extremes of a body's hull vertices
double lowest_z(const Shape& shape, const Pose& pose) {
    double low = 1e9;
    for (size_t h = 0; h < shape.hull_vertices.size(); h += 1) {
        const std::vector<Vec3>& vertices = shape.hull_vertices[h];
        for (size_t i = 0; i < vertices.size(); i += 1) {
            const Vec3 w = zc::phys::rotate(pose.q, vertices[i]) + pose.p;
            low = std::min(low, w.z);
        }
    }
    return low;
}

double highest_z(const Shape& shape, const Pose& pose) {
    double high = -1e9;
    for (size_t h = 0; h < shape.hull_vertices.size(); h += 1) {
        const std::vector<Vec3>& vertices = shape.hull_vertices[h];
        for (size_t i = 0; i < vertices.size(); i += 1) {
            const Vec3 w = zc::phys::rotate(pose.q, vertices[i]) + pose.p;
            high = std::max(high, w.z);
        }
    }
    return high;
}

struct Snapshot {
    std::vector<BodyId> ids;
    std::vector<Pose> poses;
};

Snapshot snapshot(const World& world, const std::vector<BodyId>& ids) {
    Snapshot snap;
    for (size_t i = 0; i < ids.size(); i += 1) {
        snap.ids.push_back(ids[i]);
        snap.poses.push_back(world.state(ids[i]).pose);
    }
    return snap;
}

double max_drift(const World& world, const Snapshot& snap) {
    double worst = 0;
    for (size_t i = 0; i < snap.ids.size(); i += 1) {
        const BodyState state = world.state(snap.ids[i]);
        worst = std::max(worst, distance(state.pose.p, snap.poses[i].p));
    }
    return worst;
}

double max_rotation(const World& world, const Snapshot& snap) {
    double worst = 0;
    for (size_t i = 0; i < snap.ids.size(); i += 1) {
        const BodyState state = world.state(snap.ids[i]);
        worst = std::max(worst, angle_between(state.pose.q, snap.poses[i].q));
    }
    return worst;
}

// the deepest overlap among the last step's contacts (positive metres)
double deepest_overlap(const World& world) {
    const std::vector<Contact> contacts = world.contacts();
    double deepest = 0;
    for (size_t i = 0; i < contacts.size(); i += 1) {
        deepest = std::max(deepest, -contacts[i].separation);
    }
    return deepest;
}

bool all_listed_asleep(const World& world, const std::vector<BodyId>& ids) {
    for (size_t i = 0; i < ids.size(); i += 1) {
        if (!world.state(ids[i]).asleep) {
            return false;
        }
    }
    return true;
}

// steps until every listed body sleeps; returns the seconds taken, or -1
double run_until_asleep(World& world, const std::vector<BodyId>& ids, double limit_seconds) {
    const int limit = static_cast<int>(limit_seconds / kFrame);
    for (int f = 0; f < limit; f += 1) {
        world.step();
        if (all_listed_asleep(world, ids)) {
            return (f + 1) * kFrame;
        }
    }
    return -1;
}

// Sets a rock on top of the pile: its centre of mass over (x, y), its lowest
// point `gap` above the highest point of `below` (or the ground).
BodyId place_on(World& world, const Shape& shape, const Shape* below_shape, BodyId below, double x, double y, double gap) {
    double top = 0;
    if (below_shape != nullptr) {
        const Pose below_pose = world.state(below).pose;
        top = highest_z(*below_shape, below_pose);
    }
    Pose pose = pose_at(x, y, 0);
    const double low = lowest_z(shape, pose);
    pose.p.z = top + gap - low;
    return world.add_body(shape, pose);
}

struct Stack {
    std::vector<BodyId> ids;
    std::vector<Shape> shapes;   // owned here; each body keeps a reference to its shape for the world's lifetime
};

// A stack of `count` rocks of one kind, each set with its centre of mass over
// the one below and dropped from 2 mm, settling before the next.
void build_rock_stack(World& world, Stack& stack, int count, std::uint32_t seed, RockKind kind, bool concave) {
    stack.shapes.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; i += 1) {
        const std::uint32_t s = seed * 101u + static_cast<std::uint32_t>(i);
        if (concave) {
            stack.shapes.push_back(concave_rock_shape(s));
        } else {
            stack.shapes.push_back(rock_shape(s, kind));
        }
    }
    for (int i = 0; i < count; i += 1) {
        const Shape* below_shape = nullptr;
        BodyId below = 0;
        double x = 0;
        double y = 0;
        if (i > 0) {
            below_shape = &stack.shapes[static_cast<size_t>(i - 1)];
            below = stack.ids[static_cast<size_t>(i - 1)];
            const Pose below_pose = world.state(below).pose;
            x = below_pose.p.x;
            y = below_pose.p.y;
        }
        const BodyId id = place_on(world, stack.shapes[static_cast<size_t>(i)], below_shape, below, x, y, 0.002);
        stack.ids.push_back(id);
        run(world, 90);
    }
}

void build_box_stack(World& world, Stack& stack, int count) {
    stack.shapes.push_back(box_shape(0.12, 0.1, 0.04));
    for (int i = 0; i < count; i += 1) {
        const double z = 0.02 + 0.04 * i + 0.0002 * (i + 1);
        stack.ids.push_back(world.add_body(stack.shapes[0], pose_at(0, 0, z)));
    }
    run(world, 120);
}

const char* kind_name(RockKind kind) {
    if (kind == RockKind::round) {
        return "round";
    }
    if (kind == RockKind::flat) {
        return "flat";
    }
    return "jagged";
}

// ---------------------------------------------------------------- 1. rest

void test_rest() {
    const RockKind kinds[3] = {RockKind::round, RockKind::flat, RockKind::jagged};
    for (int k = 0; k < 3; k += 1) {
        double worst_drift = 0;
        double worst_rotation = 0;
        double worst_sleep = 0;
        double worst_overlap = 0;
        bool all_slept = true;
        for (std::uint32_t seed = 1; seed <= 50; seed += 1) {
            World world;
            const Shape shape = rock_shape(seed, kinds[k]);
            const BodyId id = place_on(world, shape, nullptr, 0, 0, 0, 0.01);
            std::vector<BodyId> ids;
            ids.push_back(id);
            const double slept = run_until_asleep(world, ids, 3.0);
            if (slept < 0) {
                all_slept = false;
            }
            worst_sleep = std::max(worst_sleep, slept);
            run(world, 2);
            worst_overlap = std::max(worst_overlap, deepest_overlap(world));
            const Snapshot snap = snapshot(world, ids);
            run(world, 600 * 60);
            worst_drift = std::max(worst_drift, max_drift(world, snap));
            worst_rotation = std::max(worst_rotation, max_rotation(world, snap));
        }
        const bool pass = all_slept && worst_drift < 5e-5 && worst_rotation < 0.005 && worst_overlap < 0.001;
        std::string name = std::string("1 rest: ") + kind_name(kinds[k]) + " rocks, 50 seeds, 600 s";
        report(name.c_str(), pass,
               format("asleep by %.2f s; drift %.2e mm, rotation %.2e deg", worst_sleep, worst_drift * 1000, worst_rotation) +
                   format("; overlap %.3f mm", worst_overlap * 1000));
    }
}

// ---------------------------------------------------------------- 2. tall stack

struct StackOutcome {
    double sleep_seconds = 0;
    double drift = 0;
    double max_ke_asleep = 0;
    bool slept = false;
};

StackOutcome tall_stack(std::uint32_t seed, int count, bool concave, double simulate_seconds) {
    StackOutcome outcome;
    World world;
    Stack stack;
    build_rock_stack(world, stack, count, seed, RockKind::flat, concave);
    const double slept = run_until_asleep(world, stack.ids, 5.0);
    outcome.slept = slept >= 0;
    outcome.sleep_seconds = slept;
    const Snapshot snap = snapshot(world, stack.ids);
    const int frames = static_cast<int>(simulate_seconds / kFrame);
    for (int f = 0; f < frames; f += 1) {
        world.step();
        if (world.all_asleep()) {
            outcome.max_ke_asleep = std::max(outcome.max_ke_asleep, world.kinetic_energy());
        }
    }
    outcome.drift = max_drift(world, snap);
    return outcome;
}

void test_tall_stack() {
    double worst_drift = 0;
    double worst_sleep = 0;
    double worst_ke = 0;
    int unslept = 0;
    for (std::uint32_t seed = 1; seed <= 20; seed += 1) {
        const StackOutcome o = tall_stack(seed, 15, false, 600);
        worst_drift = std::max(worst_drift, o.drift);
        worst_sleep = std::max(worst_sleep, o.sleep_seconds);
        worst_ke = std::max(worst_ke, o.max_ke_asleep);
        if (!o.slept) {
            unslept += 1;
        }
    }
    const bool pass = unslept == 0 && worst_drift < 3e-4 && worst_ke == 0;
    report("2 tall stack: 15 flat rocks, 20 seeds, 600 s", pass,
           format("asleep within %.2f s (unslept %.0f); drift %.3f mm", worst_sleep, unslept, worst_drift * 1000) +
               format("; KE while asleep %.1e J", worst_ke));
}

// ---------------------------------------------------------------- 3. mass ratio

void test_mass_ratio() {
    // a 10 kg slab on three 0.02 kg pebbles
    {
        World world;
        const double pebble_side = std::cbrt(0.02 / 2600.0);
        const Shape pebble = box_shape(pebble_side, pebble_side, pebble_side);
        const Shape slab = box_shape(0.3, 0.2, 10.0 / (2600.0 * 0.06));
        std::vector<BodyId> pebbles;
        const double spots[3][2] = {{-0.1, -0.06}, {0.1, -0.06}, {0.0, 0.07}};
        for (int i = 0; i < 3; i += 1) {
            pebbles.push_back(world.add_body(pebble, pose_at(spots[i][0], spots[i][1], 0.5 * pebble_side)));
        }
        run(world, 30);
        const double slab_half = 0.5 * 10.0 / (2600.0 * 0.06);
        const BodyId top = world.add_body(slab, pose_at(0, 0, pebble_side + slab_half + 0.001));
        run(world, 120);
        double fastest = 0;
        double deepest = 0;
        for (int f = 0; f < 120 * 60; f += 1) {
            world.step();
            for (size_t i = 0; i < pebbles.size(); i += 1) {
                const BodyState s = world.state(pebbles[i]);
                fastest = std::max(fastest, zc::phys::length(s.v));
            }
            if (f % 60 == 0) {
                deepest = std::max(deepest, deepest_overlap(world));
            }
        }
        const double slab_z = world.state(top).pose.p.z;
        const bool standing = slab_z > pebble_side + slab_half - 0.002;
        const bool pass = standing && fastest < 0.001 && deepest < 0.001;
        report("3 mass ratio: 10 kg slab on three 0.02 kg pebbles", pass,
               format("slab height error %.3f mm; pebbles' max speed %.2e mm/s; overlap %.3f mm",
                      (slab_z - pebble_side - slab_half) * 1000, fastest * 1000, deepest * 1000));
    }
    // a 0.02 kg pebble on a 10 kg slab
    {
        World world;
        const double pebble_side = std::cbrt(0.02 / 2600.0);
        const Shape pebble = box_shape(pebble_side, pebble_side, pebble_side);
        const double thickness = 10.0 / (2600.0 * 0.06);
        const Shape slab = box_shape(0.3, 0.2, thickness);
        world.add_body(slab, pose_at(0, 0, 0.5 * thickness));
        run(world, 30);
        const BodyId p = world.add_body(pebble, pose_at(0, 0, thickness + 0.5 * pebble_side + 0.001));
        run(world, 120);
        double fastest = 0;
        for (int f = 0; f < 120 * 60; f += 1) {
            world.step();
            fastest = std::max(fastest, zc::phys::length(world.state(p).v));
        }
        const double deepest = deepest_overlap(world);
        const bool pass = fastest < 0.001 && deepest < 0.001;
        report("3 mass ratio: 0.02 kg pebble on a 10 kg slab", pass,
               format("pebble max speed %.2e mm/s; overlap %.3f mm", fastest * 1000, deepest * 1000));
    }
}

// ---------------------------------------------------------------- 4. overhang

// A 20 x 5 x 2 cm slab on a 10 cm cube, its centre of mass `past` metres
// beyond the cube's edge (negative: inside). Returns its rotation after `seconds`.
double overhang_rotation(double past, double seconds) {
    World world;
    const Shape cube = box_shape(0.1, 0.1, 0.1);
    const Shape slab = box_shape(0.2, 0.05, 0.02);
    world.add_body(cube, pose_at(0, 0, 0.05));
    run(world, 30);
    const BodyId s = world.add_body(slab, pose_at(0.05 + past, 0, 0.1 + 0.01 + 0.0002));
    const Quat start = world.state(s).pose.q;
    run(world, static_cast<int>(seconds / kFrame));
    return angle_between(world.state(s).pose.q, start);
}

void test_overhang() {
    const double length = 0.2;
    const double fractions[3] = {0.01, 0.02, 0.05};
    bool pass = true;
    std::string detail;
    for (int i = 0; i < 3; i += 1) {
        const double inside = overhang_rotation(-fractions[i] * length, 60);
        const double tips = overhang_rotation(fractions[i] * length, 3);
        detail += format("%.0f%%: inside %.3f deg, outside %.1f deg; ", fractions[i] * 100, inside, tips);
        if (i > 0) {
            if (inside >= 0.1 || tips <= 10) {
                pass = false;
            }
        }
    }
    report("4 overhang: 2% and 5% inside stay, outside tip (1% may go either way)", pass, detail);
}

// ---------------------------------------------------------------- 5. incline

// gravity tilted by theta about y: a slope down +x
WorldParams tilted(double theta) {
    WorldParams params;
    params.gravity = Vec3{9.81 * std::sin(theta), 0, -9.81 * std::cos(theta)};
    params.ground_friction = 0.75;
    return params;
}

// How far a 10 cm box slides (relative to what it rests on) in `seconds`.
double incline_slide(double tan_theta, bool on_slab, double seconds) {
    const double theta = std::atan(tan_theta);
    World world(tilted(theta));
    Material material;
    material.rolling_resistance = 0;
    const Shape box = box_shape(0.1, 0.1, 0.1, material);
    BodyId base = zc::phys::kGround;
    double floor = 0;
    if (on_slab) {
        Material heavy = material;
        heavy.density = 100.0 / (1.0 * 1.0 * 0.1);
        const Shape slab = box_shape(1.0, 1.0, 0.1, heavy);
        // a static slab is the incline (a free one slides with the box riding on it, as it should)
        base = world.add_static_body(slab, pose_at(0, 0, 0.05));
        floor = 0.1;
    }
    const BodyId b = world.add_body(box, pose_at(0, 0, floor + 0.05 + 0.0002));
    run(world, 2);
    Vec3 start = world.state(b).pose.p;
    Vec3 base_start;
    if (base != zc::phys::kGround) {
        base_start = world.state(base).pose.p;
    }
    run(world, static_cast<int>(seconds / kFrame));
    Vec3 moved = world.state(b).pose.p - start;
    if (base != zc::phys::kGround) {
        const Vec3 base_moved = world.state(base).pose.p - base_start;
        moved = moved - base_moved;
    }
    return std::sqrt(moved.x * moved.x + moved.y * moved.y);
}

void test_incline() {
    const double mu = 0.75;
    const double creep_ground = incline_slide(mu - 0.05, false, 30);
    const double slide_ground = incline_slide(mu + 0.05, false, 3);
    const double creep_slab = incline_slide(mu - 0.05, true, 30);
    const double slide_slab = incline_slide(mu + 0.05, true, 3);
    const bool pass = creep_ground < 5e-4 && slide_ground > 0.1 && creep_slab < 5e-4 && slide_slab > 0.1;
    report("5 incline: holds at tan = mu - 0.05, slides at mu + 0.05", pass,
           format("ground: creep %.3f mm, slide %.0f mm; ", creep_ground * 1000, slide_ground * 1000) +
               format("on a slab: creep %.3f mm, slide %.0f mm", creep_slab * 1000, slide_slab * 1000));
}

// ---------------------------------------------------------------- 6. drop

struct DropOutcome {
    double bounce = 0;
    double speed_gain = 0;   // fastest speed after first contact, minus the impact speed
    double sleep_seconds = -1;
};

DropOutcome drop_rock(const Shape& rock, bool onto_flat) {
    DropOutcome outcome;
    World world;
    double floor = 0;
    std::vector<Shape> keep;
    keep.reserve(1);
    std::vector<BodyId> ids;
    if (onto_flat) {
        keep.push_back(rock_shape(4, RockKind::flat));
        const BodyId flat = place_on(world, keep[0], nullptr, 0, 0, 0, 0.001);
        run(world, 120);
        floor = highest_z(keep[0], world.state(flat).pose);
        ids.push_back(flat);
    }
    Pose pose = pose_at(0, 0, 0);
    pose.p.z = floor + 0.3 - lowest_z(rock, pose);
    const BodyId r = world.add_body(rock, pose);
    ids.push_back(r);
    bool touched = false;
    double impact_speed = 0;
    double after_peak = 0;
    double rise_peak = 0;
    double touch_z = 0;
    for (int f = 0; f < 180; f += 1) {
        const double before_speed = zc::phys::length(world.state(r).v);
        world.step();
        const BodyState s = world.state(r);
        const double bottom = lowest_z(rock, s.pose);
        if (!touched && bottom < floor + 0.003) {
            touched = true;
            impact_speed = before_speed;
            touch_z = s.pose.p.z;
        }
        if (touched) {
            after_peak = std::max(after_peak, zc::phys::length(s.v));
            rise_peak = std::max(rise_peak, s.pose.p.z - touch_z);
        }
    }
    outcome.bounce = rise_peak;
    outcome.speed_gain = after_peak - impact_speed;
    outcome.sleep_seconds = run_until_asleep(world, ids, 3.0);
    return outcome;
}

double drop_slab_final_z(bool onto_slab) {
    World world;
    const Shape slab = box_shape(0.15, 0.1, 0.01);
    double floor = 0;
    if (onto_slab) {
        world.add_body(slab, pose_at(0, 0, 0.005));
        run(world, 30);
        floor = 0.01;
    }
    const BodyId s = world.add_body(slab, pose_at(0.01, 0, floor + 0.5));
    run(world, 180);
    return world.state(s).pose.p.z - floor;
}

void test_drop() {
    double worst_bounce = 0;
    double worst_gain = 0;
    double worst_sleep = 0;
    bool slept = true;
    for (std::uint32_t seed = 1; seed <= 10; seed += 1) {
        const Shape rock = rock_shape(seed, RockKind::round);
        for (int onto = 0; onto < 2; onto += 1) {
            const DropOutcome o = drop_rock(rock, onto == 1);
            worst_bounce = std::max(worst_bounce, o.bounce);
            worst_gain = std::max(worst_gain, o.speed_gain);
            worst_sleep = std::max(worst_sleep, o.sleep_seconds);
            if (o.sleep_seconds < 0) {
                slept = false;
            }
        }
    }
    const double on_ground = drop_slab_final_z(false);
    const double on_slab = drop_slab_final_z(true);
    const bool no_tunnel = on_ground > 0.004 && on_slab > 0.004;
    const bool pass = worst_bounce < 0.005 && worst_gain <= 1e-6 && slept && no_tunnel;
    report("6 drop: round rocks from 0.3 m, slabs from 0.5 m", pass,
           format("bounce %.2f mm; speed gain %.2e m/s; asleep by %.2f s; ", worst_bounce * 1000, worst_gain, worst_sleep) +
               format("1 cm slab rests at %.1f mm on the ground and %.1f mm on a slab", on_ground * 1000, on_slab * 1000));
}

// ---------------------------------------------------------------- 7. hold

void test_hold() {
    World world;
    const Shape shape = rock_shape(5, RockKind::round);
    const BodyId id = world.add_body(shape, pose_at(0, 0, 0.3));
    Pose target = pose_at(0, 0, 0.3);
    world.hold(id, target);
    run(world, 120);
    const double error = distance(world.state(id).pose.p, target.p);
    const double tension = world.hold_state().tension;
    report("7 hold: hangs on target", error < 2e-4 && std::abs(tension - 1) < 0.01,
           format("error %.4f mm; tension %.4f", error * 1000, tension));
    for (int f = 0; f < 120; f += 1) {
        target.p.x = 0.1 * (f + 1) / 120.0;
        world.set_hold_target(target);
        world.step();
    }
    double overshoot = 0;
    for (int f = 0; f < 120; f += 1) {
        world.step();
        overshoot = std::max(overshoot, world.state(id).pose.p.x - 0.1);
    }
    const double final_x = world.state(id).pose.p.x;
    report("7 hold: moves 10 cm without overshoot", overshoot < 0.002 && std::abs(final_x - 0.1) < 2e-4,
           format("overshoot %.3f mm; final x %.5f m", overshoot * 1000, final_x));
    for (int f = 0; f < 120; f += 1) {
        const double angle = 0.5 * kPi * (f + 1) / 120.0;
        target.q = zc::phys::from_axis_angle(Vec3{0, 0, 1}, angle);
        world.set_hold_target(target);
        world.step();
    }
    run(world, 120);
    const double turn_error = angle_between(world.state(id).pose.q, target.q);
    report("7 hold: turns 90 degrees", turn_error < 0.12, format("error %.4f deg", turn_error));
}

// ---------------------------------------------------------------- 8. gentle set-down, 9. shove cap

void test_set_down() {
    World world;
    Stack stack;
    build_rock_stack(world, stack, 5, 7, RockKind::flat, false);
    run_until_asleep(world, stack.ids, 5.0);
    const Snapshot before = snapshot(world, stack.ids);
    const Shape held_shape = box_shape(0.1, 0.1, 2.0 / (2600.0 * 0.01));   // 2 kg
    const BodyId top_id = stack.ids.back();
    const Pose top_pose = world.state(top_id).pose;
    const double top = highest_z(stack.shapes.back(), top_pose);
    const double half = 0.5 * 2.0 / (2600.0 * 0.01);
    Pose target = pose_at(top_pose.p.x, top_pose.p.y, top + half + 0.03);
    const BodyId id = world.add_body(held_shape, target);
    world.hold(id, target);
    run(world, 60);
    double previous = world.hold_state().tension;
    bool monotone = true;
    int lowered = 0;
    double stack_moved_while_held = 0;
    while (lowered < 900 && world.hold_state().tension >= 0.05) {
        target.p.z = target.p.z - 0.02 / 60.0;
        world.set_hold_target(target);
        world.step();
        const double tension = world.hold_state().tension;
        if (tension > previous + 0.02) {
            monotone = false;
        }
        previous = tension;
        lowered += 1;
        stack_moved_while_held = std::max(stack_moved_while_held, max_drift(world, before));
    }
    run(world, 300);
    stack_moved_while_held = std::max(stack_moved_while_held, max_drift(world, before));
    report("8 set-down: tension falls monotonically to slack; stack still", monotone && world.hold_state().tension < 0.05 && stack_moved_while_held < 3e-4,
           format("tension %.3f after %.2f s; stack moved %.4f mm", world.hold_state().tension, lowered / 60.0, stack_moved_while_held * 1000));
    world.release();
    std::vector<BodyId> all = stack.ids;
    all.push_back(id);
    const double slept = run_until_asleep(world, all, 5.0);
    const double moved = max_drift(world, before);
    report("8 set-down: after release everything settles", slept >= 0 && moved < 5e-4,
           format("asleep after %.2f s; old rocks moved %.4f mm", slept, moved * 1000));
}

void test_shove() {
    World world;
    Stack stack;
    build_box_stack(world, stack, 5);
    const Shape held_shape = box_shape(0.1, 0.1, 0.0769);
    Pose target = pose_at(-0.2, 0, 0.18);
    const BodyId id = world.add_body(held_shape, target);
    world.hold(id, target);
    run(world, 60);
    double worst = 0;
    for (int f = 0; f < 240; f += 1) {
        const double t = std::min(1.0, (f + 1) / 180.0);
        target.p.x = -0.2 + 0.15 * t;
        world.set_hold_target(target);
        world.step();
        worst = std::max(worst, world.hold_state().lateral_force);
    }
    report("9 shove cap: side force at most 0.35 of the held weight", worst <= 0.35 * 1.05 && worst > 0.3,
           format("largest side force %.4f m g", worst));
}

// ---------------------------------------------------------------- 10. honest consequence

// a stable 6-box stack; a box five times the top box's mass placed with its
// centre of mass `offset` metres from the top box's edge (positive: beyond it)
double consequence_motion(double offset) {
    World world;
    Stack stack;
    build_box_stack(world, stack, 6);
    run_until_asleep(world, stack.ids, 5.0);
    const Snapshot before = snapshot(world, stack.ids);
    const Shape heavy = box_shape(0.2, 0.12, 0.1);   // 6.24 kg, five times a 1.25 kg stack box
    const double top = highest_z(stack.shapes[0], world.state(stack.ids.back()).pose);
    const BodyId h = world.add_body(heavy, pose_at(0.06 + offset, 0, top + 0.05 + 0.001));
    const Vec3 placed = world.state(h).pose.p;
    run(world, 300);
    const double stack_moved = max_drift(world, before);
    // the newcomer's own 1 mm settling drop is not a consequence; falling off is
    const double newcomer_moved = distance(world.state(h).pose.p, placed);
    if (offset < 0) {
        return stack_moved;
    }
    return std::max(stack_moved, newcomer_moved);
}

void test_consequence() {
    const double beyond = consequence_motion(0.03);
    const double inside = consequence_motion(-0.03);
    report("10 honest consequence: beyond the edge topples, inside stands", beyond > 0.02 && inside < 5e-4,
           format("3 cm beyond: moved %.0f mm; 3 cm inside: moved %.4f mm", beyond * 1000, inside * 1000));
}

// ---------------------------------------------------------------- 11. wake

void test_wake() {
    World world;
    Stack stack;
    build_box_stack(world, stack, 8);
    run_until_asleep(world, stack.ids, 5.0);
    const bool asleep_before = all_listed_asleep(world, stack.ids);
    const double top = highest_z(stack.shapes[0], world.state(stack.ids.back()).pose);
    world.add_body(stack.shapes[0], pose_at(0, 0, top + 0.02 + 0.001));
    run(world, 3);
    bool woke = false;
    for (size_t i = 0; i < stack.ids.size(); i += 1) {
        if (!world.state(stack.ids[i]).asleep) {
            woke = true;
        }
    }
    report("11 wake: a rock landing on a sleeping stack wakes it", asleep_before && woke,
           format("asleep before %.0f; awake after landing %.0f", asleep_before ? 1 : 0, woke ? 1 : 0));
    run(world, 300);
    const BodyId above = stack.ids[5];
    const double z_before = world.state(above).pose.p.z;
    world.remove_body(stack.ids[4]);
    run(world, 120);
    const double drop = z_before - world.state(above).pose.p.z;
    report("11 wake: removing a middle rock drops the ones above", drop > 0.03,
           format("the rock above fell %.1f mm", drop * 1000));
}

// ---------------------------------------------------------------- 12. load asleep

void test_load_asleep() {
    std::vector<Pose> saved;
    Stack stack;
    {
        World world;
        build_rock_stack(world, stack, 8, 11, RockKind::flat, false);
        run_until_asleep(world, stack.ids, 5.0);
        for (size_t i = 0; i < stack.ids.size(); i += 1) {
            saved.push_back(world.state(stack.ids[i]).pose);
        }
    }
    World world;
    std::vector<BodyId> ids;
    for (size_t i = 0; i < saved.size(); i += 1) {
        ids.push_back(world.add_body(stack.shapes[i], saved[i], Vec3{}, Vec3{}, true));
    }
    const Snapshot loaded = snapshot(world, ids);
    run(world, 600);
    const double drift_asleep = max_drift(world, loaded);
    const Shape newcomer = rock_shape(99, RockKind::flat);
    const BodyId top = ids.back();
    const Pose top_pose = world.state(top).pose;
    const BodyId n = place_on(world, newcomer, &stack.shapes.back(), top, top_pose.p.x, top_pose.p.y, 0.002);
    std::vector<BodyId> all = ids;
    all.push_back(n);
    const double slept = run_until_asleep(world, all, 5.0);
    const double moved = max_drift(world, loaded);
    report("12 load asleep: a saved stack reloads still, and takes a new rock", drift_asleep == 0 && slept >= 0 && moved < 3e-4,
           format("drift while asleep %.1e mm; asleep %.2f s after the new rock; old rocks moved %.4f mm", drift_asleep * 1000, slept, moved * 1000));
}

// ---------------------------------------------------------------- 13. concave

void test_concave() {
    double worst_drift = 0;
    double worst_sleep = 0;
    int unslept = 0;
    for (std::uint32_t seed = 1; seed <= 10; seed += 1) {
        const StackOutcome o = tall_stack(seed, 5, true, 120);
        worst_drift = std::max(worst_drift, o.drift);
        worst_sleep = std::max(worst_sleep, o.sleep_seconds);
        if (!o.slept) {
            unslept += 1;
            std::printf("      concave seed %u did not sleep within 5 s\n", seed);
        }
    }
    report("13 concave: compound rocks stacked 5 high, 10 seeds", unslept == 0 && worst_drift < 3e-4,
           format("asleep within %.2f s (unslept %.0f); drift %.4f mm", worst_sleep, unslept, worst_drift * 1000));
}

// ---------------------------------------------------------------- 14. determinism

std::uint64_t determinism_run() {
    World world;
    Stack stack;
    build_rock_stack(world, stack, 15, 3, RockKind::flat, false);
    std::uint64_t combined = 1469598103934665603ull;
    for (int f = 0; f < 600; f += 1) {
        world.step();
        combined = (combined ^ world.checksum()) * 1099511628211ull;
    }
    return combined;
}

void test_determinism() {
    const std::uint64_t a = determinism_run();
    const std::uint64_t b = determinism_run();
    std::printf("      determinism checksum %016llx (compare across processes)\n", static_cast<unsigned long long>(a));
    report("14 determinism: the same scenario twice gives identical checksums", a == b,
           std::string("run 1 ") + std::to_string(a) + ", run 2 " + std::to_string(b));
}

// ---------------------------------------------------------------- 15. robustness

void test_robustness() {
    World world;
    std::vector<Shape> shapes;
    shapes.reserve(32);
    shapes.push_back(rock_shape(1, RockKind::round));
    shapes.push_back(rock_shape(2, RockKind::round));
    std::vector<BodyId> ids;
    std::vector<double> drop_height;
    ids.push_back(world.add_body(shapes[0], pose_at(0, 0, 0.2)));
    drop_height.push_back(0.2);
    ids.push_back(world.add_body(shapes[1], pose_at(0.02, 0, 0.2)));   // 2 cm inside the first
    drop_height.push_back(0.2);
    double separating = 0;
    for (int f = 0; f < 30; f += 1) {
        world.step();
        const Vec3 rel = world.state(ids[0]).v - world.state(ids[1]).v;
        separating = std::max(separating, std::sqrt(rel.x * rel.x + rel.y * rel.y));
    }
    Random random = make_random(77);
    const RockKind kinds[3] = {RockKind::round, RockKind::flat, RockKind::jagged};
    for (int i = 0; i < 30; i += 1) {
        shapes.push_back(rock_shape(static_cast<std::uint32_t>(100 + i), kinds[i % 3]));
        const double x = random_range(random, -0.15, 0.15);
        const double y = random_range(random, -0.15, 0.15);
        const double h = random_range(random, 0.1, 0.5);
        ids.push_back(world.add_body(shapes.back(), pose_at(x, y, h)));
        drop_height.push_back(h);
    }
    double excess = 0;
    for (int f = 0; f < 600; f += 1) {
        world.step();
        for (size_t i = 0; i < ids.size(); i += 1) {
            const double speed = zc::phys::length(world.state(ids[i]).v);
            const double free_fall = std::sqrt(2 * 9.81 * drop_height[i]) * 1.05 + 0.05;
            excess = std::max(excess, speed - free_fall);
        }
    }
    const bool settled = world.all_asleep();
    const bool pass = world.guard_count() == 0 && excess <= 0 && separating <= 0.55 && settled;
    report("15 robustness: an overlap and a 30-rock heap", pass,
           format("overlap separates at %.3f m/s; worst speed beyond free fall %.3f m/s; guards %.0f", separating, excess, world.guard_count()) +
               std::string(settled ? "; all asleep by 10 s" : "; NOT asleep by 10 s"));
}

// ---------------------------------------------------------------- 16. performance

void test_performance() {
    World world;
    SolverParams solver = world.solver_params();
    solver.sleeping = false;
    world.set_solver_params(solver);
    Stack stack;
    build_rock_stack(world, stack, 15, 21, RockKind::flat, false);
    std::vector<Shape> heap;
    heap.reserve(60);
    Random random = make_random(5);
    const RockKind kinds[3] = {RockKind::round, RockKind::flat, RockKind::jagged};
    for (int i = 0; i < 60; i += 1) {
        heap.push_back(rock_shape(static_cast<std::uint32_t>(500 + i), kinds[i % 3]));
        const double x = 0.6 + random_range(random, -0.2, 0.2);
        const double y = random_range(random, -0.2, 0.2);
        const double z = 0.1 + 0.08 * (i / 10);
        world.add_body(heap.back(), pose_at(x, y, z));
    }
    double total = 0;
    double worst = 0;
    const int frames = 600;
    for (int f = 0; f < frames; f += 1) {
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        world.step();
        const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total += ms;
        worst = std::max(worst, ms);
    }
    const double awake_mean = total / frames;
    solver.sleeping = true;
    world.set_solver_params(solver);
    run(world, 600);
    double asleep_total = 0;
    for (int f = 0; f < 120; f += 1) {
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        world.step();
        const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        asleep_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    const double asleep_mean = asleep_total / 120;
    report("16 performance: 60-rock heap plus a 15-rock stack, all awake", awake_mean < 1.5 && asleep_mean < 0.1,
           format("awake mean %.3f ms (worst %.2f ms); asleep mean %.4f ms", awake_mean, worst, asleep_mean) +
               std::string(world.all_asleep() ? "" : " (not all asleep when timed)"));
}

// ---------------------------------------------------------------- features from the prototype

void test_static_body() {
    World world;
    SolverParams solver = world.solver_params();
    solver.sleeping = false;   // as in the prototype's check: settle fully rather than sleep within the slop
    world.set_solver_params(solver);
    const Shape table = box_shape(0.4, 0.4, 0.05);
    const Shape box = box_shape(0.1, 0.1, 0.1);
    const BodyId t = world.add_static_body(table, pose_at(0, 0, 0.3));
    const BodyId b = world.add_body(box, pose_at(0.05, 0, 0.3 + 0.025 + 0.05 + 0.002));
    run(world, 300);
    const double z = world.state(b).pose.p.z;
    const double table_z = world.state(t).pose.p.z;
    report("feature: a box rests on a static table", std::abs(z - 0.375) < 1e-6 && table_z == 0.3,
           format("box z %.9f; table z %.3f", z, table_z));
}

void test_restitution() {
    double peaks[2] = {0, 0};
    const double values[2] = {0.0, 0.5};
    for (int k = 0; k < 2; k += 1) {
        World world;
        SolverParams solver = world.solver_params();
        solver.sleeping = false;
        world.set_solver_params(solver);
        Material material;
        material.restitution = values[k];
        material.rolling_resistance = 0;
        const Shape box = box_shape(0.08, 0.08, 0.08, material);
        const BodyId b = world.add_body(box, pose_at(0, 0, 0.04 + 0.3));
        bool touched = false;
        for (int f = 0; f < 120; f += 1) {
            world.step();
            const double gap = world.state(b).pose.p.z - 0.04;
            if (gap < 0.002) {
                touched = true;
            }
            if (touched) {
                peaks[k] = std::max(peaks[k], gap);
            }
        }
    }
    report("feature: restitution 0 doesn't bounce; 0.5 rebounds near e^2 h", peaks[0] < 0.002 && peaks[1] > 0.055 && peaks[1] < 0.08,
           format("e = 0: %.3f mm; e = 0.5: %.1f mm (ideal 75)", peaks[0] * 1000, peaks[1] * 1000));
}

void test_rest_and_stable() {
    World world;
    Stack stack;
    build_box_stack(world, stack, 6);
    run(world, 60);
    const zc::phys::RestReport quiet = world.rest_report();
    report("feature: rest report: a settled stack is quiet", quiet.quiet && quiet.awake == 0,
           format("quiet %.0f; awake %.0f; KE %.1e J", quiet.quiet ? 1 : 0, quiet.awake, quiet.kinetic_energy));
    const Shape loose_shape = rock_shape(9, RockKind::round);
    const BodyId loose = world.add_body(loose_shape, pose_at(0.4, 0, 0.3));
    run(world, 10);
    const std::vector<BodyId> during = world.stable_set();
    bool loose_in = false;
    for (size_t i = 0; i < during.size(); i += 1) {
        if (during[i] == loose) {
            loose_in = true;
        }
    }
    report("feature: stable set while a rock falls nearby", during.size() == 6 && !loose_in && !world.rest_report().quiet,
           format("stable %.0f bodies; the falling rock included %.0f", static_cast<double>(during.size()), loose_in ? 1 : 0));
    run(world, 300);
    const std::vector<BodyId> after = world.stable_set();
    report("feature: stable set once it lands", after.size() == 7 && world.rest_report().quiet,
           format("stable %.0f bodies", static_cast<double>(after.size())));
}

struct NamedTest {
    const char* name;
    void (*run)();
};

}  // namespace

int main(int argc, char** argv) {
    const NamedTest tests[] = {
        {"rest", test_rest},
        {"stack", test_tall_stack},
        {"mass", test_mass_ratio},
        {"overhang", test_overhang},
        {"incline", test_incline},
        {"drop", test_drop},
        {"hold", test_hold},
        {"setdown", test_set_down},
        {"shove", test_shove},
        {"consequence", test_consequence},
        {"wake", test_wake},
        {"load", test_load_asleep},
        {"concave", test_concave},
        {"determinism", test_determinism},
        {"robustness", test_robustness},
        {"performance", test_performance},
        {"static", test_static_body},
        {"restitution", test_restitution},
        {"stable", test_rest_and_stable},
    };
    const int count = static_cast<int>(sizeof tests / sizeof tests[0]);
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; i += 1) {
        bool selected = argc < 2;
        for (int a = 1; a < argc; a += 1) {
            if (std::strcmp(argv[a], tests[i].name) == 0) {
                selected = true;
            }
        }
        if (selected) {
            tests[i].run();
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%s (%.1f s)\n", failures == 0 ? "ALL PASS" : (std::to_string(failures) + " FAILURES").c_str(), seconds);
    return failures == 0 ? 0 : 1;
}
