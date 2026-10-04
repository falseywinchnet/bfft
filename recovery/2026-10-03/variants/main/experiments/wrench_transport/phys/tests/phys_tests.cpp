// Placeholder until the acceptance tests are ported: one smoke check.
#include <cstdio>

#include "../physics.hpp"

int main() {
    zc::phys::World world;
    zc::phys::ShapeDesc desc;
    zc::phys::HullDesc hull;
    for (int i = 0; i < 8; i += 1) {
        zc::phys::Vec3 p{(i & 1 ? 0.05 : -0.05), (i & 2 ? 0.05 : -0.05), (i & 4 ? 0.05 : -0.05)};
        hull.points.push_back(p);
    }
    desc.hulls.push_back(hull);
    zc::phys::Pose pose;
    pose.p.z = 0.06;
    const zc::phys::BodyId id = world.add_body(zc::phys::cook(desc), pose);
    for (int f = 0; f < 300; f += 1) {
        world.step();
    }
    const zc::phys::BodyState state = world.state(id);
    const double error = state.pose.p.z - 0.05;
    const bool pass = error < 1.0e-6 && error > -1.0e-6 && world.all_asleep();
    std::printf("%s  box rests on the ground  z error=%.3e asleep=%d\n", pass ? "PASS" : "FAIL", error,
                world.all_asleep() ? 1 : 0);
    return pass ? 0 : 1;
}
