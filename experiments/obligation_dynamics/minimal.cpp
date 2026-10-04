#include "engine.hpp"
#include <iostream>

int main() {
    zc::phys::ShapeDesc description;
    description.density = 1000;
    description.friction = .5;
    description.restitution = 0;
    zc::phys::HullDesc hull;
    for (int vertex = 0; vertex < 8; ++vertex)
        hull.points.push_back({vertex & 1 ? .05 : -.05,
                               vertex & 2 ? .05 : -.05,
                               vertex & 4 ? .05 : -.05});
    description.hulls.push_back(hull);
    auto shape = zc::phys::cook(description);
    obligation::World world;
    auto body = world.add_body(shape, {{0, 0, 1}, {}}, {0, 0, -30});
    world.advance(1);
    auto state = world.state(body);
    auto before = world.statistics();
    world.advance(10);
    auto after = world.statistics();
    std::cout << "height=" << state.pose.p.z
              << " equilibrium=" << state.asleep
              << " force_residual=" << world.equilibrium_residual(body)
              << " unchanged_contact_updates="
              << after.force_atomics - before.force_atomics << '\n';
    return state.asleep && after.force_atomics == before.force_atomics ? 0 : 1;
}
