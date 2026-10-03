#include <physics.hpp>
#include <cstdio>
int main() {
    zc::phys::ShapeDesc desc;
    zc::phys::HullDesc hull;
    for (int corner=0;corner<8;++corner)
        hull.points.push_back({corner&1 ? .05 : -.05, corner&2 ? .05 : -.05, corner&4 ? .05 : -.05});
    desc.hulls.push_back(hull);
    zc::phys::World world;
    auto id=world.add_body(zc::phys::cook(desc), {{0,0,.5},{}});
    for(int frame=0;frame<300;++frame) world.step();
    auto state=world.state(id);
    std::printf("cube center %.6f m; asleep %d\n",state.pose.p.z,state.asleep);
    return state.asleep && state.pose.p.z > .049 && state.pose.p.z < .051 ? 0 : 1;
}
