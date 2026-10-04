#pragma once
#include "physics.hpp"
#include <memory>

namespace tape {
using namespace zc::phys;
struct Parameters {
    Vec3 gravity{0,0,-9.81};
    double ground_z=0,ground_friction=.5;
    double step=1.0/120,angular_step=.1;
    double skin=2e-5,velocity_tolerance=2e-5;
    int reads=32,position_reads=4;
    unsigned workers=0; // Zero selects the host hardware concurrency.
    bool moving_tape=true,streamed=true,skip_unchanged=true;
};
struct Statistics {
    std::uint64_t steps=0,reads=0,exchanges=0,swaps=0,passes=0,narrowphase=0;
    std::uint64_t unconverged=0,position_exchanges=0,parallel_batches=0,rotation_reads=0,unchanged_tasks=0;
    double max_residual=0;
    double geometry_seconds=0,solve_seconds=0,integration_seconds=0;
};
class World {
public:
    explicit World(Parameters parameters={});
    ~World();
    World(const World&)=delete;World& operator=(const World&)=delete;
    BodyId add_body(const Shape&,Pose,Vec3 velocity={},Vec3 angular={});
    BodyId add_static_body(const Shape&,Pose);
    void remove_body(BodyId);
    void apply_impulse(BodyId,Vec3 momentum,Vec3 world_point);
    void set_pose(BodyId,Pose);
    void advance(double seconds);
    double time()const;
    BodyState state(BodyId)const;
    std::vector<Contact> contacts()const;
    std::vector<BodyId> order()const;
    Statistics statistics()const;
    Parameters parameters()const;
    double kinetic_energy()const;
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
}
