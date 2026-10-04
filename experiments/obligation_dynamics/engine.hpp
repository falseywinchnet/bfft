#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "physics.hpp"

namespace obligation {
using zc::phys::Vec3;using zc::phys::Quat;using zc::phys::Pose;
using zc::phys::Shape;using zc::phys::BodyId;using zc::phys::BodyState;using zc::phys::Contact;
struct Parameters {
    Vec3 gravity{0,0,-9.81};
    double ground_z=0,ground_friction=.5;
    double lookahead=.1;           // validity horizon for free-flight envelopes
    double contact_step=1.0/240;   // upper horizon for changing contact groups
    double angular_step=.04;       // orientation integration horizon, radians
    double skin=2e-5;              // contact arrival tolerance, metres
    double release_gap=2e-4;
    double velocity_tolerance=2e-6;
    double moving_velocity_tolerance=1e-4;
    double force_tolerance=1e-5;   // acceleration residual, m/s^2 at the surface
    double equilibrium_speed=1e-5;
    double equilibrium_acceleration=2e-4;
    double restitution_threshold=.1;
    int max_atomic_updates=100000;
    int max_events_per_advance=200000;
    bool certify_equilibrium=true;
};
struct Statistics {
    std::uint64_t events=0,stale_events=0,casts=0,cast_iterations=0,narrowphase=0;
    std::uint64_t broadphase_queries=0,pending_reuses=0,manifold_reuses=0;
    std::uint64_t separation_certificates=0;
    std::uint64_t impact_atomics=0,force_atomics=0,position_atomics=0;
    std::uint64_t group_solves=0,equilibrium_certificates=0,wakes=0;
    std::uint64_t unconverged_impacts=0,unconverged_forces=0;
    std::uint64_t integrated_bodies=0;
    int largest_group=0;
    double discarded_equilibrium_energy=0,max_force_residual=0;
};
// Geometry cooking and the convex manifold routine are shared with Wrench.
// Scheduling, contact state, integration and dynamics are independent.
class World {
public:
    explicit World(Parameters parameters={});
    ~World();
    World(const World&)=delete;World& operator=(const World&)=delete;
    BodyId add_body(const Shape&,Pose,Vec3 velocity={},Vec3 angular={});
    BodyId add_static_body(const Shape&,Pose);
    void remove_body(BodyId);
    void apply_impulse(BodyId,Vec3 impulse,Vec3 world_point);
    void set_pose(BodyId,Pose);
    void advance(double seconds);
    double time() const;
    BodyState state(BodyId) const;
    std::vector<Contact> contacts() const;
    Statistics statistics() const;
    double kinetic_energy() const;
    double equilibrium_residual(BodyId) const;
private:
    struct Impl;std::unique_ptr<Impl> impl;
};
}
