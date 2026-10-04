#include "engine.hpp"
#include "broadphase.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace zc::phys;
using obligation::Parameters;
static void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
static Shape box(Vec3 size,double density=1000,double restitution=0,double friction=.5){
    ShapeDesc d;d.density=density;d.restitution=restitution;d.friction=friction;d.rolling_resistance=0;HullDesc h;
    for(int i=0;i<8;i++)h.points.push_back({(i&1?.5:-.5)*size.x,(i&2?.5:-.5)*size.y,(i&4?.5:-.5)*size.z});d.hulls.push_back(h);return cook(d);
}
static void hierarchy(){
    obligation::Tree tree;std::mt19937 rng(113);std::uniform_real_distribution<double> dist(-10,10);
    std::vector<obligation::Bounds> boxes;std::vector<int> leaves;
    for(int i=0;i<512;i++){Vec3 low{dist(rng),dist(rng),dist(rng)};boxes.push_back({low,low+Vec3{.7,.5,.9}});leaves.push_back(tree.insert(i,boxes.back()));}
    for(int round=0;round<1000;round++){
        int id=int(rng()%boxes.size());tree.erase(leaves[id]);Vec3 low{dist(rng),dist(rng),dist(rng)};boxes[id]={low,low+Vec3{.8,.8,.8}};leaves[id]=tree.insert(id,boxes[id]);
        Vec3 q{dist(rng),dist(rng),dist(rng)};obligation::Bounds query{q,q+Vec3{3,3,3}};std::vector<int> found,expected;
        tree.query(query,[&](int j){found.push_back(j);});for(int j=0;j<512;j++)if(obligation::intersects(query,boxes[j]))expected.push_back(j);
        std::sort(found.begin(),found.end());require(found==expected,"dynamic hierarchy differs from exhaustive AABB query");
    }
}
static void ballistic(){
    Parameters p;p.ground_z=-100;obligation::World a(p),b(p);Shape s=box({.1,.1,.1});Pose pose{{0,0,10},{}};Vec3 v{3,-2,4};
    auto ia=a.add_body(s,pose,v),ib=b.add_body(s,pose,v);a.advance(.53);for(int i=0;i<53;i++)b.advance(.01);
    auto x=a.state(ia),y=b.state(ib);Vec3 exact=pose.p+v*.53+p.gravity*(.5*.53*.53);
    require(length(x.pose.p-exact)<1e-12,"analytic free flight position");require(length(x.v-(v+p.gravity*.53))<1e-12,"analytic free flight velocity");
    require(length(x.pose.p-y.pose.p)<1e-12,"observation rate changed dynamics");require(a.statistics().events==b.statistics().events,"observation rate changed event schedule");
}
static void impact(){
    Parameters p;p.gravity={};p.ground_z=-100;obligation::World world(p);
    Shape a=box({.1,.1,.1},1000,1,0),b=box({.1,.1,.1},2000,1,0);
    int ia=world.add_body(a,{{-.2,0,1},{}},{2,0,0}),ib=world.add_body(b,{{.2,0,1},{}},{-1,0,0});
    double initial=world.kinetic_energy();world.advance(.2);auto x=world.state(ia),y=world.state(ib);
    require(std::abs(x.v.x+2)<2e-4&&std::abs(y.v.x-1)<2e-4,"two-mass elastic impact velocities");
    require(length(x.v*a.mass+y.v*b.mass)<1e-10,"impact violates linear momentum");
    require(std::abs(world.kinetic_energy()-initial)<initial*1e-5,"elastic impact energy");
}
static void offcentre(){
    Parameters p;p.gravity={};p.ground_z=-100;obligation::World world(p);Shape s=box({.2,.2,.2},1000,0,.3);
    int a=world.add_body(s,{{-.3,.05,1},{}},{3,0,0}),b=world.add_body(s,{{0,0,1},{}},{});
    Vec3 totalp{3*s.mass,0,0},angular0=cross(Vec3{-.3,.05,1},totalp);double initial=world.kinetic_energy();
    world.advance(.09);auto x=world.state(a),y=world.state(b);Mat3 inertia;for(int k=0;k<9;k++)inertia.m[k]=s.inertia[k];
    Vec3 angular=cross(x.pose.p,x.v*s.mass)+cross(y.pose.p,y.v*s.mass)+mul(rotated(to_matrix(x.pose.q),inertia),x.w)+mul(rotated(to_matrix(y.pose.q),inertia),y.w);
    require(length(x.v*s.mass+y.v*s.mass-totalp)<1e-9,"off-centre momentum");require(length(angular-angular0)<1e-5,"off-centre angular momentum");
    require(world.kinetic_energy()<=initial*(1+1e-7),"inelastic atomics created kinetic energy");require(length(x.w)+length(y.w)>.1,"off-centre impact must transfer angular momentum");
}
static void fast_floor(){
    for(double speed:{1.0,10.0,30.0,300.0,3000.0}){
        std::cout<<"  slab speed "<<speed<<std::endl;
        Parameters p;p.ground_z=-10;obligation::World world(p);Shape slab=box({.5,.5,.004}),body=box({.02,.02,.02});
        world.add_static_body(slab,{{0,0,.15},{}});int id=world.add_body(body,{{0,0,.6},{}},{0,0,-speed});world.advance(.6/speed);
        auto s=world.state(id);require(s.pose.p.z>=.162-p.skin*2,"fast body crossed a 4 mm slab at "+std::to_string(speed));
    }
}
static void spinning(){
    for(double speed:{80.,800.,8000.}){
    Parameters p;p.gravity={};p.ground_z=-10;obligation::World world(p);
    Shape rod=box({.6,.015,.015}),wall=box({.8,.006,.3});
    world.add_static_body(wall,{{0,.22,1},{}});int id=world.add_body(rod,{{0,0,1},{}},{},{0,0,speed});
    world.advance(.018*80/speed);auto state=world.state(id);double high=-1e9;
    for(const auto& hull:rod.hulls)for(Vec3 v:hull.vertices)high=std::max(high,(state.pose.p+rotate(state.pose.q,v)).y);
    require(high<=.217+p.skin*4,"rotation-only sweep crossed the wall");require(world.statistics().group_solves>0,"rotating sweep created no contact event");
    }
}
static void rest_and_wake(){
    Parameters p;obligation::World world(p);Shape s=box({.1,.1,.1});int id=world.add_body(s,{{0,0,.2},{}});
    world.advance(1);auto before=world.statistics();require(world.state(id).asleep,"floor contact did not certify equilibrium");
    require(world.equilibrium_residual(id)<p.equilibrium_acceleration,"equilibrium force/torque certificate invalid");
    world.advance(5);auto after=world.statistics();require(after.narrowphase==before.narrowphase&&after.force_atomics==before.force_atomics,"unchanged equilibrium was rediscovered");
    world.apply_impulse(id,{.1,0,.3},world.state(id).pose.p);world.advance(.02);require(!world.state(id).asleep&&world.state(id).pose.p.z>.053,"impulse did not invalidate equilibrium");
}
static void stack(){
    Parameters p;obligation::World world(p);Shape s=box({.1,.1,.1});std::vector<int> ids;
    for(int i=0;i<6;i++)ids.push_back(world.add_body(s,{{0,0,.05+.10001*i},{}}));world.advance(2);
    for(int id:ids){require(world.state(id).asleep,"stack did not certify equilibrium");require(world.equilibrium_residual(id)<p.equilibrium_acceleration,"stack certificate residual");}
    auto before=world.statistics();world.advance(2);require(world.statistics().force_atomics==before.force_atomics,"resting stack performed new atomics");
    world.apply_impulse(ids.back(),{.2,0,0},world.state(ids.back()).pose.p);world.advance(.05);require(world.statistics().wakes>before.wakes,"changed top load did not wake transmission graph");
}
static void remove_support(){
    Parameters p;p.ground_z=-100;obligation::World world(p);Shape platform=box({.4,.4,.1}),s=box({.1,.1,.1});
    int support=world.add_static_body(platform,{{0,0,1},{}}),id=world.add_body(s,{{0,0,1.10001},{}});world.advance(.5);
    require(world.state(id).asleep,"platform did not support body");world.remove_body(support);world.advance(.2);
    require(world.state(id).pose.p.z<1,"removing support left floating certified body");
}
static void sparse_equilibria(){
    Parameters p;obligation::World world(p);Shape s=box({.04,.04,.04});std::vector<int> ids;
    for(int i=0;i<512;i++)ids.push_back(world.add_body(s,{{(i%32)*.15,(i/32)*.15,.02001},{}}));world.advance(.3);
    int quiet=0;for(int id:ids)quiet+=world.state(id).asleep;require(quiet==512,"512 isolated bodies failed equilibrium");
    auto before=world.statistics();world.advance(5);auto after=world.statistics();
    require(before.narrowphase==after.narrowphase&&before.broadphase_queries==after.broadphase_queries,"resting scene made geometric queries");
    world.apply_impulse(ids[0],{0,0,.02},world.state(ids[0]).pose.p);world.advance(.01);
    require(world.statistics().largest_group==1,"isolated impulse joined unrelated bodies");
}
static void cavity(){
    Parameters p;obligation::World world(p);ShapeDesc cup;cup.density=1000;cup.restitution=0;cup.rolling_resistance=0;cup.friction=.5;
    struct Part{Vec3 size,centre;};Part parts[]={{{.16,.16,.01},{0,0,.005}},{{.01,.16,.12},{-.075,0,.07}},{{.01,.16,.12},{.075,0,.07}},{{.14,.01,.12},{0,-.075,.07}},{{.14,.01,.12},{0,.075,.07}}};
    for(auto part:parts){HullDesc hull;for(int i=0;i<8;i++)hull.points.push_back(part.centre+Vec3{(i&1?.5:-.5)*part.size.x,(i&2?.5:-.5)*part.size.y,(i&4?.5:-.5)*part.size.z});cup.hulls.push_back(hull);}
    Shape shape=cook(cup);world.add_static_body(shape,{shape.com_offset,{}});int id=world.add_body(box({.03,.03,.03}),{{0,0,.3},{}});world.advance(1);
    require(std::abs(world.state(id).pose.p.z-.025)<5e-5,"compound cavity was filled or bottom missed");
}
static void slope(){
    for(double angle:{.3,.7}){
        Parameters p;p.ground_z=-10;p.contact_step=1.0/480;obligation::World world(p);Quat q=from_axis_angle({0,1,0},angle);Vec3 normal=rotate(q,{0,0,1}),centre{0,0,1};
        Shape platform=box({2,2,.04},1000,0,.6),s=box({.1,.1,.1},1000,0,.6);
        world.add_static_body(platform,{centre,q});int id=world.add_body(s,{centre+normal*.07001,q});double energy0=world.kinetic_energy();world.advance(.2);BodyState state=world.state(id);
        if(angle==.3){require(state.asleep,"sub-friction-angle slope failed static equilibrium");require(world.equilibrium_residual(id)<p.equilibrium_acceleration,"inclined static-force residual");}
        else {require(state.v.x>.1,"super-friction-angle body did not slide");double gravitational=s.mass*9.81*((centre+normal*.07001).z-state.pose.p.z);require(world.kinetic_energy()<=energy0+gravitational+1e-5,"slope friction created mechanical energy");}
    }
}
static void free_spin(){
    double errors[2];
    for(int refine=0;refine<2;refine++){
        Parameters p;p.gravity={};p.ground_z=-100;p.angular_step=.04/(refine+1);obligation::World world(p);Shape s=box({.1,.2,.3});Vec3 omega{3,4,5};
        Mat3 inertia;for(int k=0;k<9;k++)inertia.m[k]=s.inertia[k];Vec3 momentum=mul(inertia,omega);
        int id=world.add_body(s,{{0,0,10},{}},{},omega);double energy=world.kinetic_energy();world.advance(1);auto state=world.state(id);
        require(length(mul(rotated(to_matrix(state.pose.q),inertia),state.w)-momentum)<1e-12,"free spin angular momentum");
        errors[refine]=std::abs(world.kinetic_energy()/energy-1);require(errors[refine]<.005,"free spin energy drift");
    }
    require(errors[1]<errors[0]*.7,"angular integration failed refinement check");
}
static void observation_collision(){
    Parameters p;obligation::World a(p),b(p);Shape s=box({.1,.1,.1});int ia=a.add_body(s,{{0,0,.3},{}},{1,0,-3}),ib=b.add_body(s,{{0,0,.3},{}},{1,0,-3});
    a.advance(.23);for(int i=0;i<23;i++)b.advance(.01);auto x=a.state(ia),y=b.state(ib);
    require(length(x.pose.p-y.pose.p)<1e-10&&length(x.v-y.v)<1e-10,"observation cadence changed impact dynamics");
    require(a.statistics().events==b.statistics().events,"observation cadence changed collision schedule");
}
static void standing_and_pending(){
    Parameters p;p.ground_z=-10;obligation::World world(p);ShapeDesc compound;compound.density=1000;compound.friction=0;compound.restitution=0;compound.rolling_resistance=0;
    struct Part{Vec3 size,p;};Part parts[]={{{2,.4,.004},{0,0,0}},{{.004,.4,.4},{.3,0,.2}}};
    for(auto part:parts){HullDesc hull;for(int i=0;i<8;i++)hull.points.push_back(part.p+Vec3{(i&1?.5:-.5)*part.size.x,(i&2?.5:-.5)*part.size.y,(i&4?.5:-.5)*part.size.z});compound.hulls.push_back(hull);}
    Shape shape=cook(compound);world.add_static_body(shape,{shape.com_offset,{}});
    int id=world.add_body(box({.02,.02,.02},1000,0,0),{{-.1,0,.01201},{}},{300,0,0});world.advance(.003);
    require(world.state(id).pose.p.x<=.288+p.skin*2,"standing floor contact suppressed a pending wall obligation");
    require(std::abs(world.state(id).v.x)<.001,"pending compound wall impact failed to stop the body");
}
static void shared_compound_witness(){
    ShapeDesc d;d.density=1000;d.friction=.5;d.restitution=0;d.rolling_resistance=0;
    for(double side:{-1.,1.}){HullDesc h;h.points={{0,0,0},{side*.04,0,.03},{0,.04,.03},{0,-.04,.03}};d.hulls.push_back(h);}
    Shape s=cook(d);obligation::World world;int id=world.add_body(s,{s.com_offset+Vec3{0,0,1e-5},{}});
    world.advance(.02);require(std::isfinite(world.state(id).pose.p.z),"shared compound witness stalled the event clock");
    require(world.statistics().events<1000,"deduplicated witness was repeatedly scheduled as a new collision");
}
int main(){
    struct Check{const char* name;void(*run)();};Check checks[]={{"dynamic hierarchy",hierarchy},{"analytic flight / observation invariance",ballistic},{"elastic two-mass impact",impact},{"off-centre momentum and energy",offcentre},{"thin slab 1–3000 m/s",fast_floor},{"rotation-only sweep",spinning},{"equilibrium / impulse invalidation",rest_and_wake},{"retained six-body load transmission",stack},{"support removal",remove_support},{"512 independent equilibria",sparse_equilibria},{"open compound cavity",cavity},{"static and sliding friction",slope},{"free angular momentum / energy refinement",free_spin},{"impact observation invariance",observation_collision}};
    int failures=0;for(const Check& check:checks){try{check.run();std::cout<<"PASS "<<check.name<<std::endl;}catch(const std::exception& e){failures++;std::cerr<<"FAIL "<<check.name<<": "<<e.what()<<std::endl;}}
    try{standing_and_pending();std::cout<<"PASS simultaneous standing and pending obligations"<<std::endl;}catch(const std::exception& e){failures++;std::cerr<<"FAIL simultaneous obligations: "<<e.what()<<std::endl;}
    try{shared_compound_witness();std::cout<<"PASS shared convex-sector witness"<<std::endl;}catch(const std::exception& e){failures++;std::cerr<<"FAIL shared convex-sector witness: "<<e.what()<<std::endl;}
    return failures?1:0;
}
