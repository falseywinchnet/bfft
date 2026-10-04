#include "engine.hpp"
#include "math.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace zc::phys;
static void check(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
static Shape box(Vec3 size,double friction=.5,double bounce=0){ShapeDesc d;d.friction=friction;d.restitution=bounce;d.rolling_resistance=0;HullDesc h;for(int i=0;i<8;i++)h.points.push_back({(i&1?.5:-.5)*size.x,(i&2?.5:-.5)*size.y,(i&4?.5:-.5)*size.z});d.hulls.push_back(h);return cook(d);}
static void flight(){tape::Parameters p;p.gravity={0,0,-9.81};tape::World w(p);auto id=w.add_body(box({.1,.1,.1}),{{0,0,10},{}},{1,2,3});w.advance(.2);auto s=w.state(id);check(length(s.pose.p-Vec3{.2,.4,10+.6-.5*9.81*.04})<1e-10,"free flight integral");check(length(s.v-Vec3{1,2,3-9.81*.2})<1e-10,"free flight velocity");}
static void momentum(unsigned workers){tape::Parameters p;p.gravity={};p.ground_z=-100;p.workers=workers;p.reads=24;tape::World w(p);auto shape=box({.1,.1,.1},0);std::vector<std::pair<int,int>> ids;
// Enough contacts to exercise the reference pool's actual parallel path.
for(int i=0;i<400;i++){double y=i*.3;ids.push_back({w.add_body(shape,{{-.05,y,2},{}},{1,0,0}),w.add_body(shape,{{.05,y,2},{}},{-1,0,0})});}
double before=w.kinetic_energy();w.advance(1./120);Vec3 momentum{};for(auto [a,b]:ids){auto x=w.state(a),y=w.state(b);momentum=momentum+(x.v+y.v)*shape.mass;check(x.pose.p.x<y.pose.p.x,"pair crossed");check(length(x.v)+length(y.v)<1e-3,"inelastic stop");}check(length(momentum)<1e-10,"equal opposite momentum");check(w.kinetic_energy()<=before+1e-9,"contact created energy");if(workers==4)check(w.statistics().parallel_batches>0,"parallel lane not exercised");}
static Vec3 angular_momentum(const Shape& shape,BodyState state){Mat3 local;for(int i=0;i<9;i++)local.m[i]=shape.inertia[i];return cross(state.pose.p,state.v*shape.mass)+mul(rotated(to_matrix(state.pose.q),local),state.w);}
static void shared_downstream(){
    tape::Parameters p;p.gravity={};p.ground_z=-100;p.workers=4;p.reads=64;p.streamed=true;tape::World world(p);
    Shape plate=box({10,5,.1},0),small=box({.1,.1,.1},0);int count=800;
    int base=world.add_body(plate,{{0,0,1},{}},{0,0,count*small.mass/plate.mass});
    std::vector<int> ids;for(int y=0;y<20;y++)for(int x=0;x<40;x++)ids.push_back(world.add_body(small,{{(x-19.5)*.2,(y-9.5)*.2,1.1},{}},{0,0,-1}));
    double initial=world.kinetic_energy();Vec3 initial_angular=angular_momentum(plate,world.state(base));for(int id:ids)initial_angular=initial_angular+angular_momentum(small,world.state(id));world.advance(1./120);
    Vec3 total=world.state(base).v*plate.mass;for(int id:ids)total=total+world.state(id).v*small.mass;
    check(length(total)<1e-8,"lost simultaneous downstream momentum");
    Vec3 final_angular=angular_momentum(plate,world.state(base));for(int id:ids)final_angular=final_angular+angular_momentum(small,world.state(id));
    check(length(final_angular-initial_angular)<1e-7,"lost simultaneous downstream angular momentum");
    check(world.kinetic_energy()<initial*.001,"shared downstream failed to dissipate collision energy");
    check(world.statistics().parallel_batches>0,"shared downstream did not use workers");
}
static void stack(){tape::Parameters p;p.reads=128;tape::World w(p);auto shape=box({.1,.1,.1});std::vector<int> ids;for(int i=0;i<12;i++)ids.push_back(w.add_body(shape,{{0,0,.05+i*.1},{}}));for(int f=0;f<240;f++)w.advance(1./60);double error=0,speed=0;for(int i=0;i<12;i++){auto s=w.state(ids[i]);error=std::max(error,length(s.pose.p-Vec3{0,0,.05+i*.1}));speed=std::max(speed,length(s.v));}std::cout<<"stack error "<<error<<" speed "<<speed<<"\n";check(error<.005,"stack drift");check(speed<.02,"stack failed to settle");auto order=w.order();std::sort(order.begin(),order.end());check(order==ids,"tape permutation lost body");}
static void slab(){tape::Parameters p;p.reads=32;p.gravity={};p.ground_z=-100;tape::World w(p);w.add_static_body(box({.005,2,2}),{{0,0,1},{}});auto id=w.add_body(box({.02,.02,.02}),{{-.2,0,1},{}},{100,0,0});w.advance(.02);auto s=w.state(id);std::cout<<"slab x "<<s.pose.p.x<<" vx "<<s.v.x<<" w "<<length(s.w)<<"\n";check(s.pose.p.x<-.0124,"thin slab tunneling");check(std::abs(s.v.x)<.1,"thin slab normal velocity error exceeds 0.1 percent");}
static void friction(){tape::Parameters p;p.reads=24;tape::World w(p);auto id=w.add_body(box({.1,.1,.1},.7),{{0,0,.05},{}},{1,0,0});w.advance(.5);check(length(w.state(id).v)<.01,"friction did not stop sliding");check(w.state(id).pose.p.z>.0499,"floor penetration");}
static void cavity(){
    ShapeDesc desc;desc.friction=.5;desc.restitution=0;
    auto part=[&](Vec3 centre,Vec3 size){HullDesc h;for(int i=0;i<8;i++)h.points.push_back(centre+Vec3{(i&1?.5:-.5)*size.x,(i&2?.5:-.5)*size.y,(i&4?.5:-.5)*size.z});desc.hulls.push_back(h);};
    part({0,0,.003},{.1,.1,.006});part({-.047,0,.05},{.006,.1,.1});part({.047,0,.05},{.006,.1,.1});part({0,-.047,.05},{.1,.006,.1});part({0,.047,.05},{.1,.006,.1});
    Shape cup=cook(desc);tape::Parameters p;p.ground_z=-100;p.reads=64;tape::World world(p);
    int support=world.add_static_body(cup,{cup.com_offset,{}});int item=world.add_body(box({.02,.02,.02}),{{0,0,.15},{}});
    world.advance(1);auto state=world.state(item);check(state.pose.p.z>.0159&&state.pose.p.z<.03,"compound cavity was filled or crossed");
    world.remove_body(support);world.advance(.1);check(world.state(item).v.z<-.9,"removed support retained a force");
}
static void unchanged_equivalence(){
    tape::Parameters p;p.workers=1;p.reads=32;p.streamed=false;tape::World fast(p);p.skip_unchanged=false;tape::World full(p);
    Shape shape=box({.1,.1,.1});std::vector<int> ids;
    for(int z=0;z<3;z++)for(int x=0;x<3;x++){Pose pose{{x*.1,0,.05+z*.1},{}};ids.push_back(fast.add_body(shape,pose));full.add_body(shape,pose);}
    for(int f=0;f<120;f++){
        if(f==30){Vec3 point=fast.state(ids.back()).pose.p;fast.apply_impulse(ids.back(),{.1,0,0},point);full.apply_impulse(ids.back(),{.1,0,0},point);}
        fast.advance(1./60);full.advance(1./60);
        for(int id:ids){auto a=fast.state(id),b=full.state(id);
            check(length(a.pose.p-b.pose.p)==0&&length(a.v-b.v)==0&&length(a.w-b.w)==0,"unchanged-input skip altered arithmetic result");}
    }
    check(fast.statistics().unchanged_tasks>0,"unchanged-input optimization was not exercised");
}
int main(){try{flight();momentum(1);momentum(4);shared_downstream();slab();friction();stack();cavity();unchanged_equivalence();std::cout<<"all tape checks passed\n";}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
