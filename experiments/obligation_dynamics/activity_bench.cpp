#include "engine.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace zc::phys;
using Clock=std::chrono::steady_clock;
static Shape block(){ShapeDesc d;d.density=1000;d.friction=.5;d.restitution=0;d.rolling_resistance=0;HullDesc h;for(int i=0;i<8;i++)h.points.push_back({(i&1?.02:-.02),(i&2?.02:-.02),(i&4?.02:-.02)});d.hulls.push_back(h);return cook(d);}
static double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
template<class Engine,class Step> static void run(Engine& world,Step step,const std::string& name,int count,int repeat,bool native){
    Shape shape=block();std::vector<int> ids;for(int i=0;i<count;i++)ids.push_back(world.add_body(shape,{{(i%32)*.15,(i/32)*.15,.02001},{}}));
    for(int frame=0;frame<120;frame++)step();int quiet=0;double error=0;for(int id:ids){auto state=world.state(id);quiet+=state.asleep;error=std::max(error,std::abs(state.pose.p.z-.02));}
    auto start=Clock::now();for(int frame=0;frame<1000;frame++)step();double resting=elapsed(start);
    start=Clock::now();for(int cycle=0;cycle<20;cycle++){world.set_pose(ids[0],{{0,0,.07},{}});for(int frame=0;frame<30;frame++)step();}double changing=elapsed(start);
    double final_speed=0;for(int id:ids){auto state=world.state(id);final_speed=std::max(final_speed,length(state.v)+shape.radius*length(state.w));error=std::max(error,std::abs(state.pose.p.z-.02));}
    std::cout<<std::setprecision(12)<<"{\"engine\":\""<<name<<"\",\"count\":"<<count<<",\"repeat\":"<<repeat<<",\"quiet\":"<<quiet<<",\"resting1000FramesSeconds\":"<<resting<<",\"oneChanging600FramesSeconds\":"<<changing<<",\"finalMaxSpeed\":"<<final_speed<<",\"maxHeightErrorMm\":"<<error*1000<<",\"newModel\":"<<(native?"true":"false")<<"}"<<std::endl;
}
int main(int argc,char** argv){
    int count=argc>1?std::stoi(argv[1]):512;
    for(int repeat=0;repeat<4;repeat++)for(int order=0;order<3;order++){
        int engine=(repeat+order)%3;
        if(engine==0){obligation::World world;run(world,[&](){world.advance(1.0/60);},"obligation",count,repeat,true);}
        else {WorldParams p;p.frame_dt=1.0/60;p.linear_damping=p.angular_damping=0;p.ground_friction=.5;World world(p);SolverParams s;s.sleeping=engine==1;s.ground_rolling_resistance=0;world.set_solver_params(s);run(world,[&](){world.step();},engine==1?"wrench-sleeping":"wrench-awake",count,repeat,false);}
    }
}
