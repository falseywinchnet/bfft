#include "physics.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace zc::phys;
using Clock=std::chrono::steady_clock;
struct Item {int shape;double release;Pose pose;Vec3 v,w;BodyId id=-1;};
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static std::vector<Vec3> box(Vec3 s){std::vector<Vec3> p;for(int i=0;i<8;i++)p.push_back({(i&1?.5:-.5)*s.x,(i&2?.5:-.5)*s.y,(i&4?.5:-.5)*s.z});return p;}
int main(int argc,char**argv){
 if(argc<4||argc>5)return 2;
 std::ifstream in(argv[1]);std::ofstream out(argv[2]);const int hz=std::stoi(argv[3]);const int internal_hz=argc==5?std::stoi(argv[4]):120;if(internal_hz<hz||internal_hz%hz)return 2;
 std::string magic;double density,friction,duration;Vec3 g;int ns,sample_hz,geometry_hz;
 in>>magic>>density>>friction>>g.x>>g.y>>g.z>>duration>>sample_hz>>geometry_hz>>ns;
 if(magic!="PACK1"||hz<60||hz%60)return 2;
 const auto setup=Clock::now();std::vector<Shape> shapes;
 for(int s=0;s<ns;s++){ShapeDesc d;d.density=density;d.friction=friction;d.restitution=0;d.rolling_resistance=0;int nh;in>>nh;for(int h=0;h<nh;h++){HullDesc hd;int nv;in>>nv;for(int v=0;v<nv;v++){Vec3 p;in>>p.x>>p.y>>p.z;hd.points.push_back(p);}d.hulls.push_back(hd);}shapes.push_back(cook(d));}
 WorldParams p;p.frame_dt=1.0/internal_hz;p.gravity=g;p.linear_damping=0;p.angular_damping=0;p.ground_friction=friction;World world(p);SolverParams solver;solver.sleeping=false;solver.ground_rolling_resistance=0;world.set_solver_params(solver);
 int nw;in>>nw;for(int i=0;i<nw;i++){Vec3 c,s;in>>c.x>>c.y>>c.z>>s.x>>s.y>>s.z;ShapeDesc d;d.hulls.push_back({box(s)});d.friction=friction;d.restitution=0;d.rolling_resistance=0;Pose pose;pose.p=c;world.add_static_body(cook(d),pose);}
 int nf;in>>nf;for(int i=0;i<nf;i++){int shape;Pose pose;in>>shape>>pose.p.x>>pose.p.y>>pose.p.z>>pose.q.w>>pose.q.x>>pose.q.y>>pose.q.z;world.add_static_body(shapes[shape],pose);}
 int n;in>>n;std::vector<Item> items(n);double release_end=0;for(auto&b:items){in>>b.shape>>b.release>>b.pose.p.x>>b.pose.p.y>>b.pose.p.z>>b.pose.q.w>>b.pose.q.x>>b.pose.q.y>>b.pose.q.z>>b.v.x>>b.v.y>>b.v.z>>b.w.x>>b.w.y>>b.w.z;release_end=std::max(release_end,b.release);}if(!in)return 3;
 out<<std::setprecision(12)<<"{\"engine\":\"wrench\",\"hz\":"<<hz<<",\"internalHz\":"<<internal_hz<<",\"setupSeconds\":"<<elapsed(setup)<<",\"masses\":[";for(int i=0;i<ns;i++)out<<(i?",":"")<<shapes[i].mass;out<<"],\"samples\":[";
 std::vector<double> timings,active_timings;double total=0,late=0,activation=0;int late_steps=0;long long contact_sum=0;bool first=true;
 for(int f=0;f<int(std::round(duration*hz));f++){
  auto start=Clock::now();for(auto&b:items)if(b.id<0&&b.release<=double(f)/hz+1e-9)b.id=world.add_body(shapes[b.shape],b.pose,b.v,b.w);activation+=elapsed(start);
  start=Clock::now();for(int step=0;step<internal_hz/hz;step++)world.step();const double cost=elapsed(start);timings.push_back(cost*1000);total+=cost;if(double(f)/hz>=release_end-1e-9){late+=cost;late_steps++;active_timings.push_back(cost*1000);}
  if((f+1)%std::max(1,hz/sample_hz)==0){double speed=0;for(const auto&b:items)if(b.id>=0){const auto s=world.state(b.id);speed=std::max(speed,length(s.v)+shapes[b.shape].radius*length(s.w));}const auto contacts=world.contacts();contact_sum+=contacts.size();out<<(first?"":",")<<"{\"t\":"<<double(f+1)/hz<<",\"maxSpeed\":"<<speed<<",\"contacts\":"<<contacts.size();first=false;
   if((f+1)%std::max(1,hz/geometry_hz)==0||f+1==int(std::round(duration*hz))){out<<",\"poses\":[";for(int i=0;i<n;i++){if(i)out<<",";if(items[i].id<0){out<<"null";continue;}const auto s=world.state(items[i].id);out<<"["<<s.pose.p.x<<","<<s.pose.p.y<<","<<s.pose.p.z<<","<<s.pose.q.w<<","<<s.pose.q.x<<","<<s.pose.q.y<<","<<s.pose.q.z<<"]";}out<<"]";}out<<"}";
  }
 }
 std::sort(active_timings.begin(),active_timings.end());auto active_quantile=[&](double q){return active_timings.empty()?0:active_timings[std::size_t(q*(active_timings.size()-1))];};
 std::sort(timings.begin(),timings.end());auto quantile=[&](double q){return timings[std::size_t(q*(timings.size()-1))];};const auto stats=world.solve_stats();
 out<<"],\"wallSeconds\":"<<total<<",\"activeSeconds\":"<<late<<",\"activeSteps\":"<<late_steps<<",\"activeTiming\":{\"p50Ms\":"<<active_quantile(.5)<<",\"p95Ms\":"<<active_quantile(.95)<<",\"p99Ms\":"<<active_quantile(.99)<<"},\"activationSeconds\":"<<activation<<",\"timing\":{\"p50Ms\":"<<quantile(.5)<<",\"p95Ms\":"<<quantile(.95)<<",\"p99Ms\":"<<quantile(.99)<<",\"maxMs\":"<<timings.back()<<"},\"guards\":"<<world.guard_count()<<",\"work\":{\"newton\":"<<stats.newton_iterations<<",\"factorizations\":"<<stats.factorizations<<",\"unconverged\":"<<stats.not_converged<<",\"contactSamples\":"<<contact_sum<<"}}\n";
 out.flush();if(!out){std::cerr<<"Result write failed: "<<argv[2]<<"\n";return 4;}
 std::cout<<"wrench "<<n<<" objects "<<hz<<"Hz "<<total<<"s guards "<<world.guard_count()<<"\n";
}
