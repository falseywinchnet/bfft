#include "engine.hpp"
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
 if(argc<4)return 2;
 bool trace=false;obligation::Parameters parameters;for(int i=4;i<argc;i++){std::string arg=argv[i];if(arg=="trace"||arg=="--trace")trace=true;else if(arg=="--no-equilibrium")parameters.certify_equilibrium=false;else if(arg.rfind("--contact-hz=",0)==0)parameters.contact_step=1/std::stod(arg.substr(13));else if(arg.rfind("--moving-tolerance=",0)==0)parameters.moving_velocity_tolerance=std::stod(arg.substr(19));else return 2;}
 std::ifstream in(argv[1]);std::ofstream out(argv[2]);const int hz=std::stoi(argv[3]);
 std::string magic;double density,friction,duration;Vec3 g;int ns,sample_hz,geometry_hz;
 in>>magic>>density>>friction>>g.x>>g.y>>g.z>>duration>>sample_hz>>geometry_hz>>ns;
 if(magic!="PACK1"||hz<60||hz%60)return 2;
 const auto setup=Clock::now();std::vector<Shape> shapes;
 for(int s=0;s<ns;s++){ShapeDesc d;d.density=density;d.friction=friction;d.restitution=0;d.rolling_resistance=0;int nh;in>>nh;for(int h=0;h<nh;h++){HullDesc hd;int nv;in>>nv;for(int v=0;v<nv;v++){Vec3 p;in>>p.x>>p.y>>p.z;hd.points.push_back(p);}d.hulls.push_back(hd);}shapes.push_back(cook(d));}
 obligation::Parameters p=parameters;p.gravity=g;p.ground_friction=friction;obligation::World world(p);
 int nw;in>>nw;for(int i=0;i<nw;i++){Vec3 c,s;in>>c.x>>c.y>>c.z>>s.x>>s.y>>s.z;ShapeDesc d;d.hulls.push_back({box(s)});d.friction=friction;d.restitution=0;d.rolling_resistance=0;Pose pose;pose.p=c;world.add_static_body(cook(d),pose);}
 int nf;in>>nf;for(int i=0;i<nf;i++){int shape;Pose pose;in>>shape>>pose.p.x>>pose.p.y>>pose.p.z>>pose.q.w>>pose.q.x>>pose.q.y>>pose.q.z;world.add_static_body(shapes[shape],pose);}
 int n;in>>n;std::vector<Item> items(n);double release_end=0;for(auto&b:items){in>>b.shape>>b.release>>b.pose.p.x>>b.pose.p.y>>b.pose.p.z>>b.pose.q.w>>b.pose.q.x>>b.pose.q.y>>b.pose.q.z>>b.v.x>>b.v.y>>b.v.z>>b.w.x>>b.w.y>>b.w.z;release_end=std::max(release_end,b.release);}if(!in)return 3;
 out<<std::setprecision(12)<<"{\"engine\":\"obligation\",\"hz\":"<<hz<<",\"settings\":{\"equilibrium\":"<<(p.certify_equilibrium?"true":"false")<<",\"contactStep\":"<<p.contact_step<<",\"skin\":"<<p.skin<<",\"movingTolerance\":"<<p.moving_velocity_tolerance<<"},\"setupSeconds\":"<<elapsed(setup)<<",\"masses\":[";for(int i=0;i<ns;i++)out<<(i?",":"")<<shapes[i].mass;out<<"],\"samples\":[";
 std::vector<double> timings;double total=0,late=0,activation=0;int late_steps=0;long long contact_sum=0;bool first=true;
 for(int f=0;f<int(std::round(duration*hz));f++){
  auto start=Clock::now();for(auto&b:items)if(b.id<0&&b.release<=double(f)/hz+1e-9)b.id=world.add_body(shapes[b.shape],b.pose,b.v,b.w);activation+=elapsed(start);
  start=Clock::now();world.advance(1.0/hz);const double cost=elapsed(start);timings.push_back(cost*1000);total+=cost;if(double(f)/hz>=release_end-1e-9){late+=cost;late_steps++;}
  if((f+1)%std::max(1,hz/sample_hz)==0){double speed=0;for(const auto&b:items)if(b.id>=0){const auto s=world.state(b.id);speed=std::max(speed,length(s.v)+shapes[b.shape].radius*length(s.w));}const auto contacts=world.contacts();contact_sum+=contacts.size();if(trace&&(f+1)%hz==0){const auto diagnostic=world.statistics();std::cerr<<"t "<<double(f+1)/hz<<" events "<<diagnostic.events<<" casts "<<diagnostic.casts<<" narrow "<<diagnostic.narrowphase<<" impact "<<diagnostic.impact_atomics<<" force "<<diagnostic.force_atomics<<" unconverged "<<diagnostic.unconverged_impacts<<"/"<<diagnostic.unconverged_forces<<" speed "<<speed<<std::endl;}out<<(first?"":",")<<"{\"t\":"<<double(f+1)/hz<<",\"maxSpeed\":"<<speed<<",\"contacts\":"<<contacts.size();first=false;
   if((f+1)%std::max(1,hz/geometry_hz)==0||f+1==int(std::round(duration*hz))){out<<",\"poses\":[";for(int i=0;i<n;i++){if(i)out<<",";if(items[i].id<0){out<<"null";continue;}const auto s=world.state(items[i].id);out<<"["<<s.pose.p.x<<","<<s.pose.p.y<<","<<s.pose.p.z<<","<<s.pose.q.w<<","<<s.pose.q.x<<","<<s.pose.q.y<<","<<s.pose.q.z<<"]";}out<<"]";}out<<"}";
  }
 }
 std::sort(timings.begin(),timings.end());auto quantile=[&](double q){return timings[std::size_t(q*(timings.size()-1))];};const auto stats=world.statistics();
 out<<"],\"wallSeconds\":"<<total<<",\"activeSeconds\":"<<late<<",\"activeSteps\":"<<late_steps<<",\"activationSeconds\":"<<activation<<",\"timing\":{\"p50Ms\":"<<quantile(.5)<<",\"p95Ms\":"<<quantile(.95)<<",\"p99Ms\":"<<quantile(.99)<<",\"maxMs\":"<<timings.back()<<"},\"guards\":"<<0<<",\"work\":{\"unconverged\":"<<(stats.unconverged_impacts+stats.unconverged_forces)<<",\"contactSamples\":"<<contact_sum<<",\"events\":"<<stats.events<<",\"casts\":"<<stats.casts<<",\"separationCertificates\":"<<stats.separation_certificates<<",\"castIterations\":"<<stats.cast_iterations<<",\"narrowphase\":"<<stats.narrowphase<<",\"impactAtomics\":"<<stats.impact_atomics<<",\"forceAtomics\":"<<stats.force_atomics<<",\"positionAtomics\":"<<stats.position_atomics<<",\"certificates\":"<<stats.equilibrium_certificates<<",\"discardedEnergy\":"<<stats.discarded_equilibrium_energy<<",\"largestGroup\":"<<stats.largest_group<<"},\"bodyEquilibrium\":[";
 for(int i=0;i<n;i++)out<<(i?",":"")<<(world.state(items[i].id).asleep?"true":"false");out<<"],\"forceResiduals\":[";
 for(int i=0;i<n;i++)out<<(i?",":"")<<world.equilibrium_residual(items[i].id);out<<"]}\n";
 std::cout<<"obligation "<<n<<" objects "<<hz<<"Hz "<<total<<"s guards "<<0<<"\n";
}
