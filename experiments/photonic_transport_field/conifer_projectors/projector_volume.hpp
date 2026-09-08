#pragma once
// Included after the native engine's Vec3/Ray/RGB and intersection definitions.
#include <complex>
#include <map>
#include <filesystem>
#include <random>
#include <functional>
namespace smoke_demo {
struct Counters { std::atomic<uint64_t> density{0}, visibility{0}, source_gathers{0}, fallbacks{0}, object_rays{0}, camera_rays{0}; };
inline Counters counts;
struct Response { double T=1; RGB S{}; };
inline Response compose(Response front,Response back){return{front.T*back.T,front.S+back.S*front.T};}
inline RGB apply(Response a,RGB background){return a.S+background*a.T;}
inline double sinc(double x){return std::abs(x)<1e-7?1-x*x/6:std::sin(x)/x;}
struct FourierSmoke {
 Vec3 lo{-4,.05,-2.5},hi{4,7.8,5};double amplitude=.62,albedo=.96,g=.5;
 struct Mode {Vec3 k;std::complex<double> c;};std::vector<Mode> modes;
 FourierSmoke(){using Key=std::array<int,3>;std::map<Key,std::complex<double>> coefficients;coefficients[{0,0,0}]=amplitude;
  auto factor=[&](Key k,double constant,double cosine,double phase){std::map<Key,std::complex<double>> next;for(auto [key,c]:coefficients){next[key]+=c*constant;for(int sign:{-1,1}){Key q=key;for(int j=0;j<3;++j)q[j]+=sign*k[j];next[q]+=c*(cosine*.5)*std::polar(1.,sign*phase);}}coefficients=std::move(next);};
  factor({1,0,0},.5,-.5,0);factor({0,1,0},.5,-.5,0);factor({0,0,1},.5,-.5,0);
  factor({2,1,1},.68,.32,.7);factor({1,-2,2},.65,.35,-.5);
  for(auto [k,c]:coefficients)if(std::abs(c)>1e-14)modes.push_back({{2*pi*k[0]/(hi.x-lo.x),2*pi*k[1]/(hi.y-lo.y),2*pi*k[2]/(hi.z-lo.z)},c});
 }
 bool chord(Ray ray,double limit,double& a,double& b)const{a=0;b=limit;for(int j=0;j<3;++j){double o=coordinate(ray.origin,j),d=coordinate(ray.direction,j),l=coordinate(lo,j),h=coordinate(hi,j);if(std::abs(d)<1e-12){if(o<l||o>h)return false;continue;}double u=(l-o)/d,v=(h-o)/d;if(u>v)std::swap(u,v);a=std::max(a,u);b=std::min(b,v);if(a>=b)return false;}return b>a;}
 double density(Vec3 p)const{++counts.density;double x=(p.x-lo.x)/(hi.x-lo.x),y=(p.y-lo.y)/(hi.y-lo.y),z=(p.z-lo.z)/(hi.z-lo.z);if(x<=0||x>=1||y<=0||y>=1||z<=0||z>=1)return 0;
  double sx=std::sin(pi*x),sy=std::sin(pi*y),sz=std::sin(pi*z);return amplitude*sx*sx*sy*sy*sz*sz*(.68+.32*std::cos(2*pi*(2*x+y+z)+.7))*(.65+.35*std::cos(2*pi*(x-2*y+2*z)-.5));}
 double spectral_density(Vec3 p)const{if(p.x<lo.x||p.x>hi.x||p.y<lo.y||p.y>hi.y||p.z<lo.z||p.z>hi.z)return 0;std::complex<double> sum{};for(auto m:modes)sum+=m.c*std::polar(1.,dot(m.k,p-lo));return sum.real();}

 struct March {
  const FourierSmoke& cloud;Vec3 p,step;std::array<std::complex<double>,5> state,rotation;
  March(const FourierSmoke&c,Vec3 initial,Vec3 delta):cloud(c),p(initial),step(delta){Vec3 q{2*pi*(p.x-c.lo.x)/(c.hi.x-c.lo.x),2*pi*(p.y-c.lo.y)/(c.hi.y-c.lo.y),2*pi*(p.z-c.lo.z)/(c.hi.z-c.lo.z)},d{2*pi*step.x/(c.hi.x-c.lo.x),2*pi*step.y/(c.hi.y-c.lo.y),2*pi*step.z/(c.hi.z-c.lo.z)};double phase[5]{q.x,q.y,q.z,2*q.x+q.y+q.z+.7,q.x-2*q.y+2*q.z-.5},advance[5]{d.x,d.y,d.z,2*d.x+d.y+d.z,d.x-2*d.y+2*d.z};for(int i=0;i<5;++i){state[i]=std::polar(1.,phase[i]);rotation[i]=std::polar(1.,advance[i]);}}
  double next(){++counts.density;double value=0;if(p.x>cloud.lo.x&&p.x<cloud.hi.x&&p.y>cloud.lo.y&&p.y<cloud.hi.y&&p.z>cloud.lo.z&&p.z<cloud.hi.z)value=cloud.amplitude*.125*(1-state[0].real())*(1-state[1].real())*(1-state[2].real())*(.68+.32*state[3].real())*(.65+.35*state[4].real());for(int i=0;i<5;++i)state[i]*=rotation[i];p=p+step;return std::max(0.,value);}
 };
 double tau(Ray ray,double limit,int samples=32)const{double a,b;if(!chord(ray,limit,a,b))return 0;double h=(b-a)/samples,sum=0;March march(*this,ray.origin+ray.direction*(a+.5*h),ray.direction*h);for(int i=0;i<samples;++i)sum+=march.next();return sum*h;}
 double exact_tau(Ray ray,double limit)const{double a,b;if(!chord(ray,limit,a,b))return 0;Vec3 middle=ray.origin+ray.direction*((a+b)*.5)-lo;double sum=0;for(auto m:modes)sum+=(m.c*std::polar(1.,dot(m.k,middle))).real()*(b-a)*sinc(dot(m.k,ray.direction)*(b-a)*.5);return std::max(0.,sum);}
 double phase(double cosine)const{return (1+3*g*cosine+5*g*g*(1.5*cosine*cosine-.5))/(4*pi);}
};
struct PointLight {Vec3 p{-6,8.5,-4.5};RGB intensity{220,220,220};};
inline bool unobstructed(const Scene&s,Vec3 a,Vec3 b,int ignore=-1){++counts.visibility;Vec3 d=b-a;double length=norm(d);return !first_hit(s,{a,d/length},length-2e-4,ignore).valid;}
struct ProjectorFrame {
 Vec3 p,forward,right,up;double u0=-1,u1=1,v0=-1,v1=1,far=20;
 ProjectorFrame(Vec3 center,Vec3 target):p(center){forward=unit(target-center);right=unit(cross(forward,std::abs(forward.y)>.99?Vec3{0,0,1}:Vec3{0,1,0}));up=cross(right,forward);}
 Vec3 direction(double u,double v)const{return unit(forward+right*u+up*v);}
 void fit(const FourierSmoke& cloud){u0=v0=1e9;u1=v1=-1e9;far=0;for(int i=0;i<8;++i){Vec3 q{(i&1)?cloud.hi.x:cloud.lo.x,(i&2)?cloud.hi.y:cloud.lo.y,(i&4)?cloud.hi.z:cloud.lo.z};Vec3 d=q-p;double z=dot(d,forward);if(z<=0)throw std::runtime_error("source plane does not cover volume");double u=dot(d,right)/z,v=dot(d,up)/z;u0=std::min(u0,u);u1=std::max(u1,u);v0=std::min(v0,v);v1=std::max(v1,v);far=std::max(far,z);}u0-=.02;u1+=.02;v0-=.02;v1+=.02;far+=.5;}
};
struct SourceProjector {
 ProjectorFrame frame;int side,depths;std::vector<float> tau;std::vector<float> first_depth;std::vector<int> primitive;
 SourceProjector(PointLight l,const FourierSmoke& cloud,int n=256,int d=96):frame(l.p,(cloud.lo+cloud.hi)*.5),side(n),depths(d){frame.fit(cloud);tau.resize(size_t(n)*n*(d+1));first_depth.resize(n*n);primitive.resize(n*n);}
 void bake(const Scene&s,const FourierSmoke& cloud){std::atomic<int> row{0};std::vector<std::thread> jobs;for(int worker=0;worker<6;++worker)jobs.emplace_back([&]{for(;;){int y=row.fetch_add(1);if(y>=side)break;for(int x=0;x<side;++x){int pixel=y*side+x;double u=frame.u0+(frame.u1-frame.u0)*x/(side-1),v=frame.v0+(frame.v1-frame.v0)*y/(side-1);Vec3 d=frame.direction(u,v);++counts.visibility;Hit h=first_hit(s,{frame.p,d});first_depth[pixel]=h.valid?dot(h.position-frame.p,frame.forward):1e9;primitive[pixel]=h.valid?h.primitive:-1;
  double step=frame.far/depths/dot(d,frame.forward),total=0;FourierSmoke::March march(cloud,frame.p+d*(.5*step),d*step);for(int z=1;z<=depths;++z){total+=step*march.next();tau[size_t(z)*side*side+pixel]=total;}}}});for(auto&t:jobs)t.join();}
 double transmission(Vec3 p)const{Vec3 delta=p-frame.p;double z=dot(delta,frame.forward);if(z<=0)return 1;double gx=(dot(delta,frame.right)/z-frame.u0)/(frame.u1-frame.u0)*(side-1),gy=(dot(delta,frame.up)/z-frame.v0)/(frame.v1-frame.v0)*(side-1),gz=z/frame.far*depths;if(gx<0||gy<0||gx>side-1||gy>side-1||gz>depths)return -1;
  int x=std::min(int(gx),side-2),y=std::min(int(gy),side-2),k=std::min(int(gz),depths-1);double sum=0;for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)sum+=tau[size_t(k+dz)*side*side+(y+dy)*side+x+dx]*(dx?gx-x:1-gx+x)*(dy?gy-y:1-gy+y)*(dz?gz-k:1-gz+k);return std::exp(-sum);}
 double illumination_visibility(const Scene&s,Vec3 p)const{++counts.source_gathers;Vec3 delta=p-frame.p;double z=dot(delta,frame.forward);if(z<=0)return 0;double gx=(dot(delta,frame.right)/z-frame.u0)/(frame.u1-frame.u0)*(side-1),gy=(dot(delta,frame.up)/z-frame.v0)/(frame.v1-frame.v0)*(side-1);
  if(gx<1||gy<1||gx>side-2||gy>side-2){++counts.fallbacks;return unobstructed(s,frame.p,p);}
  int x=int(gx),y=int(gy);int id=primitive[y*side+x];bool same=true;double near=1e9,far=0;for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int q=(y+dy)*side+x+dx;same&=primitive[q]==id;near=std::min(near,double(first_depth[q]));far=std::max(far,double(first_depth[q]));}
  // Sampling is not a conservative silhouette certificate; mixed tiles use exact visibility.
  if(same&&(z<near-.03||z>far+.03))return z<near?1:0;
  ++counts.fallbacks;return unobstructed(s,frame.p,p);
 }
};
// Positive degree-two angular scattering law. Ten stored moments, one redundant trace.
struct AngularRadiance {
 std::array<RGB,10> m{};
 void add(RGB value,Vec3 d,double weight){double b[10]{1,d.x,d.y,d.z,d.x*d.x,d.y*d.y,d.z*d.z,d.x*d.y,d.x*d.z,d.y*d.z};for(int i=0;i<10;++i)m[i]+=value*(weight*b[i]);}
 void scale(double x){for(auto&v:m)v=v*x;}
 RGB toward(Vec3 d,double g)const{RGB first=m[1]*d.x+m[2]*d.y+m[3]*d.z;RGB second=m[4]*(d.x*d.x)+m[5]*(d.y*d.y)+m[6]*(d.z*d.z)+m[7]*(2*d.x*d.y)+m[8]*(2*d.x*d.z)+m[9]*(2*d.y*d.z);return m[0]+first*(3*g)+(second*1.5+m[0]*(-.5))*(5*g*g);}
};
struct DiffuseGrid {
 Vec3 lo,hi;int n=24,face_side=32;std::vector<RGB> incident;std::vector<AngularRadiance> angular;
 DiffuseGrid(const FourierSmoke& c):lo(c.lo),hi(c.hi),incident(n*n*n),angular(n*n*n){}
 Vec3 point(int x,int y,int z)const{return{lo.x+(hi.x-lo.x)*x/(n-1),lo.y+(hi.y-lo.y)*y/(n-1),lo.z+(hi.z-lo.z)*z/(n-1)};}
 RGB gather(Vec3 p)const{double a[3]{(p.x-lo.x)/(hi.x-lo.x)*(n-1),(p.y-lo.y)/(hi.y-lo.y)*(n-1),(p.z-lo.z)/(hi.z-lo.z)*(n-1)};int k[3];for(int j=0;j<3;++j){a[j]=std::clamp(a[j],0.,double(n-1));k[j]=std::min(int(a[j]),n-2);}RGB sum{};for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x){double w=(x?a[0]-k[0]:1-a[0]+k[0])*(y?a[1]-k[1]:1-a[1]+k[1])*(z?a[2]-k[2]:1-a[2]+k[2]);sum+=incident[((k[2]+z)*n+k[1]+y)*n+k[0]+x]*w;}return sum;}
};
struct CaptureKey {uint64_t geometry=1,medium=1,light=1,material=1;uint64_t camera=0;auto operator<=>(const CaptureKey&)const=default;};
struct Pixel {float T=1,S[3]{},background[3]{},depth=0;};
static_assert(sizeof(Pixel)==32);
struct Sheet {int width=0,height=0;CaptureKey key;std::vector<Pixel> pixels;};
struct CaptureRegistry {
 std::map<CaptureKey,std::shared_ptr<const Sheet>> entries;uint64_t builds=0,hits=0;
 const Sheet& get(CaptureKey key,const std::function<Sheet()>&build){auto it=entries.find(key);if(it!=entries.end()){++hits;return *it->second;}Sheet s=build();s.key=key;auto value=std::make_shared<const Sheet>(std::move(s));entries[key]=value;++builds;return *value;}
};
inline uint64_t checksum(const Sheet&s){uint64_t h=1469598103934665603ULL;for(size_t i=0;i<s.pixels.size()*sizeof(Pixel);++i){h^=reinterpret_cast<const unsigned char*>(s.pixels.data())[i];h*=1099511628211ULL;}return h;}
inline void save_sheet(const Sheet&s,const std::string&path){std::ofstream f(path,std::ios::binary);uint64_t header[]{0x3154454548534650ULL,uint64_t(s.width),uint64_t(s.height),checksum(s),s.key.geometry,s.key.medium,s.key.light,s.key.material,s.key.camera};f.write((char*)header,sizeof header);f.write((char*)s.pixels.data(),s.pixels.size()*sizeof(Pixel));if(!f)throw std::runtime_error("sheet write failed");}
inline Sheet load_sheet(const std::string&path){std::ifstream f(path,std::ios::binary);uint64_t header[9];f.read((char*)header,sizeof header);if(!f||header[0]!=0x3154454548534650ULL||header[1]==0||header[2]==0||header[1]>8192||header[2]>8192)throw std::runtime_error("bad sheet header");Sheet s;s.width=header[1];s.height=header[2];s.key={header[4],header[5],header[6],header[7],header[8]};s.pixels.resize(size_t(s.width)*s.height);f.read((char*)s.pixels.data(),s.pixels.size()*sizeof(Pixel));if(!f||checksum(s)!=header[3])throw std::runtime_error("sheet integrity failed");return s;}
}
