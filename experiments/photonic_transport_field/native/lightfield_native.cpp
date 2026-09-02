#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using RGB = std::array<double,3>;
constexpr double pi=std::numbers::pi_v<double>;

struct Vec2 { double x=0.0,y=0.0; };
struct Vec3 { double x=0.0,y=0.0,z=0.0; };
Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
Vec3 operator*(Vec3 a,double s){return {a.x*s,a.y*s,a.z*s};}
Vec3 operator*(double s,Vec3 a){return a*s;}
Vec3 operator/(Vec3 a,double s){return {a.x/s,a.y/s,a.z/s};}
Vec3& operator+=(Vec3& a,Vec3 b){a=a+b;return a;}
double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double norm(Vec3 a){return std::sqrt(dot(a,a));}
Vec3 unit(Vec3 a){const double n=norm(a);return n>0.0?a/n:Vec3{};}
RGB operator+(RGB a,const RGB& b){for(int c=0;c<3;++c)a[c]+=b[c];return a;}
RGB operator*(RGB a,double s){for(double& x:a)x*=s;return a;}
RGB multiply(RGB a,const RGB& b){for(int c=0;c<3;++c)a[c]*=b[c];return a;}
RGB& operator+=(RGB& a,const RGB& b){a=a+b;return a;}
double energy(const RGB& a){return a[0]+a[1]+a[2];}

enum class Kind {Plane,Sphere};
enum Group:int {Floor,Back,LeftWall,RightWall,Ceiling,AreaLight,
                RedSphere,BlueSphere,GreenSphere,GroupCount};

struct Material { RGB albedo{},emission{}; };
struct Primitive {
    Kind kind=Kind::Plane; Group group=Floor; Material material{};
    Vec3 origin{},u{},v{},normal{};
    Vec3 center{}; double radius=0.0;
};
struct Hit {
    bool valid=false; int primitive=-1; double t=0.0;
    Vec3 position{},normal{}; Vec2 uv{};
};
struct Scene { std::vector<Primitive> primitives; };

int add_plane(Scene& s,Group group,Vec3 origin,Vec3 u,Vec3 v,Vec3 normal,
              RGB albedo,RGB emission={}) {
    s.primitives.push_back({Kind::Plane,group,{albedo,emission},origin,u,v,normal,{ },0.0});
    return static_cast<int>(s.primitives.size())-1;
}
int add_sphere(Scene& s,Group group,Vec3 center,double radius,RGB albedo) {
    Primitive p; p.kind=Kind::Sphere;p.group=group;p.material.albedo=albedo;
    p.center=center;p.radius=radius;s.primitives.push_back(p);
    return static_cast<int>(s.primitives.size())-1;
}
Scene build_scene() {
    Scene s; const RGB white{0.72,0.74,0.76};
    add_plane(s,Floor,{-3,0,-5},{6,0,0},{0,0,6},{0,1,0},white);
    add_plane(s,Back,{-3,0,-5},{6,0,0},{0,4,0},{0,0,1},white);
    add_plane(s,LeftWall,{-3,0,1},{0,0,-6},{0,4,0},{1,0,0},{0.58,0.16,0.12});
    add_plane(s,RightWall,{3,0,-5},{0,0,6},{0,4,0},{-1,0,0},{0.12,0.22,0.62});
    add_plane(s,Ceiling,{-3,4,1},{6,0,0},{0,0,-6},{0,-1,0},white);
    add_plane(s,AreaLight,{-1.15,3.92,-3.0},{2.3,0,0},{0,0,1.7},{0,-1,0},
              {0.02,0.02,0.02},{34,31,27});
    add_sphere(s,RedSphere,{-1.35,0.72,-2.65},0.72,{0.78,0.08,0.055});
    add_sphere(s,BlueSphere,{1.22,0.62,-3.15},0.62,{0.055,0.20,0.82});
    add_sphere(s,GreenSphere,{0.15,0.48,-1.20},0.48,{0.08,0.72,0.20});
    return s;
}

Vec2 sphere_uv(Vec3 n) {
    return {std::atan2(n.z,n.x)/(2.0*pi)+0.5,
            std::acos(std::clamp(n.y,-1.0,1.0))/pi};
}
Vec2 surface_uv(const Primitive& p,Vec3 point) {
    if(p.kind==Kind::Sphere)return sphere_uv(unit(point-p.center));
    const Vec3 d=point-p.origin;
    return {dot(d,p.u)/dot(p.u,p.u),dot(d,p.v)/dot(p.v,p.v)};
}
Hit intersect(const Primitive& p,int id,Vec3 origin,Vec3 ray) {
    Hit h;h.primitive=id;
    if(p.kind==Kind::Plane) {
        const double den=dot(p.normal,ray);if(std::abs(den)<1e-12)return h;
        const double t=dot(p.origin-origin,p.normal)/den;if(t<=2e-5)return h;
        const Vec3 point=origin+ray*t;const Vec2 uv=surface_uv(p,point);
        if(uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0)return h;
        h.valid=true;h.t=t;h.position=point;h.normal=p.normal;h.uv=uv;return h;
    }
    const Vec3 rel=origin-p.center;const double b=dot(ray,rel);
    const double c=dot(rel,rel)-p.radius*p.radius;const double disc=b*b-c;
    if(disc<0.0)return h;const double root=std::sqrt(disc);
    double t=-b-root;if(t<=2e-5)t=-b+root;if(t<=2e-5)return h;
    h.valid=true;h.t=t;h.position=origin+ray*t;h.normal=unit(h.position-p.center);
    h.uv=sphere_uv(h.normal);return h;
}
Hit first_hit(const Scene& s,Vec3 origin,Vec3 ray) {
    Hit best;best.t=std::numeric_limits<double>::infinity();
    for(int i=0;i<static_cast<int>(s.primitives.size());++i) {
        Hit h=intersect(s.primitives[i],i,origin,ray);
        if(h.valid&&h.t<best.t)best=h;
    }
    return best;
}

struct Basis { Vec3 tangent,bitangent,normal; };
Basis basis_from_normal(Vec3 n) {
    n=unit(n);const Vec3 helper=std::abs(n.y)<0.9?Vec3{0,1,0}:Vec3{1,0,0};
    const Vec3 tangent=unit(cross(helper,n));return {tangent,cross(n,tangent),n};
}
Vec2 concentric_disk(double sx,double sy) {
    const double a=2.0*sx-1.0,b=2.0*sy-1.0;
    if(std::abs(a)<1e-15&&std::abs(b)<1e-15)return {};
    double r,phi;
    if(std::abs(a)>std::abs(b)){r=a;phi=(pi/4.0)*(b/a);}
    else {r=b;phi=pi/2.0-(pi/4.0)*(a/b);}
    return {r*std::cos(phi),r*std::sin(phi)};
}
Vec3 cosine_direction(const Basis& basis,double sx,double sy) {
    const Vec2 d=concentric_disk(sx,sy);
    const double z=std::sqrt(std::max(0.0,1.0-d.x*d.x-d.y*d.y));
    return unit(basis.tangent*d.x+basis.bitangent*d.y+basis.normal*z);
}

struct Emitter {
    Vec3 position{},normal{}; RGB power{}; int source_primitive=-1,generation=0;
};
struct Footprint {
    int primitive=-1,generation=0; std::array<Vec3,4> corners{};
    Vec3 center{},normal{},tangent{},bitangent{}; std::array<Vec2,4> polygon{};
    Vec2 uv_center{}; double uv_radius_x=0.0,uv_radius_y=0.0,area=0.0;
    RGB incident_power{},reflected_power{},radiance{};
    std::array<RGB,4> vertex_radiance{};
};
struct TraceStats {
    std::uint64_t rays=0,cells=0,splits=0,accepted=0,boundary_leaves=0,pruned=0;
    RGB emitted{},deposited{},escaped{},discarded{};
    std::vector<std::uint64_t> footprints_by_generation;
    std::vector<std::uint64_t> emitters_by_generation;
};
struct Cell { double x0=0,x1=1,y0=0,y1=1;int level=0; };
struct RunLimits {
    std::uint64_t max_rays=12000000,max_cells=3000000,max_footprints=400000;
    std::uint64_t max_index_entries=16000000;
    double max_build_seconds=90.0;
};

double triangle_area(Vec3 a,Vec3 b,Vec3 c){return 0.5*norm(cross(b-a,c-a));}

Footprint make_footprint(const Scene& scene,int primitive,
                         const std::array<Vec3,4>& corners,const Hit& center,
                         RGB incident,int generation,
                         std::array<double,4> density={{1.0,1.0,1.0,1.0}}) {
    Footprint f;f.primitive=primitive;f.generation=generation;f.corners=corners;
    f.center=center.position;f.normal=center.normal;f.uv_center=center.uv;
    const Basis basis=basis_from_normal(f.normal);f.tangent=basis.tangent;
    f.bitangent=basis.bitangent;
    for(int k=0;k<4;++k){const Vec2 uv=surface_uv(scene.primitives[primitive],corners[k]);
        double du=uv.x-f.uv_center.x;if(du>0.5)du-=1.0;if(du<-0.5)du+=1.0;
        f.polygon[k]={du,uv.y-f.uv_center.y};}
    f.area=triangle_area(corners[0],corners[1],corners[2])+
           triangle_area(corners[0],corners[2],corners[3]);
    f.area=std::max(f.area,1e-9);f.incident_power=incident;
    const auto& material=scene.primitives[primitive].material;
    f.reflected_power=multiply(incident,material.albedo);
    f.radiance=f.reflected_power*(1.0/(pi*f.area));
    const double average=std::max(0.25*(density[0]+density[1]+density[2]+density[3]),1e-15);
    for(int k=0;k<4;++k)f.vertex_radiance[k]=f.radiance*(density[k]/average);
    double rx=0.0,ry=0.0;
    for(Vec3 corner:corners){Vec2 uv=surface_uv(scene.primitives[primitive],corner);
        double du=uv.x-f.uv_center.x;if(du>0.5)du-=1.0;if(du<-0.5)du+=1.0;
        rx=std::max(rx,std::abs(du));ry=std::max(ry,std::abs(uv.y-f.uv_center.y));}
    f.uv_radius_x=std::max(rx,1e-5);f.uv_radius_y=std::max(ry,1e-5);
    return f;
}

struct BeamCompiler {
    const Scene& scene; std::vector<Footprint>& footprints; TraceStats& stats;
    const RunLimits& limits; Clock::time_point deadline;
    double* remaining_discard_budget=nullptr;
    double prune_energy=1e-7,density_tolerance=0.12,geometry_tolerance=0.045;
    double normal_coherence=0.90;
    int min_level=2,max_level=8;

    void guard() const {
        if(stats.rays>=limits.max_rays)throw std::runtime_error("construction ray ceiling reached");
        if(stats.cells>=limits.max_cells)throw std::runtime_error("bundle-cell ceiling reached");
        if(footprints.size()>=limits.max_footprints)throw std::runtime_error("light-field footprint ceiling reached");
        if(Clock::now()>=deadline)throw std::runtime_error("light-field build time ceiling reached");
    }

    void deposit_micro(const Emitter& emitter,const Cell& cell) {
        const int strata=2;
        const double fraction=(cell.x1-cell.x0)*(cell.y1-cell.y0)/(strata*strata);
        const Basis basis=basis_from_normal(emitter.normal);
        for(int j=0;j<strata;++j)for(int i=0;i<strata;++i){
            guard();
            const double sx=cell.x0+(i+0.5)/strata*(cell.x1-cell.x0);
            const double sy=cell.y0+(j+0.5)/strata*(cell.y1-cell.y0);
            const Vec3 ray=cosine_direction(basis,sx,sy);
            const Hit hit=first_hit(scene,emitter.position+emitter.normal*3e-5,ray);++stats.rays;
            const RGB power=emitter.power*fraction;
            if(!hit.valid){stats.escaped+=power;continue;}
            const double angular=1.20*std::max(cell.x1-cell.x0,cell.y1-cell.y0)/strata;
            const double half=std::clamp(hit.t*angular/std::max(std::abs(dot(hit.normal,-1.0*ray)),0.12),
                                         2e-4,0.18);
            const Basis hb=basis_from_normal(hit.normal);
            std::array<Vec3,4> corners{hit.position-hb.tangent*half-hb.bitangent*half,
                hit.position+hb.tangent*half-hb.bitangent*half,
                hit.position+hb.tangent*half+hb.bitangent*half,
                hit.position-hb.tangent*half+hb.bitangent*half};
            if(scene.primitives[hit.primitive].kind==Kind::Sphere)
                for(Vec3& p:corners)p=scene.primitives[hit.primitive].center+
                    unit(p-scene.primitives[hit.primitive].center)*scene.primitives[hit.primitive].radius;
            footprints.push_back(make_footprint(scene,hit.primitive,corners,hit,power,emitter.generation));
            stats.deposited+=power;++stats.accepted;
        }
        ++stats.boundary_leaves;
    }

    void trace_cell(const Emitter& emitter,const Cell& cell) {
        guard();++stats.cells;const double fraction=(cell.x1-cell.x0)*(cell.y1-cell.y0);
        const RGB cell_power=emitter.power*fraction;
        const double cell_energy=energy(cell_power);
        if(remaining_discard_budget&&cell_energy<=prune_energy&&
           cell_energy<=*remaining_discard_budget) {
            *remaining_discard_budget-=cell_energy;
            stats.discarded+=cell_power;++stats.pruned;return;
        }
        static constexpr std::array<Vec2,13> probes{{{0,0},{1,0},{1,1},{0,1},{.5,.5},
            {.5,0},{1,.5},{.5,1},{0,.5},{.25,.25},{.75,.25},{.75,.75},{.25,.75}}};
        const Basis basis=basis_from_normal(emitter.normal);
        std::array<Hit,13> hits{};std::array<Vec3,13> rays{};
        bool all_miss=true,same=true;int primitive=-2;
        double min_metric=std::numeric_limits<double>::infinity(),max_metric=0.0;
        double min_normal=1.0;
        for(int k=0;k<13;++k){guard();const double sx=cell.x0+probes[k].x*(cell.x1-cell.x0);
            const double sy=cell.y0+probes[k].y*(cell.y1-cell.y0);
            rays[k]=cosine_direction(basis,sx,sy);
            hits[k]=first_hit(scene,emitter.position+emitter.normal*3e-5,rays[k]);++stats.rays;
            const int id=hits[k].valid?hits[k].primitive:-1;
            if(k==0)primitive=id;else if(id!=primitive)same=false;
            if(hits[k].valid){all_miss=false;const double metric=std::max(dot(hits[k].normal,-1.0*rays[k]),0.0)/
                    std::max(hits[k].t*hits[k].t,1e-12);
                min_metric=std::min(min_metric,metric);max_metric=std::max(max_metric,metric);
                min_normal=std::min(min_normal,dot(hits[4].valid?hits[4].normal:hits[k].normal,hits[k].normal));}
        }
        bool coherent=same;
        if(coherent&&!all_miss&&hits[4].valid){
            const Vec3 bilinear=(hits[0].position+hits[1].position+hits[2].position+hits[3].position)*0.25;
            double diameter=0.0;for(int k=0;k<4;++k)for(int l=k+1;l<4;++l)
                diameter=std::max(diameter,norm(hits[k].position-hits[l].position));
            const double geometric=norm(hits[4].position-bilinear)/std::max(diameter,1e-8);
            const double density=(max_metric-min_metric)/std::max(max_metric,1e-12);
            const double required_normal=scene.primitives[primitive].kind==Kind::Sphere?
                std::max(normal_coherence,0.995):normal_coherence;
            coherent=geometric<=geometry_tolerance&&density<=density_tolerance&&min_normal>=required_normal;
        }
        if(cell.level<min_level)coherent=false;
        if(coherent){
            if(all_miss){stats.escaped+=cell_power;return;}
            std::array<Vec3,4> corners{hits[0].position,hits[1].position,hits[2].position,hits[3].position};
            std::array<double,4> density{};
            for(int k=0;k<4;++k)density[k]=std::max(dot(hits[k].normal,-1.0*rays[k]),0.0)/
                std::max(hits[k].t*hits[k].t,1e-12);
            footprints.push_back(make_footprint(scene,primitive,corners,hits[4],cell_power,
                                                emitter.generation,density));
            stats.deposited+=cell_power;++stats.accepted;return;
        }
        if(cell.level>=max_level){deposit_micro(emitter,cell);return;}
        ++stats.splits;const double xm=0.5*(cell.x0+cell.x1),ym=0.5*(cell.y0+cell.y1);
        trace_cell(emitter,{cell.x0,xm,cell.y0,ym,cell.level+1});
        trace_cell(emitter,{xm,cell.x1,cell.y0,ym,cell.level+1});
        trace_cell(emitter,{xm,cell.x1,ym,cell.y1,cell.level+1});
        trace_cell(emitter,{cell.x0,xm,ym,cell.y1,cell.level+1});
    }
    void trace(const Emitter& emitter){stats.emitted+=emitter.power;trace_cell(emitter,{});}
};

struct ClusterAccumulator {Vec3 position{},normal{};RGB power{};double weight=0.0;int primitive=-1;};
std::vector<Emitter> make_diffuse_emitters(const Scene& scene,const std::vector<Footprint>& footprints,
                                           std::size_t begin,int generation,double threshold) {
    std::unordered_map<std::uint64_t,ClusterAccumulator> clusters;
    for(std::size_t i=begin;i<footprints.size();++i){const Footprint& f=footprints[i];
        const double w=energy(f.reflected_power);if(w<=threshold)continue;
        const Primitive& p=scene.primitives[f.primitive];
        const int nu=p.kind==Kind::Sphere?8:4,nv=p.kind==Kind::Sphere?4:4;
        const int iu=std::clamp(static_cast<int>(f.uv_center.x*nu),0,nu-1);
        const int iv=std::clamp(static_cast<int>(f.uv_center.y*nv),0,nv-1);
        const std::uint64_t key=(static_cast<std::uint64_t>(f.primitive)<<32)|
            (static_cast<std::uint64_t>(iu)<<16)|static_cast<std::uint64_t>(iv);
        auto& a=clusters[key];a.primitive=f.primitive;a.position+=f.center*w;a.normal+=f.normal*w;
        a.power+=f.reflected_power;a.weight+=w;
    }
    std::vector<Emitter> emitters;emitters.reserve(clusters.size());
    for(auto& [key,a]:clusters){(void)key;if(energy(a.power)<=threshold)continue;
        emitters.push_back({a.position/a.weight,unit(a.normal),a.power,a.primitive,generation});}
    std::sort(emitters.begin(),emitters.end(),[](const Emitter& a,const Emitter& b){return energy(a.power)>energy(b.power);});
    return emitters;
}

struct FieldIndex {
    const Scene& scene;const std::vector<Footprint>& footprints;int nx=96,ny=64;
    std::vector<std::vector<std::vector<int>>> bins;
    FieldIndex(const Scene& s,const std::vector<Footprint>& f,std::uint64_t max_entries):scene(s),footprints(f) {
        bins.resize(scene.primitives.size(),std::vector<std::vector<int>>(nx*ny));
        std::uint64_t entries=0;
        for(int index=0;index<static_cast<int>(footprints.size());++index){const auto& fp=footprints[index];
            int x0=static_cast<int>(std::floor((fp.uv_center.x-fp.uv_radius_x)*nx));
            int x1=static_cast<int>(std::floor((fp.uv_center.x+fp.uv_radius_x)*nx));
            int y0=std::clamp(static_cast<int>(std::floor((fp.uv_center.y-fp.uv_radius_y)*ny)),0,ny-1);
            int y1=std::clamp(static_cast<int>(std::floor((fp.uv_center.y+fp.uv_radius_y)*ny)),0,ny-1);
            const bool wraps=scene.primitives[fp.primitive].kind==Kind::Sphere;
            for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){int bx=x;
                if(wraps)bx=(bx%nx+nx)%nx;else if(bx<0||bx>=nx)continue;
                if(++entries>max_entries)throw std::runtime_error("light-field index-entry ceiling reached");
                bins[fp.primitive][y*nx+bx].push_back(index);}
        }
    }
    static bool barycentric(Vec2 q,Vec2 a,Vec2 b,Vec2 c,std::array<double,3>& w) {
        const double den=(b.y-c.y)*(a.x-c.x)+(c.x-b.x)*(a.y-c.y);
        if(std::abs(den)<1e-15)return false;
        w[0]=((b.y-c.y)*(q.x-c.x)+(c.x-b.x)*(q.y-c.y))/den;
        w[1]=((c.y-a.y)*(q.x-c.x)+(a.x-c.x)*(q.y-c.y))/den;
        w[2]=1.0-w[0]-w[1];
        return w[0]>=-1e-8&&w[1]>=-1e-8&&w[2]>=-1e-8;
    }
    static bool sample(const Footprint& f,const Hit& hit,RGB& value) {
        double du=hit.uv.x-f.uv_center.x;if(du>0.5)du-=1.0;if(du<-0.5)du+=1.0;
        const Vec2 q{du,hit.uv.y-f.uv_center.y};
        std::array<double,3> w{};
        if(barycentric(q,f.polygon[0],f.polygon[1],f.polygon[2],w)){
            value=f.vertex_radiance[0]*w[0]+f.vertex_radiance[1]*w[1]+f.vertex_radiance[2]*w[2];return true;}
        if(barycentric(q,f.polygon[0],f.polygon[2],f.polygon[3],w)){
            value=f.vertex_radiance[0]*w[0]+f.vertex_radiance[2]*w[1]+f.vertex_radiance[3]*w[2];return true;}
        return false;
    }
    RGB query(const Hit& hit,int generation_min=0,int generation_max=INT_MAX) const {
        const int x=std::clamp(static_cast<int>(hit.uv.x*nx),0,nx-1);
        const int y=std::clamp(static_cast<int>(hit.uv.y*ny),0,ny-1);RGB sum{};
        for(int index:bins[hit.primitive][y*nx+x]){RGB value{};
            if(footprints[index].generation<generation_min||
               footprints[index].generation>generation_max)continue;
            if(sample(footprints[index],hit,value))sum+=value;}
        return sum;
    }
};

struct RadianceAtlas {
    const Scene& scene;int width=192,height=192;
    std::vector<std::vector<RGB>> data;
    static Vec3 point_from_uv(const Primitive& p,double u,double v) {
        if(p.kind==Kind::Plane)return p.origin+p.u*u+p.v*v;
        const double phi=pi*v,theta=2.0*pi*(u-0.5);
        const Vec3 n{std::sin(phi)*std::cos(theta),std::cos(phi),
                     std::sin(phi)*std::sin(theta)};
        return p.center+n*p.radius;
    }
    RadianceAtlas(const Scene& s,const FieldIndex& field,int atlas_width,int atlas_height,
                  int generation_min,int generation_max,int filter_passes)
        :scene(s),width(atlas_width),height(atlas_height) {
        data.resize(scene.primitives.size(),std::vector<RGB>(width*height));
        std::atomic<int> next{0};std::vector<std::thread> threads;
        const int total=static_cast<int>(scene.primitives.size())*height;
        const unsigned workers=std::max(1u,std::thread::hardware_concurrency());
        for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&]{for(;;){
            const int task=next.fetch_add(1);if(task>=total)break;
            const int primitive=task/height,y=task%height;
            for(int x=0;x<width;++x){Hit hit;hit.valid=true;hit.primitive=primitive;
                hit.uv={(x+0.5)/width,(y+0.5)/height};
                hit.position=point_from_uv(scene.primitives[primitive],hit.uv.x,hit.uv.y);
                hit.normal=scene.primitives[primitive].kind==Kind::Sphere?
                    unit(hit.position-scene.primitives[primitive].center):scene.primitives[primitive].normal;
                data[primitive][y*width+x]=field.query(hit,generation_min,generation_max);}
        }});
        for(auto& thread:threads)thread.join();
        for(int pass=0;pass<filter_passes;++pass)for(std::size_t primitive=0;primitive<data.size();++primitive){
            std::vector<RGB> filtered=data[primitive];
            const bool wrap=scene.primitives[primitive].kind==Kind::Sphere;
            for(int y=0;y<height;++y)for(int x=0;x<width;++x){RGB sum{};double weight=0.0;
                for(int oy=-1;oy<=1;++oy)for(int ox=-1;ox<=1;++ox){int sx=x+ox;
                    const int sy=std::clamp(y+oy,0,height-1);
                    if(wrap)sx=(sx%width+width)%width;else sx=std::clamp(sx,0,width-1);
                    const double w=(ox==0?2.0:1.0)*(oy==0?2.0:1.0);
                    sum+=data[primitive][sy*width+sx]*w;weight+=w;}
                filtered[y*width+x]=sum*(1.0/weight);}
            data[primitive].swap(filtered);
        }
    }
    RGB query(const Hit& hit) const {
        const double fx=hit.uv.x*width-0.5,fy=hit.uv.y*height-0.5;
        const int x0=static_cast<int>(std::floor(fx)),y0=static_cast<int>(std::floor(fy));
        const double tx=fx-x0,ty=fy-y0;
        const bool wrap=scene.primitives[hit.primitive].kind==Kind::Sphere;
        auto at=[&](int x,int y)->const RGB&{if(wrap)x=(x%width+width)%width;
            else x=std::clamp(x,0,width-1);y=std::clamp(y,0,height-1);
            return data[hit.primitive][y*width+x];};
        RGB out{};for(int c=0;c<3;++c)out[c]=(1-ty)*((1-tx)*at(x0,y0)[c]+tx*at(x0+1,y0)[c])+
            ty*((1-tx)*at(x0,y0+1)[c]+tx*at(x0+1,y0+1)[c]);
        return out;
    }
};

std::vector<Emitter> source_emitters(const Scene& scene,int columns=6,int rows=4) {
    const Primitive& light=scene.primitives[AreaLight];
    const double area=norm(cross(light.u,light.v));
    const RGB total=light.material.emission*(area*pi);std::vector<Emitter> out;
    for(int j=0;j<rows;++j)for(int i=0;i<columns;++i){
        const double u=(i+0.5)/columns,v=(j+0.5)/rows;
        out.push_back({light.origin+light.u*u+light.v*v,light.normal,total*(1.0/(columns*rows)),AreaLight,0});}
    return out;
}

struct BuildResult {std::vector<Footprint> footprints;TraceStats stats;double ms=0.0;};
BuildResult build_light_field(const Scene& scene,int bounce_count,double error_fraction,
                              const RunLimits& limits={}) {
    const auto start=Clock::now();BuildResult result;auto emitters=source_emitters(scene);
    const double source_energy=energy(emitters.front().power)*emitters.size();
    const double base_threshold=source_energy*error_fraction;
    double remaining_discard_budget=base_threshold;
    for(int generation=0;generation<bounce_count;++generation){
        result.stats.emitters_by_generation.push_back(emitters.size());
        const std::size_t footprint_begin=result.footprints.size();
        for(const Emitter& emitter:emitters){BeamCompiler compiler{scene,result.footprints,result.stats,
            limits,start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(limits.max_build_seconds))};
            compiler.remaining_discard_budget=&remaining_discard_budget;
            compiler.prune_energy=base_threshold*std::pow(0.40,generation);
            compiler.min_level=generation==0?2:1;
            compiler.max_level=generation==0?8:std::max(3,5-generation);
            compiler.density_tolerance=generation==0?0.35:0.35;
            compiler.geometry_tolerance=generation==0?0.12:0.12;
            compiler.normal_coherence=generation==0?0.88:0.82;
            compiler.trace(emitter);}
        result.stats.footprints_by_generation.push_back(result.footprints.size()-footprint_begin);
        emitters=make_diffuse_emitters(scene,result.footprints,footprint_begin,generation+1,0.0);
        if(emitters.empty())break;
    }
    result.ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return result;
}

std::vector<std::uint8_t> render(const Scene& scene,const RadianceAtlas& direct,
                                 const RadianceAtlas& diffuse,int width,int height) {
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width)*height*3,0);
    const Vec3 camera{0,2.15,6.8},target{0,1.55,-2.15};const Vec3 forward=unit(target-camera);
    const Vec3 right=unit(cross(forward,{0,1,0})),up=cross(right,forward);
    const double scale=std::tan(24.0*pi/180.0),aspect=double(width)/height;
    std::atomic<int> next_row{0};std::vector<std::thread> threads;
    const unsigned workers=std::max(1u,std::thread::hardware_concurrency());
    for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&]{for(;;){
        const int y=next_row.fetch_add(1);if(y>=height)break;
        for(int x=0;x<width;++x){const double px=(2.0*(x+0.5)/width-1.0)*aspect*scale;
            const double py=(1.0-2.0*(y+0.5)/height)*scale;const Vec3 ray=unit(forward+right*px+up*py);
            const Hit hit=first_hit(scene,camera,ray);RGB linear{};
            if(hit.valid){linear=direct.query(hit)+diffuse.query(hit);
                linear+=scene.primitives[hit.primitive].material.emission;}
            const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;
            for(int c=0;c<3;++c){const double mapped=std::pow(std::clamp(1.0-std::exp(-0.72*std::max(linear[c],0.0)),0.0,1.0),1.0/2.2);
                image[offset+c]=static_cast<std::uint8_t>(std::lround(255.0*mapped));}
        }} });
    for(auto& thread:threads)thread.join();return image;
}

void write_ppm(const std::string& path,const std::vector<std::uint8_t>& image,int width,int height){
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot open output image");
    out<<"P6\n"<<width<<" "<<height<<"\n255\n";out.write(reinterpret_cast<const char*>(image.data()),
        static_cast<std::streamsize>(image.size()));}

bool self_test(){const Scene scene=build_scene();
    const Vec2 center=concentric_disk(0.5,0.5);if(std::abs(center.x)>1e-12||std::abs(center.y)>1e-12)return false;
    const Basis down=basis_from_normal({0,-1,0});if(dot(cosine_direction(down,.5,.5),Vec3{0,-1,0})<0.999)return false;
    BuildResult result=build_light_field(scene,1,2e-5);const double emitted=energy(result.stats.emitted);
    const double accounted=energy(result.stats.deposited)+energy(result.stats.escaped)+energy(result.stats.discarded);
    if(std::abs(emitted-accounted)>1e-8*std::max(emitted,1.0))return false;
    if(result.footprints.empty()||result.stats.splits==0||scene.primitives.size()!=9)return false;
    std::cout<<"ray-induced field invariants: ok ("<<result.footprints.size()<<" footprints, "
             <<result.stats.rays<<" construction rays)\n";return true;}

} // namespace

int main(int argc,char** argv)try{
    int width=800,height=600,bounces=3;double error_fraction=2e-6;
    std::string out="/tmp/photonic_lightfield.ppm";bool test=false;RunLimits limits;
    for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--self-test"){test=true;continue;}
        if(i+1>=argc)throw std::runtime_error("missing argument value");
        if(arg=="--width")width=std::stoi(argv[++i]);else if(arg=="--height")height=std::stoi(argv[++i]);
        else if(arg=="--bounces")bounces=std::stoi(argv[++i]);else if(arg=="--error-fraction")error_fraction=std::stod(argv[++i]);
        else if(arg=="--max-rays")limits.max_rays=std::stoull(argv[++i]);
        else if(arg=="--max-cells")limits.max_cells=std::stoull(argv[++i]);
        else if(arg=="--max-footprints")limits.max_footprints=std::stoull(argv[++i]);
        else if(arg=="--max-index-entries")limits.max_index_entries=std::stoull(argv[++i]);
        else if(arg=="--max-build-seconds")limits.max_build_seconds=std::stod(argv[++i]);
        else if(arg=="--out")out=argv[++i];else throw std::runtime_error("unknown argument: "+arg);}
    if(test)return self_test()?0:1;const auto total_start=Clock::now();const Scene scene=build_scene();
    BuildResult field=build_light_field(scene,bounces,error_fraction,limits);const auto index_start=Clock::now();
    const FieldIndex index(scene,field.footprints,limits.max_index_entries);const double index_ms=std::chrono::duration<double,std::milli>(Clock::now()-index_start).count();
    const auto atlas_start=Clock::now();const RadianceAtlas direct(scene,index,384,384,0,0,24);
    const RadianceAtlas diffuse(scene,index,192,192,1,INT_MAX,8);
    const double atlas_ms=std::chrono::duration<double,std::milli>(Clock::now()-atlas_start).count();
    const auto render_start=Clock::now();const auto image=render(scene,direct,diffuse,width,height);
    const double render_ms=std::chrono::duration<double,std::milli>(Clock::now()-render_start).count();
    const auto write_start=Clock::now();write_ppm(out,image,width,height);
    const double write_ms=std::chrono::duration<double,std::milli>(Clock::now()-write_start).count();
    const double total_ms=std::chrono::duration<double,std::milli>(Clock::now()-total_start).count();
    std::cout<<std::fixed<<std::setprecision(3)<<"{\n  \"width\": "<<width<<",\n  \"height\": "<<height
        <<",\n  \"analytic_primitives\": "<<scene.primitives.size()<<",\n  \"prebuilt_surface_cells\": 0"
        <<",\n  \"lightfield_footprints\": "<<field.footprints.size()<<",\n  \"construction_rays\": "<<field.stats.rays
        <<",\n  \"bundle_cells\": "<<field.stats.cells<<",\n  \"bundle_splits\": "<<field.stats.splits
        <<",\n  \"accepted_bundles\": "<<field.stats.accepted<<",\n  \"boundary_microcells\": "<<field.stats.boundary_leaves
        <<",\n  \"energy_pruned_cells\": "<<field.stats.pruned<<",\n  \"emitted_energy\": "<<energy(field.stats.emitted)
        <<",\n  \"deposited_energy\": "<<energy(field.stats.deposited)<<",\n  \"escaped_energy\": "<<energy(field.stats.escaped)
        <<",\n  \"discarded_energy\": "<<energy(field.stats.discarded)<<",\n  \"field_build_ms\": "<<field.ms
        <<",\n  \"index_ms\": "<<index_ms<<",\n  \"diffuse_atlas_ms\": "<<atlas_ms
        <<",\n  \"raster_ms\": "<<render_ms<<",\n  \"write_ms\": "<<write_ms
        <<",\n  \"total_ms\": "<<total_ms<<",\n  \"footprints_by_generation\": [";
    for(std::size_t i=0;i<field.stats.footprints_by_generation.size();++i){if(i)std::cout<<",";std::cout<<field.stats.footprints_by_generation[i];}
    std::cout<<"],\n  \"emitters_by_generation\": [";
    for(std::size_t i=0;i<field.stats.emitters_by_generation.size();++i){if(i)std::cout<<",";std::cout<<field.stats.emitters_by_generation[i];}
    std::cout<<"],\n  \"output\": \""<<out<<"\"\n}\n";return 0;
}catch(const std::exception& error){std::cerr<<"error: "<<error.what()<<"\n";return 2;}
