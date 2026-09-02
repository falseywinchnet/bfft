#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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
#include <utility>
#include <vector>

extern "C" int conv_resize_lines_f32(const float* input,int source_count,int lanes,
                                      float* output,int target_count);
extern "C" int conv_evaluate_profile_f32(const float* input,int source_count,int lanes,
                                          const float* position,float* output,int sample_count);

namespace {

using Clock=std::chrono::steady_clock;
using RGB=std::array<double,3>;
constexpr double pi=std::numbers::pi_v<double>;

struct Vec3{double x=0,y=0,z=0;};
Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
Vec3 operator*(Vec3 a,double s){return {a.x*s,a.y*s,a.z*s};}
Vec3 operator/(Vec3 a,double s){return {a.x/s,a.y/s,a.z/s};}
double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double norm(Vec3 a){return std::sqrt(dot(a,a));}
Vec3 unit(Vec3 a){const double n=norm(a);return n>0?a/n:Vec3{};}
RGB operator+(RGB a,const RGB& b){for(int c=0;c<3;++c)a[c]+=b[c];return a;}
RGB operator*(RGB a,double s){for(double& x:a)x*=s;return a;}
RGB multiply(RGB a,const RGB& b){for(int c=0;c<3;++c)a[c]*=b[c];return a;}
RGB& operator+=(RGB& a,const RGB& b){a=a+b;return a;}
double energy(const RGB& a){return a[0]+a[1]+a[2];}

enum class Shape{Plane,Sphere};
enum Group:int{Floor,Back,LeftWall,RightWall,Ceiling,AreaLight,
               RedSphere,BlueSphere,GreenSphere,GroupCount};
struct Object{
    Shape shape=Shape::Plane;Group group=Floor;RGB albedo{},emission{};
    Vec3 origin{},u{},v{},normal{},center{};double radius=0,area=0,extent=0;
};
struct Scene{std::vector<Object> objects;};

int add_plane(Scene& s,Group group,Vec3 origin,Vec3 u,Vec3 v,Vec3 normal,
              RGB albedo,RGB emission={}){
    Object o;o.shape=Shape::Plane;o.group=group;o.albedo=albedo;o.emission=emission;
    o.origin=origin;o.u=u;o.v=v;o.normal=normal;o.center=origin+u*.5+v*.5;
    o.area=norm(cross(u,v));o.extent=std::sqrt(o.area/pi);s.objects.push_back(o);
    return static_cast<int>(s.objects.size())-1;
}
int add_sphere(Scene& s,Group group,Vec3 center,double radius,RGB albedo){
    Object o;o.shape=Shape::Sphere;o.group=group;o.center=center;o.radius=radius;
    o.area=4*pi*radius*radius;o.extent=radius;o.albedo=albedo;s.objects.push_back(o);
    return static_cast<int>(s.objects.size())-1;
}
Scene build_scene(){Scene s;const RGB white{.72,.74,.76};
    add_plane(s,Floor,{-3,0,-5},{6,0,0},{0,0,6},{0,1,0},white);
    add_plane(s,Back,{-3,0,-5},{6,0,0},{0,4,0},{0,0,1},white);
    add_plane(s,LeftWall,{-3,0,1},{0,0,-6},{0,4,0},{1,0,0},{.58,.16,.12});
    add_plane(s,RightWall,{3,0,-5},{0,0,6},{0,4,0},{-1,0,0},{.12,.22,.62});
    add_plane(s,Ceiling,{-3,4,1},{6,0,0},{0,0,-6},{0,-1,0},white);
    add_plane(s,AreaLight,{-1.15,3.92,-3.0},{2.3,0,0},{0,0,1.7},{0,-1,0},
              {.02,.02,.02},{34,31,27});
    add_sphere(s,RedSphere,{-1.35,.72,-2.65},.72,{.78,.08,.055});
    add_sphere(s,BlueSphere,{1.22,.62,-3.15},.62,{.055,.20,.82});
    add_sphere(s,GreenSphere,{.15,.48,-1.20},.48,{.08,.72,.20});return s;
}

struct SurfacePoint{Vec3 position{},normal{};double jacobian=0;};
SurfacePoint plane_point(const Object& o,double u,double v){
    return {o.origin+o.u*u+o.v*v,o.normal,o.area};
}

struct Hit{bool valid=false;int object=-1;double t=0;SurfacePoint surface{};};
Hit intersect(const Object& o,int id,Vec3 origin,Vec3 ray){Hit h;h.object=id;
    if(o.shape==Shape::Plane){const double den=dot(o.normal,ray);if(std::abs(den)<1e-12)return h;
        const double t=dot(o.origin-origin,o.normal)/den;if(t<=1e-6)return h;const Vec3 point=origin+ray*t;
        const Vec3 d=point-o.origin;const double u=dot(d,o.u)/dot(o.u,o.u),v=dot(d,o.v)/dot(o.v,o.v);
        if(u<0||u>1||v<0||v>1)return h;h.valid=true;h.t=t;h.surface=plane_point(o,u,v);return h;}
    const Vec3 rel=origin-o.center;const double b=dot(ray,rel),c=dot(rel,rel)-o.radius*o.radius;
    const double disc=b*b-c;if(disc<0)return h;const double root=std::sqrt(disc);
    double t=-b-root;if(t<=1e-6)t=-b+root;if(t<=1e-6)return h;
    const Vec3 point=origin+ray*t;h.valid=true;h.t=t;h.surface={point,unit(point-o.center),0};return h;
}
Hit first_hit(const Scene& s,Vec3 origin,Vec3 ray){Hit best;best.t=std::numeric_limits<double>::infinity();
    for(int i=0;i<static_cast<int>(s.objects.size());++i){const Hit h=intersect(s.objects[i],i,origin,ray);
        if(h.valid&&h.t<best.t)best=h;}return best;
}

double circle_overlap(double r,double R,double d){
    if(d>=r+R)return 0;if(d<=std::abs(R-r)){const double q=std::min(r,R);return pi*q*q;}
    const double a=std::acos(std::clamp((d*d+r*r-R*R)/(2*d*r),-1.0,1.0));
    const double b=std::acos(std::clamp((d*d+R*R-r*r)/(2*d*R),-1.0,1.0));
    const double q=.5*std::sqrt(std::max(0.0,(-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)));
    return r*r*a+R*R*b-q;
}

constexpr int MaxBlockers=16;
constexpr int MaxCuts=2+2*MaxBlockers;
struct Relation{
    int source=-1,receiver=-1;double factor=0,scale=1;int implicit_boundaries=0;
    std::array<int,MaxBlockers> blockers{};int blocker_count=0;
};
struct EvalStats{
    std::uint64_t surface_hits=0,relation_evaluations=0,plane_evaluations=0,sphere_evaluations=0;
    std::uint64_t blocker_candidates=0,blocker_overlaps=0,unoccluded_fast_paths=0;
    std::uint64_t clipped_source_integrals=0,blocked_intervals=0;
    EvalStats& operator+=(const EvalStats& other){surface_hits+=other.surface_hits;
        relation_evaluations+=other.relation_evaluations;plane_evaluations+=other.plane_evaluations;
        sphere_evaluations+=other.sphere_evaluations;blocker_candidates+=other.blocker_candidates;
        blocker_overlaps+=other.blocker_overlaps;unoccluded_fast_paths+=other.unoccluded_fast_paths;
        clipped_source_integrals+=other.clipped_source_integrals;blocked_intervals+=other.blocked_intervals;return *this;}
};

Relation make_relation(const Scene& scene,int source,int receiver){Relation relation;
    relation.source=source;relation.receiver=receiver;
    for(int k=0;k<static_cast<int>(scene.objects.size());++k){if(k==source||k==receiver)continue;
        if(scene.objects[k].shape!=Shape::Sphere)continue;
        if(relation.blocker_count>=MaxBlockers)throw std::runtime_error("relation blocker ceiling reached");
        relation.blockers[relation.blocker_count++]=k;}
    relation.implicit_boundaries=relation.blocker_count;return relation;
}

bool segment_visible(const Scene& scene,const Relation& relation,const std::array<int,MaxBlockers>& blockers,
                     int blocker_count,Vec3 target,Vec3 source_point){
    const Vec3 segment=source_point-target;const double length_squared=dot(segment,segment);
    if(length_squared<=1e-16)return true;
    for(int b=0;b<blocker_count;++b){const Object& blocker=scene.objects[blockers[b]];
        const double t=dot(blocker.center-target,segment)/length_squared;
        if(t<=1e-9||t>=1-1e-9)continue;const Vec3 nearest=target+segment*t;
        if(dot(nearest-blocker.center,nearest-blocker.center)<blocker.radius*blocker.radius*(1-1e-12))return false;
    }
    (void)relation;
    return true;
}

double plane_cell_geometry(const Object& source,const SurfacePoint& target,
                           double u0=0,double u1=1,double v0=0,double v1=1){
    const Vec3 center=source.origin+source.u*((u0+u1)*.5)+source.v*((v0+v1)*.5);
    if(dot(source.normal,target.position-center)<=1e-12)return 0;
    const Vec3 p00=source.origin+source.u*u0+source.v*v0;
    const Vec3 p10=source.origin+source.u*u1+source.v*v0;
    const Vec3 p11=source.origin+source.u*u1+source.v*v1;
    const Vec3 p01=source.origin+source.u*u0+source.v*v1;
    std::array<Vec3,8> polygon{{p00,p10,p11,p01}};int count=4;
    if(dot(cross(source.u,source.v),source.normal)>0){polygon[1]=p01;polygon[3]=p10;}
    std::array<Vec3,8> clipped{};int clipped_count=0;
    for(int i=0;i<count;++i){const Vec3 a=polygon[i],b=polygon[(i+1)%count];
        const double da=dot(target.normal,a-target.position),db=dot(target.normal,b-target.position);
        const bool ia=da>1e-12,ib=db>1e-12;if(ia)clipped[clipped_count++]=a;
        if(ia!=ib){const double t=da/(da-db);clipped[clipped_count++]=a+(b-a)*t;}}
    if(clipped_count<3)return 0;Vec3 edge_sum{};
    for(int i=0;i<clipped_count;++i){const Vec3 a=unit(clipped[i]-target.position);
        const Vec3 b=unit(clipped[(i+1)%clipped_count]-target.position);const Vec3 edge=cross(a,b);
        const double sine=norm(edge);if(sine<=1e-14)continue;
        const double angle=std::atan2(sine,std::clamp(dot(a,b),-1.0,1.0));edge_sum=edge_sum+edge*(angle/sine);}
    return std::max(.5*dot(target.normal,edge_sum),0.0)/(pi*source.area);
}

void append_conic_roots(const Object& source,const Object& blocker,const SurfacePoint& target,
                        double v,std::array<double,MaxCuts>& cuts,int& cut_count){
    const Vec3 base=source.origin+source.v*v-target.position,U=source.u,w=blocker.center-target.position;
    const double k=dot(w,w)-blocker.radius*blocker.radius;
    const double wu=dot(w,U),wb=dot(w,base);
    const double A=wu*wu-k*dot(U,U);
    const double B=2*(wb*wu-k*dot(base,U));
    const double C=wb*wb-k*dot(base,base);
    auto add=[&](double x){if(x>1e-10&&x<1-1e-10){
        if(cut_count>=MaxCuts)throw std::runtime_error("conic cut ceiling reached");
        cuts[cut_count++]=x;}};
    if(std::abs(A)<1e-14){if(std::abs(B)>1e-14)add(-C/B);return;}
    const double d=B*B-4*A*C;if(d<0)return;const double root=std::sqrt(std::max(0.0,d));
    const double q=-.5*(B+std::copysign(root,B));
    if(std::abs(q)>1e-18){add(q/A);add(C/q);}else add(-B/(2*A));
}

static constexpr std::array<double,8> gl_x{{
    -.9602898564975363,-.7966664774136267,-.5255324099163290,-.1834346424956498,
     .1834346424956498, .5255324099163290, .7966664774136267, .9602898564975363}};
static constexpr std::array<double,8> gl_w{{
    .1012285362903763,.2223810344533745,.3137066458778873,.3626837833783620,
    .3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763}};
static constexpr std::array<double,4> gl4_x{{
    -.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526}};
static constexpr std::array<double,4> gl4_w{{
    .3478548451374539,.6521451548625461,.6521451548625461,.3478548451374539}};
int interval_quadrature_order=8;

double plane_integrand(const Object& source,const SurfacePoint& target,double u,double v){
    const Vec3 point=source.origin+source.u*u+source.v*v;const Vec3 to_target=target.position-point;
    const double d=norm(to_target);if(d<=1e-12)return 0;const Vec3 direction=to_target/d;
    const double source_cos=std::max(dot(source.normal,direction),0.0);
    const double receiver_cos=std::max(dot(target.normal,direction*-1.0),0.0);
    return source_cos*receiver_cos/(pi*d*d);
}

// The receiver is never tiled. For each v ordinate, sphere silhouettes contribute
// their exact cone/plane roots, and only the resulting visible source intervals are
// integrated. These intervals are the source-defined bifurcations.
bool plane_blocker_active(const Object& plane,const Object& blocker,const SurfacePoint& target){
    const Vec3 source_axis=plane.center-target.position;const double source_distance=norm(source_axis);
    const double source_radius=.5*std::sqrt(dot(plane.u,plane.u)+dot(plane.v,plane.v));
    const Vec3 source_direction=source_axis/std::max(source_distance,1e-15);
    const Vec3 delta=blocker.center-target.position;const double distance=norm(delta);
    if(dot(delta,source_direction)+blocker.radius<=0)return false;
    if(distance-blocker.radius>=source_distance+source_radius)return false;
    const double source_angle=source_distance<=source_radius?pi:
        std::asin(std::clamp(source_radius/source_distance,0.0,1.0));
    const double blocker_angle=distance<=blocker.radius?pi*.5:
        std::asin(std::clamp(blocker.radius/distance,0.0,1.0));
    const double separation=std::acos(std::clamp(dot(source_direction,delta/std::max(distance,1e-15)),-1.0,1.0));
    return separation<source_angle+blocker_angle;
}

double clipped_plane_geometry(const Scene& scene,const Relation& relation,const SurfacePoint& target,
                              EvalStats* stats=nullptr,bool* occluded=nullptr){
    if(occluded)*occluded=false;
    const Object& plane=scene.objects[relation.source];if(stats){++stats->plane_evaluations;
        stats->blocker_candidates+=relation.blocker_count;}
    const double unoccluded=plane_cell_geometry(plane,target);if(unoccluded<=0)return 0;
    std::array<int,MaxBlockers> active{};int active_count=0;
    for(int b=0;b<relation.blocker_count;++b){const int id=relation.blockers[b];const Object& blocker=scene.objects[id];
        if(plane_blocker_active(plane,blocker,target))active[active_count++]=id;}
    if(stats)stats->blocker_overlaps+=active_count;
    if(active_count==0){if(stats)++stats->unoccluded_fast_paths;return unoccluded;}
    if(stats)++stats->clipped_source_integrals;double blocked_integral=0;
    for(int j=0;j<8;++j){const double v=.5*(gl_x[j]+1);std::array<double,MaxCuts> cuts{};
        int cut_count=2;cuts[0]=0;cuts[1]=1;
        for(int b=0;b<active_count;++b)append_conic_roots(plane,scene.objects[active[b]],target,v,cuts,cut_count);
        std::sort(cuts.begin(),cuts.begin()+cut_count);const auto unique_end=std::unique(cuts.begin(),cuts.begin()+cut_count,
            [](double a,double b){return std::abs(a-b)<1e-10;});cut_count=static_cast<int>(unique_end-cuts.begin());
        double blocked_row=0;
        for(int c=1;c<cut_count;++c){const double lo=cuts[c-1],hi=cuts[c];if(hi-lo<1e-12)continue;
            if(segment_visible(scene,relation,active,active_count,target.position,
                plane.origin+plane.u*((lo+hi)*.5)+plane.v*v))continue;
            if(occluded)*occluded=true;
            if(stats)++stats->blocked_intervals;
            double interval=0;if(interval_quadrature_order==4){for(int i=0;i<4;++i){
                const double u=.5*((hi-lo)*gl4_x[i]+hi+lo);interval+=gl4_w[i]*plane_integrand(plane,target,u,v);}}
            else{for(int i=0;i<8;++i){const double u=.5*((hi-lo)*gl_x[i]+hi+lo);
                interval+=gl_w[i]*plane_integrand(plane,target,u,v);}}
            blocked_row+=.5*(hi-lo)*interval;}
        blocked_integral+=gl_w[j]*blocked_row;
    }
    blocked_integral*=.5;
    return std::clamp(unoccluded-blocked_integral,0.0,unoccluded);
}

double angular_visibility(const Scene& scene,const Relation& relation,const SurfacePoint& target,
                          EvalStats* stats=nullptr,bool* occluded=nullptr){
    if(occluded)*occluded=false;
    const Object& emitter=scene.objects[relation.source];const Vec3 delta=emitter.center-target.position;
    const double source_distance=norm(delta);if(source_distance<=1e-10)return 0;
    const Vec3 axis=delta/source_distance;
    const double source_angle=emitter.shape==Shape::Sphere?
        std::asin(std::clamp(emitter.radius/source_distance,0.0,1.0)):
        std::atan2(emitter.extent,source_distance);
    if(source_angle<=1e-10)return 1;double visible=1;
    if(stats)stats->blocker_candidates+=relation.blocker_count;
    for(int b=0;b<relation.blocker_count;++b){const Object& blocker=scene.objects[relation.blockers[b]];
        const Vec3 w=blocker.center-target.position;const double d=norm(w);if(d>=source_distance+emitter.extent)continue;
        const double blocker_angle=d<=blocker.radius*(1+1e-12)?pi*.5:
            std::asin(std::clamp(blocker.radius/d,0.0,1.0));
        const double separation=std::acos(std::clamp(dot(axis,w/std::max(d,1e-15)),-1.0,1.0));
        const double blocked=circle_overlap(source_angle,blocker_angle,separation)/(pi*source_angle*source_angle);
        if(occluded&&blocked>0)*occluded=true;
        visible*=1-std::clamp(blocked,0.0,1.0);
    }
    return std::clamp(visible,0.0,1.0);
}

double raw_relation_density(const Scene& scene,const Relation& relation,const SurfacePoint& target,
                            EvalStats* stats=nullptr,bool* occluded=nullptr){
    if(occluded)*occluded=false;if(stats)++stats->relation_evaluations;const Object& emitter=scene.objects[relation.source];
    if(emitter.shape==Shape::Plane){
        return clipped_plane_geometry(scene,relation,target,stats,occluded);
    }
    if(stats)++stats->sphere_evaluations;
    const Vec3 delta=emitter.center-target.position;const double distance=norm(delta);
    if(distance<=emitter.radius+1e-9)return 0;const Vec3 direction=delta/distance;
    const double receiver_cos=std::max(dot(target.normal,direction),0.0);
    return receiver_cos/(4*pi*distance*distance)*angular_visibility(scene,relation,target,stats,occluded);
}

template<class F> double integrate_surface(const Object& receiver,F&& function){
    double sum=0;if(receiver.shape==Shape::Plane){constexpr int n=14;const double step=1.0/n;
        for(int y=0;y<n;++y)for(int x=0;x<n;++x){const SurfacePoint p=plane_point(receiver,(x+.5)*step,(y+.5)*step);
            sum+=function(p)*receiver.area*step*step;}}
    else{constexpr int n=384;constexpr double golden=2.3999632297286533;
        for(int i=0;i<n;++i){const double z=1-2*(i+.5)/n;const double r=std::sqrt(std::max(0.0,1-z*z));
            const double a=golden*i;const Vec3 normal{r*std::cos(a),z,r*std::sin(a)};
            const SurfacePoint p{receiver.center+normal*receiver.radius,normal,receiver.area/n};
            sum+=function(p)*receiver.area/n;}}
    return sum;
}

double relation_density(const Scene& scene,const Relation& relation,const SurfacePoint& target,EvalStats* stats=nullptr){
    return relation.scale*raw_relation_density(scene,relation,target,stats);
}

std::vector<Relation> build_relations(const Scene& scene){std::vector<Relation> relations;
    for(int source=0;source<static_cast<int>(scene.objects.size());++source)
        for(int receiver=0;receiver<static_cast<int>(scene.objects.size());++receiver){if(source==receiver)continue;
            Relation relation=make_relation(scene,source,receiver);
            relation.factor=integrate_surface(scene.objects[receiver],[&](const SurfacePoint& p){
                return raw_relation_density(scene,relation,p);});
            if(relation.factor>1e-12)relations.push_back(relation);}
    for(int source=0;source<static_cast<int>(scene.objects.size());++source){double sum=0;
        for(const Relation& r:relations)if(r.source==source)sum+=r.factor;
        const double scale=sum>.9?.9/sum:1;for(Relation& r:relations)if(r.source==source){r.scale=scale;r.factor*=scale;}}
    return relations;
}

struct Coupling{int outgoing=-1;double weight=0;};
struct WindowOperator{std::vector<std::vector<Coupling>> from_incoming;std::uint64_t nonzeros=0;};
WindowOperator build_operator(const Scene& scene,const std::vector<Relation>& relations){
    const int count=static_cast<int>(scene.objects.size());std::vector<std::vector<int>> index(count,std::vector<int>(count,-1));
    for(int i=0;i<static_cast<int>(relations.size());++i)index[relations[i].source][relations[i].receiver]=i;
    WindowOperator op;op.from_incoming.resize(relations.size());
    for(int incoming=0;incoming<static_cast<int>(relations.size());++incoming){const Relation& r=relations[incoming];
        const int surface=r.receiver;double column_sum=0;
        for(int outgoing=0;outgoing<static_cast<int>(relations.size());++outgoing){const Relation& s=relations[outgoing];
            if(s.source!=surface)continue;const int reverse_index=index[s.receiver][surface];if(reverse_index<0)continue;
            const Relation& reverse=relations[reverse_index];const double overlap=integrate_surface(scene.objects[surface],
                [&](const SurfacePoint& p){return relation_density(scene,r,p)*relation_density(scene,reverse,p);});
            const double weight=scene.objects[surface].area*s.factor*overlap/(r.factor*reverse.factor);
            if(weight<=1e-12)continue;op.from_incoming[incoming].push_back({outgoing,weight});column_sum+=weight;}
        const double scale=column_sum>.9?.9/column_sum:1;for(Coupling& c:op.from_incoming[incoming])c.weight*=scale;
        op.nonzeros+=op.from_incoming[incoming].size();}
    return op;
}

struct MarchResult{std::vector<RGB> total_delivered;std::vector<double> depth_energy;int depths=0;};
MarchResult march(const Scene& scene,const std::vector<Relation>& relations,const WindowOperator& op,
                  int max_depth,double threshold){std::vector<RGB> frontier(relations.size()),total(relations.size());
    for(int i=0;i<static_cast<int>(relations.size());++i){const Object& source=scene.objects[relations[i].source];
        frontier[i]=total[i]=source.emission*(source.area*pi*relations[i].factor);}
    MarchResult result;for(int depth=0;depth<max_depth;++depth){double active=0;for(const RGB& q:frontier)active+=energy(q);
        result.depth_energy.push_back(active);if(active<=threshold)break;std::vector<RGB> next(relations.size());
        for(int incoming=0;incoming<static_cast<int>(relations.size());++incoming){
            const RGB reflected=multiply(frontier[incoming],scene.objects[relations[incoming].receiver].albedo);
            for(const Coupling& c:op.from_incoming[incoming])next[c.outgoing]+=reflected*c.weight;}
        for(int i=0;i<static_cast<int>(relations.size());++i)total[i]+=next[i];frontier.swap(next);++result.depths;}
    result.total_delivered=std::move(total);return result;
}

struct EvaluatorProgram{std::array<std::vector<int>,GroupCount> incoming;};
EvaluatorProgram compile_evaluator(const std::vector<Relation>& relations){EvaluatorProgram program;
    for(int i=0;i<static_cast<int>(relations.size());++i)program.incoming[relations[i].receiver].push_back(i);
    return program;
}

RGB surface_radiance(const Scene& scene,const std::vector<Relation>& relations,const MarchResult& march,
                     const EvaluatorProgram& program,int receiver,const SurfacePoint& point,EvalStats* stats){RGB incident{};
    for(const int i:program.incoming[receiver]){const Relation& r=relations[i];
        incident+=march.total_delivered[i]*(relation_density(scene,r,point,stats)/r.factor);}
    return scene.objects[receiver].emission+multiply(incident,scene.objects[receiver].albedo)*(1.0/pi);
}

struct TerminalTerm{int relation=-1;RGB coefficient{};};
struct TerminalProgram{std::array<std::vector<TerminalTerm>,GroupCount> incoming;std::uint64_t fused_terms=0;};
TerminalProgram compile_terminal_program(const std::vector<Relation>& relations,const MarchResult& marched){
    TerminalProgram program;for(int i=0;i<static_cast<int>(relations.size());++i){const Relation& relation=relations[i];
        const double multiplier=relation.scale/relation.factor;
        program.incoming[relation.receiver].push_back({i,marched.total_delivered[i]*multiplier});++program.fused_terms;}
    return program;
}

RGB terminal_surface_radiance(const Scene& scene,const std::vector<Relation>& relations,
                              const TerminalProgram& program,int receiver,const SurfacePoint& point,EvalStats* stats,
                              std::uint64_t* behavior_signature=nullptr){RGB incident{};std::uint64_t signature=0;int bit=0;
    for(const TerminalTerm& term:program.incoming[receiver]){if(bit>=64)throw std::runtime_error("terminal signature ceiling reached");
        bool occluded=false;incident+=term.coefficient*raw_relation_density(
            scene,relations[term.relation],point,stats,behavior_signature?&occluded:nullptr);
        if(occluded)signature|=std::uint64_t{1}<<bit;
        ++bit;}
    if(behavior_signature)*behavior_signature=signature;
    return scene.objects[receiver].emission+multiply(incident,scene.objects[receiver].albedo)*(1.0/pi);
}

bool plane_relation_occluded(const Scene& scene,const Relation& relation,const SurfacePoint& target){
    const Object& plane=scene.objects[relation.source];if(plane_cell_geometry(plane,target)<=0)return false;
    std::array<int,MaxBlockers> active{};int active_count=0;
    for(int b=0;b<relation.blocker_count;++b){const int id=relation.blockers[b];
        if(plane_blocker_active(plane,scene.objects[id],target))active[active_count++]=id;}
    if(active_count==0)return false;
    for(int j=0;j<8;++j){const double v=.5*(gl_x[j]+1);std::array<double,MaxCuts> cuts{};
        int cut_count=2;cuts[0]=0;cuts[1]=1;
        for(int b=0;b<active_count;++b)append_conic_roots(plane,scene.objects[active[b]],target,v,cuts,cut_count);
        std::sort(cuts.begin(),cuts.begin()+cut_count);const auto unique_end=std::unique(cuts.begin(),cuts.begin()+cut_count,
            [](double a,double b){return std::abs(a-b)<1e-10;});cut_count=static_cast<int>(unique_end-cuts.begin());
        for(int c=1;c<cut_count;++c){const double lo=cuts[c-1],hi=cuts[c];if(hi-lo<1e-12)continue;
            if(!segment_visible(scene,relation,active,active_count,target.position,
                plane.origin+plane.u*((lo+hi)*.5)+plane.v*v))return true;}}
    return false;
}

bool sphere_relation_occluded(const Scene& scene,const Relation& relation,const SurfacePoint& target){
    const Object& emitter=scene.objects[relation.source];const Vec3 delta=emitter.center-target.position;
    const double source_distance=norm(delta);if(source_distance<=1e-10)return false;const Vec3 axis=delta/source_distance;
    const double source_angle=std::asin(std::clamp(emitter.radius/source_distance,0.0,1.0));if(source_angle<=1e-10)return false;
    for(int b=0;b<relation.blocker_count;++b){const Object& blocker=scene.objects[relation.blockers[b]];
        const Vec3 w=blocker.center-target.position;const double d=norm(w);if(d>=source_distance+emitter.extent)continue;
        const double blocker_angle=d<=blocker.radius*(1+1e-12)?pi*.5:
            std::asin(std::clamp(blocker.radius/d,0.0,1.0));
        const double separation=std::acos(std::clamp(dot(axis,w/std::max(d,1e-15)),-1.0,1.0));
        if(circle_overlap(source_angle,blocker_angle,separation)>0)return true;}
    return false;
}

std::uint64_t terminal_behavior_signature(const Scene& scene,const std::vector<Relation>& relations,
                                          const TerminalProgram& program,int receiver,const SurfacePoint& point){
    std::uint64_t signature=0;int bit=0;for(const TerminalTerm& term:program.incoming[receiver]){
        if(bit>=64)throw std::runtime_error("terminal signature ceiling reached");const Relation& relation=relations[term.relation];
        const bool occluded=scene.objects[relation.source].shape==Shape::Plane?
            plane_relation_occluded(scene,relation,point):sphere_relation_occluded(scene,relation,point);
        if(occluded)signature|=std::uint64_t{1}<<bit;++bit;}return signature;
}

struct Camera{Vec3 origin{},forward{},right{},up{};double scale=0,aspect=1;};
Camera make_camera(int width,int height){const Vec3 origin{0,2.15,6.8},target{0,1.55,-2.15};
    Camera camera;camera.origin=origin;camera.forward=unit(target-origin);
    camera.right=unit(cross(camera.forward,{0,1,0}));camera.up=cross(camera.right,camera.forward);
    camera.scale=std::tan(24*pi/180.0);camera.aspect=double(width)/height;return camera;
}
Vec3 camera_ray(const Camera& camera,int width,int height,double x,double y){
    const double px=(2*(x+.5)/width-1)*camera.aspect*camera.scale;
    const double py=(1-2*(y+.5)/height)*camera.scale;
    return unit(camera.forward+camera.right*px+camera.up*py);
}

std::vector<float> conv_resize_rgb(const std::vector<float>& source,int source_width,int source_height,
                                   int target_width,int target_height){
    const int x_lanes=source_height*3;std::vector<float> x_lines(static_cast<std::size_t>(source_width)*x_lanes);
    for(int x=0;x<source_width;++x)for(int y=0;y<source_height;++y)for(int c=0;c<3;++c)
        x_lines[static_cast<std::size_t>(x)*x_lanes+y*3+c]=source[(static_cast<std::size_t>(y)*source_width+x)*3+c];
    std::vector<float> x_output(static_cast<std::size_t>(target_width)*x_lanes);
    if(conv_resize_lines_f32(x_lines.data(),source_width,x_lanes,x_output.data(),target_width)!=0)
        throw std::runtime_error("native CONV horizontal synthesis failed");
    const int y_lanes=target_width*3;std::vector<float> y_lines(static_cast<std::size_t>(source_height)*y_lanes);
    for(int y=0;y<source_height;++y)for(int x=0;x<target_width;++x)for(int c=0;c<3;++c)
        y_lines[static_cast<std::size_t>(y)*y_lanes+x*3+c]=x_output[static_cast<std::size_t>(x)*x_lanes+y*3+c];
    std::vector<float> output(static_cast<std::size_t>(target_height)*y_lanes);
    if(conv_resize_lines_f32(y_lines.data(),source_height,y_lanes,output.data(),target_height)!=0)
        throw std::runtime_error("native CONV vertical synthesis failed");
    return output;
}

std::uint8_t tone_byte(double linear){
    const double mapped=std::pow(std::clamp(1-std::exp(-.72*std::max(linear,0.0)),0.0,1.0),1/2.2);
    return static_cast<std::uint8_t>(std::lround(255*mapped));
}

struct TerminalRenderResult{
    std::vector<std::uint8_t> image;EvalStats stats;int scanlines=0;
    std::uint64_t anchor_samples=0,topology_queries=0,topology_safe_cells=0,validation_samples=0;
    std::uint64_t rejected_cells=0,refinement_failed_cells=0;
    std::uint64_t interpolated_pixels=0,curvature_fallback_pixels=0;
    double terminal_raster_ms=0;
};

struct TerminalLabel{int object=-1;std::uint64_t signature=0;};
bool operator==(const TerminalLabel& a,const TerminalLabel& b){return a.object==b.object&&a.signature==b.signature;}

// Each primary-surface scanline span begins as one optical interval. Exact
// source-window labels and CONV residual witnesses recursively bifurcate that
// interval; accepted leaves alone are synthesized. No spatial scale or image
// tile participates in the construction.
TerminalRenderResult render_terminal(const Scene& scene,const std::vector<Relation>& relations,
                                     const TerminalProgram& program,int width,int height,int terminal_error,
                                     bool profile){TerminalRenderResult result;
    result.scanlines=height;
    result.image.resize(static_cast<std::size_t>(width)*height*3);const Camera camera=make_camera(width,height);
    const unsigned workers=std::max(1u,std::thread::hardware_concurrency());std::vector<EvalStats> worker_stats(workers);
    struct Counts{std::uint64_t exact_samples=0,topology_queries=0,regions=0,validation=0,rejected=0,refinement_failed=0;
        std::uint64_t interpolated=0,curvature=0;};
    std::vector<Counts> counts(workers);std::atomic<int> next_row{0};std::vector<std::thread> threads;
    const auto render_start=Clock::now();
    for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&,worker]{
        std::vector<Hit> hits(width);std::vector<std::uint8_t> known(width),label_known(width);
        std::vector<RGB> exact(width);std::vector<TerminalLabel> labels(width);std::vector<float> synthesized;
        for(;;){const int y=next_row.fetch_add(1);if(y>=height)break;
            std::fill(known.begin(),known.end(),0);std::fill(label_known.begin(),label_known.end(),0);
            for(int x=0;x<width;++x)hits[x]=first_hit(scene,camera.origin,camera_ray(camera,width,height,x,y));
            auto evaluate=[&](int x)->const RGB&{if(known[x])return exact[x];known[x]=1;++counts[worker].exact_samples;
                label_known[x]=1;++counts[worker].topology_queries;
                if(!hits[x].valid){labels[x]={-1,0};exact[x]={};return exact[x];}
                EvalStats* stats=profile?&worker_stats[worker]:nullptr;if(stats)++stats->surface_hits;
                std::uint64_t signature=0;exact[x]=terminal_surface_radiance(
                    scene,relations,program,hits[x].object,hits[x].surface,stats,&signature);
                labels[x]={hits[x].object,signature};
                return exact[x];};
            auto classify=[&](int x)->const TerminalLabel&{if(!label_known[x])evaluate(x);return labels[x];};
            // Camera rays give exact primary-surface boundaries.  Transport
            // boundaries are queried only at the dyadic witnesses below; a
            // changed source/occluder ownership rejects the whole interval and
            // bifurcates it.  There is no dense topology-label image.
            int run_begin=0;while(run_begin<width){const int object=hits[run_begin].valid?hits[run_begin].object:-1;
                int run_end=run_begin;while(run_end+1<width&&
                    (hits[run_end+1].valid?hits[run_end+1].object:-1)==object)++run_end;
                if(object<0){run_begin=run_end+1;continue;}
                ++counts[worker].regions;
                auto exact_segment=[&](int begin,int end){++counts[worker].refinement_failed;
                    for(int x=begin;x<=end;++x){const RGB& value=evaluate(x);
                        const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;
                        for(int c=0;c<3;++c)result.image[offset+c]=tone_byte(value[c]);++counts[worker].curvature;}};
                auto approximate=[&](auto&& self,int begin,int end)->void{const int segment_length=end-begin+1;
                    if(segment_length<5){exact_segment(begin,end);return;}
                    constexpr std::array<double,5> control_fraction{{0,.25,.5,.75,1}};
                    // Certify the interpolant between all five controls.  The
                    // controls occupy multiples of 4/16; every other interior
                    // sixteenth is an independent exact residual witness.
                    constexpr std::array<double,12> probe_fraction{{
                        1./16,2./16,3./16,5./16,6./16,7./16,
                        9./16,10./16,11./16,13./16,14./16,15./16}};
                    std::array<int,5> control{};std::array<float,15> source{};
                    for(int i=0;i<5;++i){control[i]=std::clamp(static_cast<int>(std::lround(
                            begin+control_fraction[i]*(end-begin))),begin,end);const RGB& value=evaluate(control[i]);
                        for(int c=0;c<3;++c)source[i*3+c]=static_cast<float>(value[c]);}
                    const TerminalLabel segment_label=classify(control[0]);bool reject=false;
                    for(int i=1;i<5;++i)reject=reject||classify(control[i])!=segment_label;
                    if(reject){++counts[worker].rejected;const int middle=(begin+end)/2;
                        if(middle<=begin||middle>=end){exact_segment(begin,end);return;}
                        self(self,begin,middle);self(self,middle+1,end);return;}
                    const int probe_tolerance=std::max(0,terminal_error-1);
                    std::array<int,12> probe{};std::array<float,12> probe_position{};
                    std::array<float,36> probe_synthesized{};
                    for(int i=0;i<12;++i){probe[i]=std::clamp(static_cast<int>(std::lround(
                            begin+probe_fraction[i]*(end-begin))),begin,end);++counts[worker].validation;
                        if(classify(probe[i])!=segment_label){reject=true;break;}
                        probe_position[i]=static_cast<float>(4.0*(probe[i]-begin)/(end-begin));}
                    if(!reject&&conv_evaluate_profile_f32(source.data(),5,3,probe_position.data(),
                            probe_synthesized.data(),12)!=0)
                        throw std::runtime_error("native CONV adaptive certificate failed");
                    for(int i=0;!reject&&i<12;++i){const RGB& value=evaluate(probe[i]);
                        for(int c=0;!reject&&c<3;++c)reject=std::abs(int(tone_byte(value[c]))-
                            int(tone_byte(probe_synthesized[i*3+c])))>probe_tolerance;}
                    if(reject){++counts[worker].rejected;const int middle=(begin+end)/2;
                        if(middle<=begin||middle>=end){exact_segment(begin,end);return;}
                        self(self,begin,middle);self(self,middle+1,end);return;}
                    synthesized.resize(static_cast<std::size_t>(segment_length)*3);
                    if(conv_resize_lines_f32(source.data(),5,3,synthesized.data(),segment_length)!=0)
                        throw std::runtime_error("native CONV adaptive regional synthesis failed");
                    for(int x=begin;x<=end;++x){const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;
                        bool control_site=false;for(int site:control)control_site=control_site||x==site;
                        if(control_site){const RGB& value=evaluate(x);
                            for(int c=0;c<3;++c)result.image[offset+c]=tone_byte(value[c]);++counts[worker].curvature;}
                        else{const std::size_t local=static_cast<std::size_t>(x-begin)*3;
                            for(int c=0;c<3;++c)result.image[offset+c]=tone_byte(synthesized[local+c]);
                            ++counts[worker].interpolated;}}};
                approximate(approximate,run_begin,run_end);
                run_begin=run_end+1;}
        }});
    for(auto& thread:threads)thread.join();result.terminal_raster_ms=std::chrono::duration<double,std::milli>(Clock::now()-render_start).count();
    for(unsigned worker=0;worker<workers;++worker){result.stats+=worker_stats[worker];
        result.anchor_samples+=counts[worker].exact_samples;result.topology_queries+=counts[worker].topology_queries;
        result.topology_safe_cells+=counts[worker].regions;
        result.validation_samples+=counts[worker].validation;result.rejected_cells+=counts[worker].rejected;
        result.refinement_failed_cells+=counts[worker].refinement_failed;
        result.interpolated_pixels+=counts[worker].interpolated;
        result.curvature_fallback_pixels+=counts[worker].curvature;}return result;
}

struct RenderResult{std::vector<std::uint8_t> image;EvalStats stats;};
RenderResult render(const Scene& scene,const std::vector<Relation>& relations,const MarchResult& marched,
                    const EvaluatorProgram& program,int width,int height,bool profile){
    RenderResult result;result.image.resize(static_cast<std::size_t>(width)*height*3);
    const Camera camera=make_camera(width,height);std::atomic<int> next_row{0};
    std::vector<std::thread> threads;const unsigned workers=std::max(1u,std::thread::hardware_concurrency());
    std::vector<EvalStats> worker_stats(workers);
    for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&,worker]{for(;;){const int y=next_row.fetch_add(1);if(y>=height)break;
        for(int x=0;x<width;++x){const Hit hit=first_hit(scene,camera.origin,camera_ray(camera,width,height,x,y));RGB linear{};
            if(hit.valid){EvalStats* stats=profile?&worker_stats[worker]:nullptr;if(stats)++stats->surface_hits;
                linear=surface_radiance(scene,relations,marched,program,hit.object,hit.surface,stats);}
            const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;
            for(int c=0;c<3;++c)result.image[offset+c]=tone_byte(linear[c]);}}});
    for(auto& thread:threads)thread.join();for(const EvalStats& stats:worker_stats)result.stats+=stats;return result;
}

void write_ppm(const std::string& path,const std::vector<std::uint8_t>& image,int width,int height){
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot open output");
    out<<"P6\n"<<width<<" "<<height<<"\n255\n";out.write(reinterpret_cast<const char*>(image.data()),image.size());
}

bool self_test(){const Scene scene=build_scene();const SurfacePoint probe=plane_point(scene.objects[Floor],.37,.41);
    const double whole=plane_cell_geometry(scene.objects[AreaLight],probe);
    const double parts=plane_cell_geometry(scene.objects[AreaLight],probe,0,.5,0,.5)+
        plane_cell_geometry(scene.objects[AreaLight],probe,.5,1,0,.5)+
        plane_cell_geometry(scene.objects[AreaLight],probe,.5,1,.5,1)+
        plane_cell_geometry(scene.objects[AreaLight],probe,0,.5,.5,1);
    if(whole<=0||std::abs(whole-parts)>1e-10*whole)return false;
    const auto relations=build_relations(scene);if(relations.empty()||relations.size()>72)return false;
    int boundaries=0;std::vector<double> rows(scene.objects.size());for(const Relation& r:relations){
        if(r.factor<0)return false;rows[r.source]+=r.factor;boundaries+=r.implicit_boundaries;}
    if(*std::max_element(rows.begin(),rows.end())>.900001||boundaries>300)return false;
    const SurfacePoint contact{{-1.35,0,-2.65},{0,1,0},scene.objects[Floor].area};
    const SurfacePoint open{{-2.5,0,-1.0},{0,1,0},scene.objects[Floor].area};
    const Relation light_floor=make_relation(scene,AreaLight,Floor);
    const double contact_light=raw_relation_density(scene,light_floor,contact);
    const double open_light=raw_relation_density(scene,light_floor,open);
    if(!(contact_light<.1*open_light)){std::cerr<<"contact visibility failure "<<contact_light<<" "<<open_light<<"\n";return false;}
    const auto op=build_operator(scene,relations);if(op.nonzeros==0)return false;
    for(const auto& column:op.from_incoming){double sum=0;for(const Coupling& c:column){if(c.weight<0)return false;sum+=c.weight;}
        if(sum>.900001)return false;}
    const auto result=march(scene,relations,op,8,1e-6);if(result.depth_energy.size()<2)return false;
    const auto evaluator=compile_evaluator(relations);const auto terminal=compile_terminal_program(relations,result);
    EvalStats exact_stats,terminal_stats;const RGB exact=surface_radiance(scene,relations,result,evaluator,Floor,open,&exact_stats);
    std::uint64_t fused_signature=0;const RGB fused=terminal_surface_radiance(
        scene,relations,terminal,Floor,open,&terminal_stats,&fused_signature);
    for(int c=0;c<3;++c)if(std::abs(exact[c]-fused[c])>1e-12*std::max(1.0,std::abs(exact[c])))return false;
    if(fused_signature!=terminal_behavior_signature(scene,relations,terminal,Floor,open))return false;
    terminal_surface_radiance(scene,relations,terminal,Floor,contact,nullptr,&fused_signature);
    if(fused_signature!=terminal_behavior_signature(scene,relations,terminal,Floor,contact))return false;
    std::vector<float> constant(5*5*3,.375f);const auto enlarged=conv_resize_rgb(constant,5,5,11,9);
    for(float value:enlarged)if(std::abs(value-.375f)>2e-6f)return false;
    const auto exact_camera=render(scene,relations,result,evaluator,160,120,false);
    const auto terminal_camera=render_terminal(scene,relations,terminal,160,120,1,false);
    if(exact_camera.image.size()!=terminal_camera.image.size())return false;
    for(std::size_t i=0;i<exact_camera.image.size();++i)
        if(std::abs(int(exact_camera.image[i])-int(terminal_camera.image[i]))>1)return false;
    std::cout<<"source-region invariants: ok ("<<relations.size()<<" directed regions, "<<boundaries
        <<" implicit occluder boundaries, "<<op.nonzeros<<" couplings, "<<terminal.fused_terms
        <<" backward-fused terminal terms, terminal max error <= 1)\n";return true;
}

} // namespace

int main(int argc,char** argv)try{int width=800,height=600,max_depth=10,terminal_error=1;double threshold=1e-5;
    std::string out="/tmp/source_region_transport.ppm";bool test=false,profile=false;std::uint64_t max_regions=1000;double max_build_seconds=60;
    for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--self-test"){test=true;continue;}
        if(arg=="--profile-evaluator"){profile=true;continue;}
        if(i+1>=argc)throw std::runtime_error("missing argument value");
        if(arg=="--width")width=std::stoi(argv[++i]);else if(arg=="--height")height=std::stoi(argv[++i]);
        else if(arg=="--depth")max_depth=std::stoi(argv[++i]);else if(arg=="--threshold")threshold=std::stod(argv[++i]);
        else if(arg=="--max-nodes"||arg=="--max-regions")max_regions=std::stoull(argv[++i]);
        else if(arg=="--max-build-seconds")max_build_seconds=std::stod(argv[++i]);
        else if(arg=="--interval-order"){interval_quadrature_order=std::stoi(argv[++i]);
            if(interval_quadrature_order!=4&&interval_quadrature_order!=8)throw std::runtime_error("interval order must be 4 or 8");}
        else if(arg=="--terminal-error"){terminal_error=std::stoi(argv[++i]);
            if(terminal_error<0)throw std::runtime_error("terminal error must be nonnegative");}
        else if(arg=="--out")out=argv[++i];else throw std::runtime_error("unknown argument: "+arg);}
    if(test)return self_test()?0:1;
    const auto total_start=Clock::now();const Scene scene=build_scene();
    const auto build_start=Clock::now();const auto relations=build_relations(scene);
    int boundaries=0;for(const Relation& r:relations)boundaries+=r.implicit_boundaries;
    if(relations.size()+static_cast<std::size_t>(boundaries)>max_regions)throw std::runtime_error("source-region ceiling reached");
    const double build_ms=std::chrono::duration<double,std::milli>(Clock::now()-build_start).count();
    if(build_ms>max_build_seconds*1000)throw std::runtime_error("source-region build-time ceiling reached");
    const auto operator_start=Clock::now();const auto op=build_operator(scene,relations);
    const double operator_ms=std::chrono::duration<double,std::milli>(Clock::now()-operator_start).count();
    const auto march_start=Clock::now();const auto marched=march(scene,relations,op,max_depth,threshold);
    const double march_ms=std::chrono::duration<double,std::milli>(Clock::now()-march_start).count();
    const auto evaluator=compile_evaluator(relations);const auto terminal_program=compile_terminal_program(relations,marched);
    EvalStats evaluation_stats;std::vector<std::uint8_t> image;int terminal_scanlines=0;
    std::uint64_t anchor_samples=0,topology_queries=0,topology_safe_cells=0,validation_samples=0;
    std::uint64_t rejected_cells=0,refinement_failed_cells=0;
    std::uint64_t interpolated_pixels=0,curvature_fallback_pixels=0;
    double terminal_raster_ms=0;const auto raster_start=Clock::now();
    if(terminal_error>0){auto rendered=render_terminal(scene,relations,terminal_program,width,height,terminal_error,profile);
        evaluation_stats=rendered.stats;image=std::move(rendered.image);terminal_scanlines=rendered.scanlines;
        anchor_samples=rendered.anchor_samples;topology_queries=rendered.topology_queries;
        topology_safe_cells=rendered.topology_safe_cells;
        validation_samples=rendered.validation_samples;
        rejected_cells=rendered.rejected_cells;refinement_failed_cells=rendered.refinement_failed_cells;
        interpolated_pixels=rendered.interpolated_pixels;
        curvature_fallback_pixels=rendered.curvature_fallback_pixels;
        terminal_raster_ms=rendered.terminal_raster_ms;}
    else{auto rendered=render(scene,relations,marched,evaluator,width,height,profile);
        evaluation_stats=rendered.stats;image=std::move(rendered.image);}
    const double raster_ms=std::chrono::duration<double,std::milli>(Clock::now()-raster_start).count();
    const auto write_start=Clock::now();write_ppm(out,image,width,height);
    const double write_ms=std::chrono::duration<double,std::milli>(Clock::now()-write_start).count();
    const double total_ms=std::chrono::duration<double,std::milli>(Clock::now()-total_start).count();
    std::cout<<std::fixed<<std::setprecision(3)<<"{\n  \"width\": "<<width<<",\n  \"height\": "<<height
        <<",\n  \"analytic_objects\": "<<scene.objects.size()<<",\n  \"source_regions\": "<<relations.size()
        <<",\n  \"implicit_occluder_boundaries\": "<<boundaries<<",\n  \"stored_spatial_nodes\": 0"
        <<",\n  \"transport_couplings\": "<<op.nonzeros<<",\n  \"march_depths\": "<<marched.depths
        <<",\n  \"interval_quadrature_order\": "<<interval_quadrature_order
        <<",\n  \"evaluator_profiled\": "<<(profile?"true":"false")
        <<",\n  \"terminal_error\": "<<terminal_error<<",\n  \"backward_fused_terms\": "<<terminal_program.fused_terms
        <<",\n  \"terminal_scanlines\": "<<terminal_scanlines
        <<",\n  \"terminal_exact_radiance_samples\": "<<anchor_samples
        <<",\n  \"terminal_boundary_labels\": "<<topology_queries
        <<",\n  \"terminal_primary_surface_runs\": "<<topology_safe_cells
        <<",\n  \"terminal_certificate_samples\": "<<validation_samples
        <<",\n  \"terminal_subdivided_intervals\": "<<rejected_cells
        <<",\n  \"terminal_exact_leaf_intervals\": "<<refinement_failed_cells
        <<",\n  \"terminal_interpolated_pixels\": "<<interpolated_pixels
        <<",\n  \"terminal_exact_output_pixels\": "<<curvature_fallback_pixels
        <<",\n  \"terminal_raster_ms\": "<<terminal_raster_ms
        <<",\n  \"build_ms\": "<<build_ms<<",\n  \"operator_ms\": "<<operator_ms<<",\n  \"march_ms\": "<<march_ms
        <<",\n  \"raster_ms\": "<<raster_ms<<",\n  \"write_ms\": "<<write_ms<<",\n  \"total_ms\": "<<total_ms
        <<",\n  \"evaluator\": {\n    \"surface_hits\": "<<evaluation_stats.surface_hits
        <<",\n    \"relation_evaluations\": "<<evaluation_stats.relation_evaluations
        <<",\n    \"plane_evaluations\": "<<evaluation_stats.plane_evaluations
        <<",\n    \"sphere_evaluations\": "<<evaluation_stats.sphere_evaluations
        <<",\n    \"blocker_candidates\": "<<evaluation_stats.blocker_candidates
        <<",\n    \"blocker_overlaps\": "<<evaluation_stats.blocker_overlaps
        <<",\n    \"unoccluded_fast_paths\": "<<evaluation_stats.unoccluded_fast_paths
        <<",\n    \"clipped_source_integrals\": "<<evaluation_stats.clipped_source_integrals
        <<",\n    \"blocked_intervals\": "<<evaluation_stats.blocked_intervals<<"\n  }"
        <<",\n  \"depth_energy\": [";for(std::size_t i=0;i<marched.depth_energy.size();++i){if(i)std::cout<<",";std::cout<<marched.depth_energy[i];}
    std::cout<<"],\n  \"output\": \""<<out<<"\"\n}\n";return 0;
}catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}
