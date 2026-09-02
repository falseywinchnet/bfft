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

namespace {

using Clock=std::chrono::steady_clock;
using RGB=std::array<double,3>;
constexpr double pi=std::numbers::pi_v<double>;

struct Vec2{double x=0,y=0;};
struct Vec3{double x=0,y=0,z=0;};
Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
Vec3 operator*(Vec3 a,double s){return {a.x*s,a.y*s,a.z*s};}
Vec3 operator*(double s,Vec3 a){return a*s;}
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
    add_sphere(s,GreenSphere,{.15,.48,-1.20},.48,{.08,.72,.20});return s;}

struct SurfacePoint{Vec3 position{},normal{};double jacobian=0;int face=0;Vec2 uv{};};
Vec3 cube_direction(int face,double a,double b){
    switch(face){case 0:return unit({1,b,-a});case 1:return unit({-1,b,a});
        case 2:return unit({a,1,-b});case 3:return unit({a,-1,b});
        case 4:return unit({a,b,1});default:return unit({-a,b,-1});}
}
SurfacePoint chart_point(const Object& o,int face,double u,double v){SurfacePoint p;p.face=face;p.uv={u,v};
    if(o.shape==Shape::Plane){p.position=o.origin+o.u*u+o.v*v;p.normal=o.normal;p.jacobian=o.area;return p;}
    const double a=2*u-1,b=2*v-1;const Vec3 n=cube_direction(face,a,b);
    p.position=o.center+n*o.radius;p.normal=n;
    p.jacobian=4*o.radius*o.radius/std::pow(1+a*a+b*b,1.5);return p;
}
SurfacePoint sphere_chart(const Object& o,Vec3 position){SurfacePoint p;p.position=position;p.normal=unit(position-o.center);
    const Vec3 n=p.normal;const double ax=std::abs(n.x),ay=std::abs(n.y),az=std::abs(n.z);double a=0,b=0;
    if(ax>=ay&&ax>=az){if(n.x>0){p.face=0;a=-n.z/n.x;b=n.y/n.x;}
        else{p.face=1;const double q=-n.x;a=n.z/q;b=n.y/q;}}
    else if(ay>=az){if(n.y>0){p.face=2;a=n.x/n.y;b=-n.z/n.y;}
        else{p.face=3;const double q=-n.y;a=n.x/q;b=n.z/q;}}
    else{if(n.z>0){p.face=4;a=n.x/n.z;b=n.y/n.z;}
        else{p.face=5;const double q=-n.z;a=-n.x/q;b=n.y/q;}}
    p.uv={(a+1)*.5,(b+1)*.5};return p;
}

struct Hit{bool valid=false;int object=-1;double t=0;SurfacePoint surface{};};
Hit intersect(const Object& o,int id,Vec3 origin,Vec3 ray){Hit h;h.object=id;
    if(o.shape==Shape::Plane){const double den=dot(o.normal,ray);if(std::abs(den)<1e-12)return h;
        const double t=dot(o.origin-origin,o.normal)/den;if(t<=1e-6)return h;const Vec3 point=origin+ray*t;
        const Vec3 d=point-o.origin;const double u=dot(d,o.u)/dot(o.u,o.u),v=dot(d,o.v)/dot(o.v,o.v);
        if(u<0||u>1||v<0||v>1)return h;h.valid=true;h.t=t;h.surface=chart_point(o,0,u,v);return h;}
    const Vec3 rel=origin-o.center;const double b=dot(ray,rel),c=dot(rel,rel)-o.radius*o.radius;
    const double disc=b*b-c;if(disc<0)return h;const double root=std::sqrt(disc);
    double t=-b-root;if(t<=1e-6)t=-b+root;if(t<=1e-6)return h;
    h.valid=true;h.t=t;h.surface=sphere_chart(o,origin+ray*t);return h;}
Hit first_hit(const Scene& s,Vec3 origin,Vec3 ray){Hit best;best.t=std::numeric_limits<double>::infinity();
    for(int i=0;i<static_cast<int>(s.objects.size());++i){Hit h=intersect(s.objects[i],i,origin,ray);
        if(h.valid&&h.t<best.t)best=h;}return best;}

double circle_overlap(double r,double R,double d){
    if(d>=r+R)return 0;if(d<=std::abs(R-r)){const double q=std::min(r,R);return pi*q*q;}
    const double a=std::acos(std::clamp((d*d+r*r-R*R)/(2*d*r),-1.0,1.0));
    const double b=std::acos(std::clamp((d*d+R*R-r*r)/(2*d*R),-1.0,1.0));
    const double q=.5*std::sqrt(std::max(0.0,(-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)));
    return r*r*a+R*R*b-q;
}
double visibility_fraction(const Scene& scene,int source,int receiver,Vec3 source_point,Vec3 target){
    const Vec3 to_source=source_point-target;const double source_distance=norm(to_source);
    if(source_distance<=1e-8)return 0;const Vec3 source_direction=to_source/source_distance;
    const double source_angle=std::atan2(scene.objects[source].extent,source_distance);
    double visible=1.0;
    for(int k=0;k<static_cast<int>(scene.objects.size());++k){if(k==source||k==receiver)continue;
        const Object& blocker=scene.objects[k];if(blocker.shape!=Shape::Sphere)continue;
        const Vec3 to_blocker=blocker.center-target;const double distance=norm(to_blocker);
        if(distance<=blocker.radius||distance>=source_distance)continue;
        const double blocker_angle=std::asin(std::clamp(blocker.radius/distance,0.0,1.0));
        const double separation=std::acos(std::clamp(dot(source_direction,to_blocker/distance),-1.0,1.0));
        double blocked=0;
        if(source_angle<1e-7)blocked=separation<blocker_angle?1.0:0.0;
        else blocked=circle_overlap(source_angle,blocker_angle,separation)/(pi*source_angle*source_angle);
        visible*=1.0-std::clamp(blocked,0.0,1.0);
    }
    return std::clamp(visible,0.0,1.0);
}

bool segment_visible(const Scene& scene,int source,int receiver,Vec3 target,Vec3 source_point){
    const Vec3 segment=source_point-target;const double length_squared=dot(segment,segment);
    if(length_squared<=1e-16)return true;
    for(int k=0;k<static_cast<int>(scene.objects.size());++k){if(k==source||k==receiver)continue;
        const Object& blocker=scene.objects[k];if(blocker.shape!=Shape::Sphere)continue;
        const double t=dot(blocker.center-target,segment)/length_squared;
        if(t<=1e-7||t>=1-1e-7)continue;const Vec3 nearest=target+segment*t;
        if(dot(nearest-blocker.center,nearest-blocker.center)<blocker.radius*blocker.radius)return false;}
    return true;
}

double plane_cell_geometry(const Object& source,const SurfacePoint& target,
                           double u0,double u1,double v0,double v1){
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
        const bool inside_a=da>1e-12,inside_b=db>1e-12;
        if(inside_a)clipped[clipped_count++]=a;
        if(inside_a!=inside_b){const double t=da/(da-db);clipped[clipped_count++]=a+(b-a)*t;}}
    if(clipped_count<3)return 0;

    Vec3 edge_sum{};
    for(int i=0;i<clipped_count;++i){const Vec3 a=unit(clipped[i]-target.position);
        const Vec3 b=unit(clipped[(i+1)%clipped_count]-target.position);const Vec3 edge=cross(a,b);
        const double sine=norm(edge);if(sine<=1e-14)continue;
        const double angle=std::atan2(sine,std::clamp(dot(a,b),-1.0,1.0));edge_sum=edge_sum+edge*(angle/sine);}
    const double cosine_integral=.5*dot(target.normal,edge_sum);
    return std::max(cosine_integral,0.0)/(pi*source.area);
}

double adaptive_plane_density(const Scene& scene,int source,int receiver,const SurfacePoint& target,
                              double u0=0,double u1=1,double v0=0,double v1=1,int level=0){
    const Object& plane=scene.objects[source];const double geometry=plane_cell_geometry(plane,target,u0,u1,v0,v1);
    if(geometry<=1e-16)return 0;
    static constexpr std::array<Vec2,5> probes{{{0,0},{1,0},{1,1},{0,1},{.5,.5}}};
    int visible=0;for(const Vec2 probe:probes){const Vec3 point=plane.origin+
            plane.u*(u0+probe.x*(u1-u0))+plane.v*(v0+probe.y*(v1-v0));
        visible+=segment_visible(scene,source,receiver,target.position,point)?1:0;}
    constexpr int minimum_level=2,maximum_level=7;
    if(level>=minimum_level&&visible==5)return geometry;
    if(level>=minimum_level&&visible==0)return 0;
    if(level>=maximum_level)return geometry*(visible/5.0);
    const double um=.5*(u0+u1),vm=.5*(v0+v1);
    return adaptive_plane_density(scene,source,receiver,target,u0,um,v0,vm,level+1)+
        adaptive_plane_density(scene,source,receiver,target,um,u1,v0,vm,level+1)+
        adaptive_plane_density(scene,source,receiver,target,um,u1,vm,v1,level+1)+
        adaptive_plane_density(scene,source,receiver,target,u0,um,vm,v1,level+1);
}

struct KernelSample{double density=0,param_density=0,visibility=0;};
KernelSample relation_kernel(const Scene& scene,int source,int receiver,const SurfacePoint& target){
    const Object& a=scene.objects[source];Vec3 source_point=a.center;
    double geometry=0.0,visibility=1.0;
    if(a.shape==Shape::Sphere){
        static constexpr std::array<double,2> nodes{{.2113248654051871,.7886751345948129}};
        for(int face=0;face<6;++face)for(double v:nodes)for(double u:nodes){
            const SurfacePoint sample=chart_point(a,face,u,v);const Vec3 delta=target.position-sample.position;
            const double distance=norm(delta);if(distance<=1e-7)continue;const Vec3 direction=delta/distance;
            const double ca=std::max(dot(sample.normal,direction),0.0);
            const double cb=std::max(dot(target.normal,-1.0*direction),0.0);
            geometry+=.25*sample.jacobian*ca*cb/(a.area*pi*distance*distance);}}
    else{SurfacePoint shifted=target;shifted.position=shifted.position+shifted.normal*1e-7;
        const double unoccluded=plane_cell_geometry(a,shifted,0,1,0,1);
        geometry=adaptive_plane_density(scene,source,receiver,shifted);
        visibility=unoccluded>1e-16?std::clamp(geometry/unoccluded,0.0,1.0):0.0;}
    if(geometry<=0)return {};
    if(a.shape==Shape::Sphere)visibility=visibility_fraction(scene,source,receiver,source_point,target.position);
    const double density=visibility*geometry;
    return {density,density*target.jacobian,visibility};
}

struct WindowNode{
    double u0=0,u1=1,v0=0,v1=1;std::array<int,4> child{{-1,-1,-1,-1}};
    std::array<double,4> density{};double mass=0;int level=0;bool leaf=true;
};
struct Relation{
    int source=-1,receiver=-1;std::vector<int> roots;std::vector<WindowNode> nodes;
    double factor=0,scale=1;std::uint64_t splits=0,blocked=0,mixed_leaves=0;
};
struct Limits{std::uint64_t max_nodes=1200000;double max_build_seconds=60;};

struct RelationBuilder{
    const Scene& scene;Relation& relation;const Limits& limits;Clock::time_point deadline;
    std::uint64_t& total_nodes;
    int min_level=2,max_level=8;
    double center_tolerance=.06,variation_tolerance=.30;
    void guard()const{if(total_nodes>=limits.max_nodes)throw std::runtime_error("global window-node ceiling reached");
        if(Clock::now()>=deadline)throw std::runtime_error("window build-time ceiling reached");}
    int build(int face,double u0,double u1,double v0,double v1,int level){guard();
        static constexpr std::array<Vec2,13> probes{{{0,0},{1,0},{1,1},{0,1},{.5,.5},
            {.5,0},{1,.5},{.5,1},{0,.5},{.25,.25},{.75,.25},{.75,.75},{.25,.75}}};
        std::array<KernelSample,13> samples{};double maximum=0,minimum=std::numeric_limits<double>::infinity();
        double min_visibility=1,max_visibility=0;
        for(int k=0;k<13;++k){const double u=u0+probes[k].x*(u1-u0),v=v0+probes[k].y*(v1-v0);
            samples[k]=relation_kernel(scene,relation.source,relation.receiver,
                chart_point(scene.objects[relation.receiver],face,u,v));
            maximum=std::max(maximum,samples[k].density);minimum=std::min(minimum,samples[k].density);
            min_visibility=std::min(min_visibility,samples[k].visibility);max_visibility=std::max(max_visibility,samples[k].visibility);}
        if(level>=min_level&&maximum<=1e-14){++relation.blocked;return -1;}
        const double bilinear=.25*(samples[0].density+samples[1].density+samples[2].density+samples[3].density);
        const double center_error=std::abs(samples[4].density-bilinear)/std::max(maximum,1e-14);
        const double variation=(maximum-minimum)/std::max(maximum,1e-14);
        const bool visibility_mixed=min_visibility<1e-6&&max_visibility>1e-4;
        bool accept=level>=min_level&&!visibility_mixed&&center_error<center_tolerance&&
                    variation<variation_tolerance;
        if(level>=max_level)accept=true;
        if(accept){WindowNode node;node.u0=u0;node.u1=u1;node.v0=v0;node.v1=v1;node.level=level;
            node.density={samples[0].density,samples[1].density,samples[2].density,samples[3].density};
            if(level>=max_level&&visibility_mixed)++relation.mixed_leaves;
            const double simpson=(samples[0].param_density+samples[1].param_density+samples[2].param_density+
                samples[3].param_density+4*(samples[5].param_density+samples[6].param_density+
                samples[7].param_density+samples[8].param_density)+16*samples[4].param_density)/36.0;
            node.mass=simpson*(u1-u0)*(v1-v0);const int index=static_cast<int>(relation.nodes.size());
            relation.nodes.push_back(node);++total_nodes;relation.factor+=node.mass;return index;}
        ++relation.splits;const double um=.5*(u0+u1),vm=.5*(v0+v1);std::array<int,4> child{{
            build(face,u0,um,v0,vm,level+1),build(face,um,u1,v0,vm,level+1),
            build(face,um,u1,vm,v1,level+1),build(face,u0,um,vm,v1,level+1)}};
        if(std::all_of(child.begin(),child.end(),[](int x){return x<0;}))return -1;
        WindowNode node;node.u0=u0;node.u1=u1;node.v0=v0;node.v1=v1;node.level=level;node.leaf=false;node.child=child;
        const int index=static_cast<int>(relation.nodes.size());relation.nodes.push_back(node);++total_nodes;return index;
    }
};

enum class Edge{Left,Right,Bottom,Top};

bool split_balance_leaf(const Scene& scene,Relation& relation,int index,int face,std::uint64_t& total_nodes,
                        const Limits& limits){if(index<0||!relation.nodes[index].leaf)return false;
    if(total_nodes+4>limits.max_nodes)throw std::runtime_error("global window-node ceiling reached in conforming closure");
    const WindowNode parent=relation.nodes[index];const double um=.5*(parent.u0+parent.u1),vm=.5*(parent.v0+parent.v1);
    const int level=parent.level+1;std::array<double,9> density{};
    for(int y=0;y<3;++y)for(int x=0;x<3;++x){const double u=parent.u0+.5*x*(parent.u1-parent.u0);
        const double v=parent.v0+.5*y*(parent.v1-parent.v0);density[y*3+x]=relation_kernel(
            scene,relation.source,relation.receiver,chart_point(scene.objects[relation.receiver],face,u,v)).density;}
    auto append=[&](double u0,double u1,double v0,double v1,std::array<double,4> values){WindowNode child;
        child.u0=u0;child.u1=u1;child.v0=v0;child.v1=v1;child.level=level;child.density=values;
        const int child_index=static_cast<int>(relation.nodes.size());relation.nodes.push_back(child);
        ++total_nodes;return child_index;};
    std::array<int,4> children{{
        append(parent.u0,um,parent.v0,vm,{density[0],density[1],density[4],density[3]}),
        append(um,parent.u1,parent.v0,vm,{density[1],density[2],density[5],density[4]}),
        append(um,parent.u1,vm,parent.v1,{density[4],density[5],density[8],density[7]}),
        append(parent.u0,um,vm,parent.v1,{density[3],density[4],density[7],density[6]})}};
    relation.nodes[index].leaf=false;relation.nodes[index].child=children;++relation.splits;return true;
}

int maximum_boundary_level(const Relation& relation,int index,Edge edge){if(index<0)return -1;
    const WindowNode& node=relation.nodes[index];if(node.leaf)return node.level;std::array<int,2> children{};
    switch(edge){case Edge::Left:children={node.child[0],node.child[3]};break;
        case Edge::Right:children={node.child[1],node.child[2]};break;
        case Edge::Bottom:children={node.child[0],node.child[1]};break;
        case Edge::Top:children={node.child[3],node.child[2]};break;}
    return std::max(maximum_boundary_level(relation,children[0],edge),
        maximum_boundary_level(relation,children[1],edge));
}

bool balance_vertical(const Scene& scene,Relation& relation,int left_index,int right_index,int face,
                      std::uint64_t& total_nodes,const Limits& limits){
    if(left_index<0||right_index<0)return false;bool changed=false;
    if(relation.nodes[left_index].leaf&&maximum_boundary_level(relation,right_index,Edge::Left)>
            relation.nodes[left_index].level+1)
        changed|=split_balance_leaf(scene,relation,left_index,face,total_nodes,limits);
    if(relation.nodes[right_index].leaf&&maximum_boundary_level(relation,left_index,Edge::Right)>
            relation.nodes[right_index].level+1)
        changed|=split_balance_leaf(scene,relation,right_index,face,total_nodes,limits);
    const WindowNode left=relation.nodes[left_index],right=relation.nodes[right_index];
    if(!left.leaf&&!right.leaf){changed|=balance_vertical(scene,relation,left.child[1],right.child[0],face,total_nodes,limits);
        changed|=balance_vertical(scene,relation,left.child[2],right.child[3],face,total_nodes,limits);}return changed;
}

bool balance_horizontal(const Scene& scene,Relation& relation,int bottom_index,int top_index,int face,
                        std::uint64_t& total_nodes,const Limits& limits){
    if(bottom_index<0||top_index<0)return false;bool changed=false;
    if(relation.nodes[bottom_index].leaf&&maximum_boundary_level(relation,top_index,Edge::Bottom)>
            relation.nodes[bottom_index].level+1)
        changed|=split_balance_leaf(scene,relation,bottom_index,face,total_nodes,limits);
    if(relation.nodes[top_index].leaf&&maximum_boundary_level(relation,bottom_index,Edge::Top)>
            relation.nodes[top_index].level+1)
        changed|=split_balance_leaf(scene,relation,top_index,face,total_nodes,limits);
    const WindowNode bottom=relation.nodes[bottom_index],top=relation.nodes[top_index];
    if(!bottom.leaf&&!top.leaf){changed|=balance_horizontal(scene,relation,bottom.child[3],top.child[0],face,total_nodes,limits);
        changed|=balance_horizontal(scene,relation,bottom.child[2],top.child[1],face,total_nodes,limits);}return changed;
}

bool balance_subtree(const Scene& scene,Relation& relation,int index,int face,std::uint64_t& total_nodes,
                     const Limits& limits){if(index<0||relation.nodes[index].leaf)return false;bool changed=false;
    const auto children=relation.nodes[index].child;for(int child:children)
        changed|=balance_subtree(scene,relation,child,face,total_nodes,limits);
    changed|=balance_vertical(scene,relation,children[0],children[1],face,total_nodes,limits);
    changed|=balance_vertical(scene,relation,children[3],children[2],face,total_nodes,limits);
    changed|=balance_horizontal(scene,relation,children[0],children[3],face,total_nodes,limits);
    changed|=balance_horizontal(scene,relation,children[1],children[2],face,total_nodes,limits);return changed;
}

void constrain_boundary(Relation& relation,int index,Edge edge,double start,double end){
    if(index<0)return;WindowNode& node=relation.nodes[index];
    if(node.leaf){switch(edge){case Edge::Left:node.density[0]=start;node.density[3]=end;break;
        case Edge::Right:node.density[1]=start;node.density[2]=end;break;
        case Edge::Bottom:node.density[0]=start;node.density[1]=end;break;
        case Edge::Top:node.density[3]=start;node.density[2]=end;break;}return;}
    const double middle=.5*(start+end);
    switch(edge){case Edge::Left:
        constrain_boundary(relation,node.child[0],edge,start,middle);
        constrain_boundary(relation,node.child[3],edge,middle,end);break;
    case Edge::Right:
        constrain_boundary(relation,node.child[1],edge,start,middle);
        constrain_boundary(relation,node.child[2],edge,middle,end);break;
    case Edge::Bottom:
        constrain_boundary(relation,node.child[0],edge,start,middle);
        constrain_boundary(relation,node.child[1],edge,middle,end);break;
    case Edge::Top:
        constrain_boundary(relation,node.child[3],edge,start,middle);
        constrain_boundary(relation,node.child[2],edge,middle,end);break;}
}

void reconcile_vertical(Relation& relation,int left_index,int right_index){
    if(left_index<0&&right_index<0)return;
    if(left_index<0){constrain_boundary(relation,right_index,Edge::Left,0,0);return;}
    if(right_index<0){constrain_boundary(relation,left_index,Edge::Right,0,0);return;}
    WindowNode& left=relation.nodes[left_index];
    WindowNode& right=relation.nodes[right_index];
    if(left.leaf&&right.leaf){const double bottom=.5*(left.density[1]+right.density[0]);
        const double top=.5*(left.density[2]+right.density[3]);left.density[1]=right.density[0]=bottom;
        left.density[2]=right.density[3]=top;return;}
    if(left.leaf){constrain_boundary(relation,right_index,Edge::Left,left.density[1],left.density[2]);return;}
    if(right.leaf){constrain_boundary(relation,left_index,Edge::Right,right.density[0],right.density[3]);return;}
    reconcile_vertical(relation,left.child[1],right.child[0]);
    reconcile_vertical(relation,left.child[2],right.child[3]);
}

void reconcile_horizontal(Relation& relation,int bottom_index,int top_index){
    if(bottom_index<0&&top_index<0)return;
    if(bottom_index<0){constrain_boundary(relation,top_index,Edge::Bottom,0,0);return;}
    if(top_index<0){constrain_boundary(relation,bottom_index,Edge::Top,0,0);return;}
    WindowNode& bottom=relation.nodes[bottom_index];
    WindowNode& top=relation.nodes[top_index];
    if(bottom.leaf&&top.leaf){const double left=.5*(bottom.density[3]+top.density[0]);
        const double right=.5*(bottom.density[2]+top.density[1]);bottom.density[3]=top.density[0]=left;
        bottom.density[2]=top.density[1]=right;return;}
    if(bottom.leaf){constrain_boundary(relation,top_index,Edge::Bottom,bottom.density[3],bottom.density[2]);return;}
    if(top.leaf){constrain_boundary(relation,bottom_index,Edge::Top,top.density[0],top.density[1]);return;}
    reconcile_horizontal(relation,bottom.child[3],top.child[0]);
    reconcile_horizontal(relation,bottom.child[2],top.child[1]);
}

std::array<double,2> boundary_values(const WindowNode& node,Edge edge){switch(edge){
    case Edge::Left:return {node.density[0],node.density[3]};
    case Edge::Right:return {node.density[1],node.density[2]};
    case Edge::Bottom:return {node.density[0],node.density[1]};
    case Edge::Top:return {node.density[3],node.density[2]};}return {};}

void set_boundary_values(WindowNode& node,Edge edge,double start,double end){switch(edge){
    case Edge::Left:node.density[0]=start;node.density[3]=end;break;
    case Edge::Right:node.density[1]=start;node.density[2]=end;break;
    case Edge::Bottom:node.density[0]=start;node.density[1]=end;break;
    case Edge::Top:node.density[3]=start;node.density[2]=end;break;}}

std::array<int,2> boundary_children(const WindowNode& node,Edge edge){switch(edge){
    case Edge::Left:return {node.child[0],node.child[3]};
    case Edge::Right:return {node.child[1],node.child[2]};
    case Edge::Bottom:return {node.child[0],node.child[1]};
    case Edge::Top:return {node.child[3],node.child[2]};}return {{-1,-1}};}

bool balance_edges(const Scene& scene,Relation& relation,int a_index,Edge a_edge,int face_a,
                   int b_index,Edge b_edge,int face_b,bool reversed,std::uint64_t& total_nodes,
                   const Limits& limits){if(a_index<0||b_index<0)return false;bool changed=false;
    if(relation.nodes[a_index].leaf&&maximum_boundary_level(relation,b_index,b_edge)>
            relation.nodes[a_index].level+1)
        changed|=split_balance_leaf(scene,relation,a_index,face_a,total_nodes,limits);
    if(relation.nodes[b_index].leaf&&maximum_boundary_level(relation,a_index,a_edge)>
            relation.nodes[b_index].level+1)
        changed|=split_balance_leaf(scene,relation,b_index,face_b,total_nodes,limits);
    const WindowNode a=relation.nodes[a_index],b=relation.nodes[b_index];if(!a.leaf&&!b.leaf){
        const auto ac=boundary_children(a,a_edge),bc=boundary_children(b,b_edge);
        changed|=balance_edges(scene,relation,ac[0],a_edge,face_a,reversed?bc[1]:bc[0],b_edge,face_b,
            reversed,total_nodes,limits);
        changed|=balance_edges(scene,relation,ac[1],a_edge,face_a,reversed?bc[0]:bc[1],b_edge,face_b,
            reversed,total_nodes,limits);}return changed;
}

void reconcile_edges(Relation& relation,int a_index,Edge a_edge,int b_index,Edge b_edge,bool reversed){
    if(a_index<0&&b_index<0)return;
    if(a_index<0){constrain_boundary(relation,b_index,b_edge,0,0);return;}
    if(b_index<0){constrain_boundary(relation,a_index,a_edge,0,0);return;}
    WindowNode& a=relation.nodes[a_index];WindowNode& b=relation.nodes[b_index];
    const auto av=boundary_values(a,a_edge),bv=boundary_values(b,b_edge);
    if(a.leaf&&b.leaf){const double start=.5*(av[0]+(reversed?bv[1]:bv[0]));
        const double end=.5*(av[1]+(reversed?bv[0]:bv[1]));set_boundary_values(a,a_edge,start,end);
        set_boundary_values(b,b_edge,reversed?end:start,reversed?start:end);return;}
    if(a.leaf){constrain_boundary(relation,b_index,b_edge,reversed?av[1]:av[0],reversed?av[0]:av[1]);return;}
    if(b.leaf){constrain_boundary(relation,a_index,a_edge,reversed?bv[1]:bv[0],reversed?bv[0]:bv[1]);return;}
    const auto ac=boundary_children(a,a_edge),bc=boundary_children(b,b_edge);
    reconcile_edges(relation,ac[0],a_edge,reversed?bc[1]:bc[0],b_edge,reversed);
    reconcile_edges(relation,ac[1],a_edge,reversed?bc[0]:bc[1],b_edge,reversed);
}

std::array<Vec3,3> face_edge_directions(int face,Edge edge){std::array<Vec3,3> result{};
    for(int i=0;i<3;++i){const double t=.5*i;double u=t,v=t;
        switch(edge){case Edge::Left:u=0;break;case Edge::Right:u=1;break;
            case Edge::Bottom:v=0;break;case Edge::Top:v=1;break;}
        result[i]=cube_direction(face,2*u-1,2*v-1);}return result;
}

void stitch_sphere_faces(Relation& relation){
    static constexpr std::array<Edge,4> edges{{Edge::Left,Edge::Right,Edge::Bottom,Edge::Top}};
    for(int face_a=0;face_a<6;++face_a)for(Edge edge_a:edges){
        const auto a=face_edge_directions(face_a,edge_a);
        for(int face_b=face_a+1;face_b<6;++face_b)for(Edge edge_b:edges){const auto b=face_edge_directions(face_b,edge_b);
            if(norm(a[1]-b[1])>1e-12)continue;const bool same=norm(a[0]-b[0])<=1e-12&&norm(a[2]-b[2])<=1e-12;
            const bool reversed=norm(a[0]-b[2])<=1e-12&&norm(a[2]-b[0])<=1e-12;if(!same&&!reversed)continue;
            reconcile_edges(relation,relation.roots[face_a],edge_a,relation.roots[face_b],edge_b,reversed);}}
}

bool balance_sphere_faces(const Scene& scene,Relation& relation,std::uint64_t& total_nodes,const Limits& limits){
    static constexpr std::array<Edge,4> edges{{Edge::Left,Edge::Right,Edge::Bottom,Edge::Top}};bool changed=false;
    for(int face_a=0;face_a<6;++face_a)for(Edge edge_a:edges){const auto a=face_edge_directions(face_a,edge_a);
        for(int face_b=face_a+1;face_b<6;++face_b)for(Edge edge_b:edges){const auto b=face_edge_directions(face_b,edge_b);
            if(norm(a[1]-b[1])>1e-12)continue;const bool same=norm(a[0]-b[0])<=1e-12&&norm(a[2]-b[2])<=1e-12;
            const bool reversed=norm(a[0]-b[2])<=1e-12&&norm(a[2]-b[0])<=1e-12;if(!same&&!reversed)continue;
            changed|=balance_edges(scene,relation,relation.roots[face_a],edge_a,face_a,
                relation.roots[face_b],edge_b,face_b,reversed,total_nodes,limits);}}
    return changed;
}

void conform_subtree(Relation& relation,int index){if(index<0)return;WindowNode& node=relation.nodes[index];
    if(node.leaf)return;const auto children=node.child;for(int child:children)conform_subtree(relation,child);
    reconcile_vertical(relation,children[0],children[1]);reconcile_vertical(relation,children[3],children[2]);
    reconcile_horizontal(relation,children[0],children[3]);reconcile_horizontal(relation,children[1],children[2]);
}

double integrate_conforming_leaf(const Object& receiver,const WindowNode& node,int face){
    static constexpr std::array<double,3> points{{.1127016653792583,.5,.8872983346207417}};
    static constexpr std::array<double,3> weights{{.2777777777777778,.4444444444444444,.2777777777777778}};
    double integral=0;for(int y=0;y<3;++y)for(int x=0;x<3;++x){const double tu=points[x],tv=points[y];
        const double density=(1-tv)*((1-tu)*node.density[0]+tu*node.density[1])+
            tv*(tu*node.density[2]+(1-tu)*node.density[3]);
        const double u=node.u0+tu*(node.u1-node.u0),v=node.v0+tv*(node.v1-node.v0);
        integral+=weights[x]*weights[y]*density*chart_point(receiver,face,u,v).jacobian;}
    return integral*(node.u1-node.u0)*(node.v1-node.v0);
}

double recompute_conforming_mass(const Object& receiver,Relation& relation,int index,int face){
    if(index<0)return 0;WindowNode& node=relation.nodes[index];if(node.leaf){
        node.mass=integrate_conforming_leaf(receiver,node,face);return node.mass;}
    double mass=0;for(int child:node.child)mass+=recompute_conforming_mass(receiver,relation,child,face);return mass;
}

std::vector<Relation> build_relations(const Scene& scene,const Limits& limits){const auto start=Clock::now();
    std::vector<Relation> relations;relations.reserve(scene.objects.size()*(scene.objects.size()-1));
    std::uint64_t total_nodes=0;
    for(int source=0;source<static_cast<int>(scene.objects.size());++source)
        for(int receiver=0;receiver<static_cast<int>(scene.objects.size());++receiver){if(source==receiver)continue;
            Relation r;r.source=source;r.receiver=receiver;const int charts=scene.objects[receiver].shape==Shape::Sphere?6:1;
            r.roots.resize(charts,-1);RelationBuilder builder{scene,r,limits,
                start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(limits.max_build_seconds)),
                total_nodes};
            if(source==AreaLight){builder.min_level=2;builder.max_level=8;
                builder.center_tolerance=.05;builder.variation_tolerance=.25;}
            else{builder.min_level=1;builder.max_level=5;
                builder.center_tolerance=.15;builder.variation_tolerance=.70;}
            for(int face=0;face<charts;++face)r.roots[face]=builder.build(face,0,1,0,1,0);
            for(int pass=0;pass<12;++pass){bool changed=false;
                for(int face=0;face<charts;++face)changed|=balance_subtree(
                    scene,r,r.roots[face],face,total_nodes,limits);
                if(charts==6)changed|=balance_sphere_faces(scene,r,total_nodes,limits);
                if(!changed)break;
                if(Clock::now()>=builder.deadline)throw std::runtime_error("window build-time ceiling reached in conforming closure");}
            for(int pass=0;pass<3;++pass){for(int root:r.roots)conform_subtree(r,root);
                if(charts==6)stitch_sphere_faces(r);}r.factor=0;
            for(int face=0;face<charts;++face)r.factor+=recompute_conforming_mass(
                scene.objects[receiver],r,r.roots[face],face);
            if(r.factor>1e-14)relations.push_back(std::move(r));}
    std::vector<double> row(scene.objects.size(),0.0);for(const auto& r:relations)row[r.source]+=r.factor;
    for(auto& r:relations){r.scale=std::min(1.0,.90/std::max(row[r.source],1e-30));r.factor*=r.scale;}
    return relations;
}

double query_node(const Relation& r,int index,double u,double v){if(index<0)return 0;const WindowNode& n=r.nodes[index];
    if(n.leaf){const double tu=std::clamp((u-n.u0)/std::max(n.u1-n.u0,1e-15),0.0,1.0);
        const double tv=std::clamp((v-n.v0)/std::max(n.v1-n.v0,1e-15),0.0,1.0);
        return std::max(0.0,(1-tv)*((1-tu)*n.density[0]+tu*n.density[1])+
            tv*(tu*n.density[2]+(1-tu)*n.density[3]));}
    const double um=.5*(n.u0+n.u1),vm=.5*(n.v0+n.v1);int quadrant;
    if(v<vm)quadrant=u<um?0:1;else quadrant=u<um?3:2;return query_node(r,n.child[quadrant],u,v);
}
double relation_density(const Relation& r,const SurfacePoint& p){if(p.face<0||p.face>=static_cast<int>(r.roots.size()))return 0;
    return r.scale*query_node(r,r.roots[p.face],p.uv.x,p.uv.y);}

void audit_continuity_node(const Relation& relation,int root,int index,double& maximum_jump,double& maximum_value){
    if(index<0)return;const WindowNode& node=relation.nodes[index];if(!node.leaf){
        for(int child:node.child)audit_continuity_node(relation,root,child,maximum_jump,maximum_value);return;}
    constexpr double epsilon=1e-10;const double um=.5*(node.u0+node.u1),vm=.5*(node.v0+node.v1);
    auto compare=[&](double ua,double va,double ub,double vb){const double a=query_node(relation,root,ua,va);
        const double b=query_node(relation,root,ub,vb);maximum_jump=std::max(maximum_jump,std::abs(a-b));
        maximum_value=std::max({maximum_value,std::abs(a),std::abs(b)});};
    if(node.u0>epsilon)compare(node.u0-epsilon,vm,node.u0+epsilon,vm);
    if(node.u1<1-epsilon)compare(node.u1-epsilon,vm,node.u1+epsilon,vm);
    if(node.v0>epsilon)compare(um,node.v0-epsilon,um,node.v0+epsilon);
    if(node.v1<1-epsilon)compare(um,node.v1-epsilon,um,node.v1+epsilon);
}

struct Coupling{int outgoing=-1;double weight=0;};
struct WindowOperator{std::vector<std::vector<Coupling>> from_incoming;std::uint64_t nonzeros=0;};

double relation_overlap(const Object& receiver,const Relation& a,const Relation& b){
    const int charts=receiver.shape==Shape::Sphere?6:1;
    const int resolution=receiver.shape==Shape::Sphere?24:48;
    const double step=1.0/resolution;double overlap=0;
    for(int face=0;face<charts;++face)for(int y=0;y<resolution;++y)for(int x=0;x<resolution;++x){
        const SurfacePoint p=chart_point(receiver,face,(x+.5)*step,(y+.5)*step);
        overlap+=relation_density(a,p)*relation_density(b,p)*p.jacobian*step*step;}
    return overlap;
}

WindowOperator build_window_operator(const Scene& scene,const std::vector<Relation>& relations){
    const int object_count=static_cast<int>(scene.objects.size());
    std::vector<std::vector<int>> relation_index(object_count,std::vector<int>(object_count,-1));
    for(int i=0;i<static_cast<int>(relations.size());++i)
        relation_index[relations[i].source][relations[i].receiver]=i;
    WindowOperator op;op.from_incoming.resize(relations.size());
    for(int incoming=0;incoming<static_cast<int>(relations.size());++incoming){
        const Relation& r=relations[incoming];const int surface=r.receiver;
        if(r.factor<=1e-14)continue;double column_sum=0;
        for(int outgoing=0;outgoing<static_cast<int>(relations.size());++outgoing){
            const Relation& s=relations[outgoing];if(s.source!=surface||s.factor<=1e-14)continue;
            const int reverse_index=relation_index[s.receiver][surface];if(reverse_index<0)continue;
            const Relation& reverse=relations[reverse_index];if(reverse.factor<=1e-14)continue;
            const double overlap=relation_overlap(scene.objects[surface],r,reverse);
            const double weight=scene.objects[surface].area*s.factor*overlap/(r.factor*reverse.factor);
            if(weight<=1e-12)continue;
            op.from_incoming[incoming].push_back({outgoing,weight});column_sum+=weight;}
        const double scale=column_sum>.90?.90/column_sum:1.0;
        for(Coupling& c:op.from_incoming[incoming])c.weight*=scale;
        op.nonzeros+=op.from_incoming[incoming].size();}
    return op;
}

struct MarchResult{std::vector<RGB> total_delivered;std::vector<double> depth_energy;int depths=0;};
MarchResult march(const Scene& scene,const std::vector<Relation>& relations,const WindowOperator& op,
                  int max_depth,double threshold){
    std::vector<RGB> frontier(relations.size()),total(relations.size());
    for(int i=0;i<static_cast<int>(relations.size());++i){const Relation& r=relations[i];
        const Object& source=scene.objects[r.source];const RGB emitted=source.emission*(source.area*pi);
        frontier[i]=total[i]=emitted*r.factor;}
    MarchResult result;
    for(int depth=0;depth<max_depth;++depth){double active=0;for(const RGB& q:frontier)active+=energy(q);
        result.depth_energy.push_back(active);if(active<=threshold)break;std::vector<RGB> next(relations.size());
        for(int incoming=0;incoming<static_cast<int>(relations.size());++incoming){
            if(energy(frontier[incoming])<=0)continue;
            const RGB reflected=multiply(frontier[incoming],scene.objects[relations[incoming].receiver].albedo);
            for(const Coupling& c:op.from_incoming[incoming])next[c.outgoing]+=reflected*c.weight;}
        for(int i=0;i<static_cast<int>(relations.size());++i)total[i]+=next[i];
        frontier.swap(next);++result.depths;}
    result.total_delivered=std::move(total);return result;
}

RGB surface_radiance(const Scene& scene,const std::vector<Relation>& relations,
                     const MarchResult& march,int receiver,const SurfacePoint& point){RGB incident{};
    for(int i=0;i<static_cast<int>(relations.size());++i){const Relation& r=relations[i];
        if(r.receiver==receiver&&r.factor>1e-14)
            incident+=march.total_delivered[i]*(relation_density(r,point)/r.factor);}
    return scene.objects[receiver].emission+multiply(incident,scene.objects[receiver].albedo)*(1.0/pi);
}

std::vector<std::uint8_t> render(const Scene& scene,const std::vector<Relation>& relations,
                                 const MarchResult& march,int width,int height){
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width)*height*3,0);
    const Vec3 camera{0,2.15,6.8},target{0,1.55,-2.15};const Vec3 forward=unit(target-camera);
    const Vec3 right=unit(cross(forward,{0,1,0})),up=cross(right,forward);
    const double scale=std::tan(24*pi/180.0),aspect=double(width)/height;std::atomic<int> next_row{0};
    std::vector<std::thread> threads;const unsigned workers=std::max(1u,std::thread::hardware_concurrency());
    for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&]{for(;;){const int y=next_row.fetch_add(1);
        if(y>=height)break;for(int x=0;x<width;++x){const double px=(2*(x+.5)/width-1)*aspect*scale;
            const double py=(1-2*(y+.5)/height)*scale;const Hit hit=first_hit(scene,camera,unit(forward+right*px+up*py));
            RGB linear{};if(hit.valid)linear=surface_radiance(scene,relations,march,hit.object,hit.surface);
            const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;
            for(int c=0;c<3;++c){const double mapped=std::pow(std::clamp(1-std::exp(-.72*std::max(linear[c],0.0)),0.0,1.0),1/2.2);
                image[offset+c]=static_cast<std::uint8_t>(std::lround(255*mapped));}}}});
    for(auto& t:threads)t.join();return image;
}
void write_ppm(const std::string& path,const std::vector<std::uint8_t>& image,int width,int height){
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot open output");
    out<<"P6\n"<<width<<" "<<height<<"\n255\n";out.write(reinterpret_cast<const char*>(image.data()),image.size());}

bool self_test(){const Scene scene=build_scene();Limits limits;limits.max_nodes=400000;limits.max_build_seconds=30;
    const SurfacePoint floor_probe=chart_point(scene.objects[Floor],0,.37,.41);
    const double whole=plane_cell_geometry(scene.objects[AreaLight],floor_probe,0,1,0,1);
    const double quarters=plane_cell_geometry(scene.objects[AreaLight],floor_probe,0,.5,0,.5)+
        plane_cell_geometry(scene.objects[AreaLight],floor_probe,.5,1,0,.5)+
        plane_cell_geometry(scene.objects[AreaLight],floor_probe,.5,1,.5,1)+
        plane_cell_geometry(scene.objects[AreaLight],floor_probe,0,.5,.5,1);
    if(whole<=0||std::abs(whole-quarters)>1e-10*whole)return false;
    const auto relations=build_relations(scene,limits);if(relations.empty())return false;
    std::vector<double> row(scene.objects.size(),0);std::uint64_t mixed=0;for(const auto& r:relations){
        if(r.factor<0)return false;row[r.source]+=r.factor;mixed+=r.mixed_leaves;}
    if(*std::max_element(row.begin(),row.end())>.900001||mixed==0)return false;
    double maximum_jump=0,maximum_value=0;for(const Relation& relation:relations)
        for(int root:relation.roots)audit_continuity_node(relation,root,root,maximum_jump,maximum_value);
    if(maximum_jump>1e-7*std::max(maximum_value,1.0)){std::cerr<<"continuity jump "<<maximum_jump
        <<" at scale "<<maximum_value<<"\n";return false;}
    const auto op=build_window_operator(scene,relations);if(op.nonzeros==0)return false;
    for(const auto& column:op.from_incoming){double sum=0;for(const auto& c:column){if(c.weight<0)return false;sum+=c.weight;}
        if(sum>.900001)return false;}
    const auto result=march(scene,relations,op,8,1e-6);if(result.depth_energy.size()<2)return false;
    std::cout<<"window transport invariants: ok ("<<relations.size()<<" relations, "<<op.nonzeros
        <<" supported couplings, "<<mixed<<" boundary leaves)\n";return true;}

} // namespace

int main(int argc,char** argv)try{int width=800,height=600,max_depth=10;double threshold=1e-5;
    std::string out="/tmp/window_transport.ppm";bool test=false;Limits limits;
    for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--self-test"){test=true;continue;}
        if(i+1>=argc)throw std::runtime_error("missing argument value");
        if(arg=="--width")width=std::stoi(argv[++i]);else if(arg=="--height")height=std::stoi(argv[++i]);
        else if(arg=="--depth")max_depth=std::stoi(argv[++i]);else if(arg=="--threshold")threshold=std::stod(argv[++i]);
        else if(arg=="--max-nodes")limits.max_nodes=std::stoull(argv[++i]);
        else if(arg=="--max-build-seconds")limits.max_build_seconds=std::stod(argv[++i]);
        else if(arg=="--out")out=argv[++i];else throw std::runtime_error("unknown argument: "+arg);}
    if(test)return self_test()?0:1;const auto total_start=Clock::now();const Scene scene=build_scene();
    const auto build_start=Clock::now();const auto relations=build_relations(scene,limits);
    const double build_ms=std::chrono::duration<double,std::milli>(Clock::now()-build_start).count();
    const auto operator_start=Clock::now();const auto window_operator=build_window_operator(scene,relations);
    const double operator_ms=std::chrono::duration<double,std::milli>(Clock::now()-operator_start).count();
    const auto march_start=Clock::now();const auto marched=march(scene,relations,window_operator,max_depth,threshold);
    const double march_ms=std::chrono::duration<double,std::milli>(Clock::now()-march_start).count();
    const auto raster_start=Clock::now();const auto image=render(scene,relations,marched,width,height);
    const double raster_ms=std::chrono::duration<double,std::milli>(Clock::now()-raster_start).count();
    const auto write_start=Clock::now();write_ppm(out,image,width,height);
    const double write_ms=std::chrono::duration<double,std::milli>(Clock::now()-write_start).count();
    std::uint64_t nodes=0,leaves=0,splits=0,blocked=0,mixed=0;for(const auto& r:relations){nodes+=r.nodes.size();
        splits+=r.splits;blocked+=r.blocked;mixed+=r.mixed_leaves;for(const auto& n:r.nodes)if(n.leaf)++leaves;}
    const double total_ms=std::chrono::duration<double,std::milli>(Clock::now()-total_start).count();
    std::cout<<std::fixed<<std::setprecision(3)<<"{\n  \"width\": "<<width<<",\n  \"height\": "<<height
        <<",\n  \"analytic_objects\": "<<scene.objects.size()<<",\n  \"saved_relations\": "<<relations.size()
        <<",\n  \"window_nodes\": "<<nodes<<",\n  \"visible_leaves\": "<<leaves
        <<",\n  \"window_couplings\": "<<window_operator.nonzeros
        <<",\n  \"beam_splits\": "<<splits<<",\n  \"blocked_cells_pruned\": "<<blocked
        <<",\n  \"mixed_boundary_leaves\": "<<mixed<<",\n  \"march_depths\": "<<marched.depths
        <<",\n  \"build_ms\": "<<build_ms<<",\n  \"operator_ms\": "<<operator_ms
        <<",\n  \"march_ms\": "<<march_ms
        <<",\n  \"raster_ms\": "<<raster_ms<<",\n  \"write_ms\": "<<write_ms
        <<",\n  \"total_ms\": "<<total_ms<<",\n  \"depth_energy\": [";
    for(std::size_t i=0;i<marched.depth_energy.size();++i){if(i)std::cout<<",";std::cout<<marched.depth_energy[i];}
    std::cout<<"],\n  \"output\": \""<<out<<"\"\n}\n";return 0;
}catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}
