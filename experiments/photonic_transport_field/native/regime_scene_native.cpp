#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

extern "C" int conv_resize_lines_f32(const float*,int,int,float*,int);
extern "C" int conv_evaluate_profile_f32(const float*,int,int,const float*,float*,int);
extern "C" int conv_prepare_profile_f32(const float*,int,int,float*);
extern "C" int conv_evaluate_prepared_profile_f32(
    const float*,const float*,int,int,const float*,float*,int);
extern "C" int conv_resize_prepared_lines_f32(const float*,const float*,int,int,float*,int);
extern "C" int conv_evaluate_profile_2d_f32(
    const float*,int,int,int,const float*,const float*,float*,int);

namespace {

using Clock=std::chrono::steady_clock;
using RGB=std::array<double,3>;
constexpr double pi=std::numbers::pi_v<double>;

struct Vec3{double x=0,y=0,z=0;};
struct Vec2{double x=0,y=0;};
Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
Vec3 operator-(Vec3 a){return {-a.x,-a.y,-a.z};}
Vec3 operator*(Vec3 a,double s){return {a.x*s,a.y*s,a.z*s};}
Vec3 operator/(Vec3 a,double s){return {a.x/s,a.y/s,a.z/s};}
double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double norm2(Vec3 a){return dot(a,a);}double norm(Vec3 a){return std::sqrt(norm2(a));}
Vec3 unit(Vec3 a){const double n=norm(a);return n>1e-15?a/n:Vec3{};}
Vec3 reflect(Vec3 d,Vec3 n){return d-n*(2*dot(d,n));}
bool refract(Vec3 d,Vec3 n,double eta,Vec3& out){const double c=std::clamp(-dot(d,n),0.0,1.0);
    const double k=1-eta*eta*(1-c*c);if(k<0)return false;out=unit(d*eta+n*(eta*c-std::sqrt(k)));return true;}
RGB operator+(RGB a,const RGB& b){for(int c=0;c<3;++c)a[c]+=b[c];return a;}
RGB operator*(RGB a,double s){for(double& v:a)v*=s;return a;}
RGB multiply(RGB a,const RGB& b){for(int c=0;c<3;++c)a[c]*=b[c];return a;}
RGB& operator+=(RGB& a,const RGB& b){a=a+b;return a;}
RGB clamp_rgb(RGB a,double lo=0,double hi=1e9){for(double& v:a)v=std::clamp(v,lo,hi);return a;}

enum class MaterialKind{Diffuse,Glossy,Mirror,Metal,Dielectric,Emissive};
struct Material{
    std::string name;MaterialKind kind=MaterialKind::Diffuse;RGB base{.7,.7,.7},emission{};
    double diffuse=.9,roughness=.4,metallic=0,ior=1.5;RGB ior_rgb{1.5,1.5,1.5};
    RGB absorption{};bool thin=false;
};
enum class Shape{Rectangle,Sphere,Triangle};
struct Primitive{
    std::string name;Shape shape=Shape::Rectangle;int material=-1;bool transport=true;
    Vec3 origin{},u{},v{},normal{},center{},a{},b{},c{},edge1{},edge2{};
    double radius=0,radius2=0,area=0,gram_uu=0,gram_uv=0,gram_vv=0,gram_det=0;
};
struct Bounds3{
    Vec3 lower{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    Vec3 upper{-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
};
struct BvhNode{Bounds3 bounds{};int left=-1,right=-1,begin=0,count=0;};
struct AreaLight{int primitive=-1;RGB radiance{};};
struct BeamBundle{std::string name;Vec3 origin{},direction{},axis_u{},axis_v{};double radius=.08,spread=.003;RGB power{};int fibres=19;};
struct Scene{std::vector<Material> materials;std::vector<Primitive> primitives;std::vector<AreaLight> area_lights;
    std::vector<BeamBundle> beams;int floor=-1,prism_volume=-1,prism_bottom=-1,prism_top=-1;
    int glass_sheet=-1,mirror_panel=-1,cavity_target=-1,projector_lens=-1;
    std::vector<int> bvh_primitives;std::vector<BvhNode> bvh_nodes;bool use_bvh=true;};

double coordinate(Vec3 value,int axis){return axis==0?value.x:(axis==1?value.y:value.z);}
void expand(Bounds3& bounds,Vec3 point){bounds.lower.x=std::min(bounds.lower.x,point.x);
    bounds.lower.y=std::min(bounds.lower.y,point.y);bounds.lower.z=std::min(bounds.lower.z,point.z);
    bounds.upper.x=std::max(bounds.upper.x,point.x);bounds.upper.y=std::max(bounds.upper.y,point.y);
    bounds.upper.z=std::max(bounds.upper.z,point.z);}
void expand(Bounds3& bounds,const Bounds3& other){expand(bounds,other.lower);expand(bounds,other.upper);}
Bounds3 primitive_bounds(const Primitive& primitive){Bounds3 bounds;
    if(primitive.shape==Shape::Sphere){const Vec3 radius{primitive.radius,primitive.radius,primitive.radius};
        expand(bounds,primitive.center-radius);expand(bounds,primitive.center+radius);}
    else if(primitive.shape==Shape::Rectangle){expand(bounds,primitive.origin);expand(bounds,primitive.origin+primitive.u);
        expand(bounds,primitive.origin+primitive.v);expand(bounds,primitive.origin+primitive.u+primitive.v);}
    else{expand(bounds,primitive.a);expand(bounds,primitive.b);expand(bounds,primitive.c);}
    constexpr double epsilon=1e-8;bounds.lower=bounds.lower-Vec3{epsilon,epsilon,epsilon};
    bounds.upper=bounds.upper+Vec3{epsilon,epsilon,epsilon};return bounds;}

void build_scene_bvh(Scene& scene){scene.bvh_primitives.resize(scene.primitives.size());
    for(int i=0;i<static_cast<int>(scene.primitives.size());++i)scene.bvh_primitives[i]=i;
    scene.bvh_nodes.clear();scene.bvh_nodes.reserve(scene.primitives.size()*2);
    auto build=[&](auto&& self,int begin,int end)->int{const int node_index=static_cast<int>(scene.bvh_nodes.size());
        scene.bvh_nodes.push_back({});Bounds3 bounds,centroids;
        for(int i=begin;i<end;++i){const Primitive& primitive=scene.primitives[scene.bvh_primitives[i]];
            expand(bounds,primitive_bounds(primitive));expand(centroids,primitive.center);}
        scene.bvh_nodes[node_index].bounds=bounds;const int count=end-begin;
        if(count<=4){scene.bvh_nodes[node_index].begin=begin;scene.bvh_nodes[node_index].count=count;return node_index;}
        const Vec3 extent=centroids.upper-centroids.lower;const int axis=extent.x>=extent.y&&extent.x>=extent.z?0:(extent.y>=extent.z?1:2);
        const int middle=begin+count/2;std::nth_element(scene.bvh_primitives.begin()+begin,scene.bvh_primitives.begin()+middle,
            scene.bvh_primitives.begin()+end,[&](int a,int b){return coordinate(scene.primitives[a].center,axis)<
                coordinate(scene.primitives[b].center,axis);});
        const int left=self(self,begin,middle),right=self(self,middle,end);scene.bvh_nodes[node_index].left=left;
        scene.bvh_nodes[node_index].right=right;return node_index;};
    if(!scene.primitives.empty())build(build,0,static_cast<int>(scene.primitives.size()));}

int add_material(Scene& s,Material m){s.materials.push_back(std::move(m));return static_cast<int>(s.materials.size())-1;}
int add_rect(Scene& s,std::string name,Vec3 origin,Vec3 u,Vec3 v,Vec3 normal,int material,bool transport=true){
    Primitive p;p.name=std::move(name);p.shape=Shape::Rectangle;p.material=material;p.transport=transport;
    p.origin=origin;p.u=u;p.v=v;p.normal=unit(normal);p.center=origin+(u+v)*.5;p.area=norm(cross(u,v));
    p.gram_uu=dot(u,u);p.gram_uv=dot(u,v);p.gram_vv=dot(v,v);p.gram_det=p.gram_uu*p.gram_vv-p.gram_uv*p.gram_uv;
    s.primitives.push_back(std::move(p));return static_cast<int>(s.primitives.size())-1;}
int add_sphere(Scene& s,std::string name,Vec3 center,double radius,int material,bool transport=true){
    Primitive p;p.name=std::move(name);p.shape=Shape::Sphere;p.material=material;p.transport=transport;
    p.center=center;p.radius=radius;p.radius2=radius*radius;p.area=4*pi*radius*radius;s.primitives.push_back(std::move(p));
    return static_cast<int>(s.primitives.size())-1;}
int add_triangle(Scene& s,std::string name,Vec3 a,Vec3 b,Vec3 c,int material,bool transport=false){
    Primitive p;p.name=std::move(name);p.shape=Shape::Triangle;p.material=material;p.transport=transport;
    p.a=a;p.b=b;p.c=c;p.edge1=b-a;p.edge2=c-a;p.center=(a+b+c)/3;p.normal=unit(cross(p.edge1,p.edge2));
    p.area=.5*norm(cross(p.edge1,p.edge2));
    s.primitives.push_back(std::move(p));return static_cast<int>(s.primitives.size())-1;}

void add_prism(Scene& s,Vec3 A,Vec3 B,Vec3 C,double y0,double y1,int glass){
    const std::array<Vec3,3> q{{A,B,C}};const Vec3 centre=(A+B+C)/3+Vec3{0,(y0+y1)*.5,0};
    const int first=static_cast<int>(s.primitives.size());
    for(int i=0;i<3;++i){const Vec3 p=q[i]+Vec3{0,y0,0},r=q[(i+1)%3]+Vec3{0,y0,0};
        const Vec3 u=r-p,v{0,y1-y0,0};Vec3 n=unit(cross(u,v));const Vec3 mid=p+(u+v)*.5;
        if(dot(n,mid-centre)<0)n=-n;add_rect(s,"prism_side_"+std::to_string(i),p,u,v,n,glass,false);}
    s.prism_bottom=add_triangle(s,"prism_cap_bottom",A+Vec3{0,y0,0},B+Vec3{0,y0,0},C+Vec3{0,y0,0},glass,false);
    s.prism_top=add_triangle(s,"prism_cap_top",A+Vec3{0,y1,0},C+Vec3{0,y1,0},B+Vec3{0,y1,0},glass,false);
    s.prism_volume=first;
}

Scene build_regime_scene(){Scene s;
    const int white=add_material(s,{"warm diffuse",MaterialKind::Diffuse,{.73,.72,.69},{},.93,.8});
    const int red=add_material(s,{"red diffuse",MaterialKind::Diffuse,{.72,.075,.045},{},.94,.75});
    const int blue=add_material(s,{"blue diffuse",MaterialKind::Diffuse,{.055,.16,.68},{},.94,.75});
    const int dark=add_material(s,{"cavity charcoal",MaterialKind::Diffuse,{.38,.41,.46},{},.95,.9});
    const int teal=add_material(s,{"indirect teal",MaterialKind::Diffuse,{.04,.82,.56},{},.96,.65});
    const int glossy=add_material(s,{"green clearcoat",MaterialKind::Glossy,{.055,.48,.12},{},.62,.11});
    const int mirror=add_material(s,{"silver mirror",MaterialKind::Mirror,{.94,.96,.99},{},0,.0,1});
    const int metal=add_material(s,{"rough copper",MaterialKind::Metal,{.92,.42,.17},{},.08,.24,1});
    Material sheet{"smoke glass sheet",MaterialKind::Dielectric,{.90,.96,1.0},{},0,.02,0,1.46,{1.455,1.46,1.468},{.08,.025,.012},true};
    const int glass_sheet=add_material(s,sheet);
    Material prism{"dispersive prism",MaterialKind::Dielectric,{.98,.99,1.0},{},0,.01,0,1.52,{1.490,1.520,1.560},{.018,.010,.006},false};
    const int prism_glass=add_material(s,prism);
    const int light=add_material(s,{"large soft emitter",MaterialKind::Emissive,{1,1,1},{26,24,21.5},0,0});
    const int projector=add_material(s,{"collimated emitter",MaterialKind::Emissive,{1,1,1},{44,40,34},0,0});
    const int black=add_material(s,{"projector housing",MaterialKind::Glossy,{.07,.075,.085},{},.38,.28});

    s.floor=add_rect(s,"floor",{-5,0,-10},{10,0,0},{0,0,12},{0,1,0},white);
    add_rect(s,"back_wall",{-5,0,-10},{10,0,0},{0,6,0},{0,0,1},white);
    add_rect(s,"left_wall",{-5,0,2},{0,0,-12},{0,6,0},{1,0,0},red);
    add_rect(s,"right_wall",{5,0,-10},{0,0,12},{0,6,0},{-1,0,0},blue);
    add_rect(s,"ceiling",{-5,6,2},{10,0,0},{0,0,-12},{0,-1,0},white);
    const int large=add_rect(s,"large_soft_light",{-1.7,5.93,-7.1},{3.4,0,0},{0,0,2.4},{0,-1,0},light,false);
    s.area_lights.push_back({large,s.materials[light].emission});

    const Vec3 mirror_u{2.101,0,-.460},mirror_v{.194,3.488,.889},mirror_n{.207,-.253,.946};
    const Vec3 mirror_origin{-4.398,.906,-9.364};
    s.mirror_panel=add_rect(s,"mirror_panel",mirror_origin,mirror_u,mirror_v,mirror_n,mirror,false);
    add_rect(s,"mirror_frame_left",mirror_origin-Vec3{.098,0,-.021},{.10,0,-.022},mirror_v,mirror_n,metal,false);
    add_rect(s,"mirror_frame_right",mirror_origin+mirror_u,{.10,0,-.022},mirror_v,mirror_n,metal,false);

    add_sphere(s,"diffuse_sphere",{-.75,.83,-4.65},.83,red);
    add_sphere(s,"glossy_sphere",{.85,.72,-6.35},.72,glossy);
    add_sphere(s,"rough_metal_sphere",{2.05,.88,-5.05},.88,metal,false);

    s.glass_sheet=add_rect(s,"glass_sheet",{.48,.32,-5.28},{2.72,0,-.76},{0,3.65,0},{.269,0,.963},glass_sheet,false);
    add_rect(s,"glass_frame_bottom",{.42,.25,-5.25},{2.85,0,-.80},{0,.08,0},{.269,0,.963},metal,false);
    add_rect(s,"glass_frame_top",{.42,4.02,-5.25},{2.85,0,-.80},{0,.08,0},{.269,0,.963},metal,false);

    // Keep an optical epsilon between coincident analytic faces: the prism is in
    // contact with the floor, while its bottom dielectric sheet remains hittable
    // before the opaque floor instead of losing the tie in first_hit().
    add_prism(s,{-2.95,0,-4.45},{-1.45,0,-4.45},{-2.18,0,-2.98},.001,1.72,prism_glass);

    add_rect(s,"thin_baffle",{-.18,0,-7.25},{.24,0,-1.65},{0,3.0,0},{.99,0,.14},glossy);

    add_rect(s,"cavity_back",{2.95,.48,-9.76},{1.55,0,0},{0,2.35,0},{0,0,1},dark);
    add_rect(s,"cavity_roof",{2.78,2.72,-9.7},{1.9,0,0},{0,0,4.55},{0,-1,0},dark);
    add_rect(s,"cavity_left",{2.78,.38,-9.7},{0,0,3.10},{0,2.42,0},{1,0,0},dark);
    add_rect(s,"cavity_right",{4.67,.38,-6.60},{0,0,-3.10},{0,2.42,0},{-1,0,0},dark);
    add_rect(s,"cavity_floor",{2.78,.38,-6.60},{1.9,0,0},{0,0,-3.10},{0,1,0},dark);
    s.cavity_target=add_sphere(s,"indirect_only_target",{4.08,1.00,-8.10},.55,teal);

    const Vec3 projector_origin{-3.65,5.42,-.45};
    const Vec3 direction=unit(Vec3{-2.16,.84,-3.72}-projector_origin);
    add_sphere(s,"projector_housing",projector_origin-direction*.13,.18,black,false);
    s.projector_lens=add_sphere(s,"projector_lens",projector_origin,.12,projector,false);
    Vec3 axis_u=unit(cross(direction,{0,1,0}));Vec3 axis_v=unit(cross(axis_u,direction));
    s.beams.push_back({"collimated_bundle",projector_origin+direction*.30,direction,axis_u,axis_v,.105,.0015,{300,250,190},19});
    build_scene_bvh(s);
    return s;
}

int material_named(const Scene& scene,const std::string& name){
    for(int i=0;i<static_cast<int>(scene.materials.size());++i)
        if(scene.materials[i].name==name)return i;
    throw std::runtime_error("missing material: "+name);
}

Scene build_demonstrator_scene(const std::string& mode){
    Scene scene=build_regime_scene();
    if(mode=="standard")return scene;
    const int white=material_named(scene,"warm diffuse");
    const int dark=material_named(scene,"cavity charcoal");
    const int mirror=material_named(scene,"silver mirror");
    const int metal=material_named(scene,"rough copper");
    if(mode=="aperture-canyon"){
        const int violet=add_material(scene,{"violet diffuse",MaterialKind::Diffuse,{.45,.08,.66},{},.95,.72});
        const int amber=add_material(scene,{"amber diffuse",MaterialKind::Diffuse,{.88,.34,.035},{},.95,.72});
        const int side_light=add_material(scene,{"violet side emitter",MaterialKind::Emissive,{1,1,1},{8.5,3.8,12.5},0,0});
        const int emitter=add_rect(scene,"violet_side_light",{-4.93,1.05,-8.70},{0,0,3.15},{0,3.20,0},
            {1,0,0},side_light,false);
        scene.area_lights.push_back({emitter,scene.materials[side_light].emission});
        for(int i=0;i<7;++i){
            const double x=-2.55+.82*i,z=-8.45+.19*(i%2),angle=(-18+6*i)*pi/180;
            const Vec3 span{.72*std::cos(angle),0,.72*std::sin(angle)};
            add_rect(scene,"aperture_fin_"+std::to_string(i),Vec3{x,0,z}-span*.5,span,{0,2.85,0},
                unit(cross(span,{0,1,0})),i%2?dark:metal,false);
        }
        for(int i=0;i<5;++i){
            const double radius=.24+.045*(i%3);
            add_sphere(scene,"aperture_receiver_"+std::to_string(i),{-2.45+1.22*i,radius,-7.15-.36*(i%2)},
                radius,i%2?violet:amber);
        }
    }else if(mode=="mirror-relay"){
        const int coral=add_material(scene,{"relay coral",MaterialKind::Diffuse,{.82,.10,.055},{},.94,.68});
        const int cyan=add_material(scene,{"relay cyan",MaterialKind::Diffuse,{.035,.58,.72},{},.94,.68});
        const int relay_light=add_material(scene,{"cyan relay emitter",MaterialKind::Emissive,{1,1,1},{3.5,10.5,12.0},0,0});
        const int emitter=add_rect(scene,"cyan_relay_light",{4.93,1.25,-8.85},{0,0,2.65},{0,2.70,0},
            {-1,0,0},relay_light,false);
        scene.area_lights.push_back({emitter,scene.materials[relay_light].emission});
        add_rect(scene,"relay_mirror_left",{-3.85,.72,-7.92},{1.65,0,.58},{0,2.45,0},{.33,0,.94},mirror,false);
        add_rect(scene,"relay_mirror_centre",{-.78,1.05,-9.68},{1.72,0,0},{0,2.15,0},{0,0,1},mirror,false);
        add_rect(scene,"relay_mirror_right",{2.25,.62,-8.30},{1.48,0,-.72},{0,2.62,0},{-.44,0,.90},mirror,false);
        add_rect(scene,"relay_receiver_coral",{-3.75,.04,-6.55},{1.20,0,0},{0,1.35,0},{0,0,1},coral);
        add_rect(scene,"relay_receiver_cyan",{2.15,.03,-6.78},{1.18,0,0},{0,1.42,0},{0,0,1},cyan);
        add_sphere(scene,"relay_metal_node",{.12,.46,-7.42},.46,metal,false);
    }else if(mode=="occlusion-garden"){
        const std::array<int,4> garden_materials{{
            add_material(scene,{"garden vermilion",MaterialKind::Diffuse,{.78,.075,.025},{},.95,.78}),
            add_material(scene,{"garden chartreuse",MaterialKind::Diffuse,{.22,.72,.045},{},.95,.78}),
            add_material(scene,{"garden cobalt",MaterialKind::Glossy,{.035,.20,.82},{},.64,.16}),
            add_material(scene,{"garden ivory",MaterialKind::Diffuse,{.78,.74,.61},{},.95,.78})}};
        for(int row=0;row<4;++row)for(int column=0;column<7;++column){
            const double radius=.13+.035*((row*3+column)%4);
            const double x=-3.55+1.05*column+.16*(row%2);
            const double y=.82+.66*row+.10*((column+row)%3);
            const double z=-6.05-.28*row-.10*std::sin(column*1.7);
            add_sphere(scene,"garden_orb_"+std::to_string(row)+"_"+std::to_string(column),
                {x,y,z},radius,garden_materials[(row+2*column)%4]);
        }
        for(int i=0;i<5;++i){
            const double x=-3.10+1.55*i;
            add_rect(scene,"garden_flag_"+std::to_string(i),{x,1.18,-8.95+.17*(i%2)},
                {.62,0,.10*(i%2?1:-1)},{0,1.45,0},{0,0,1},garden_materials[(i+1)%4]);
        }
        add_rect(scene,"garden_canopy",{-2.65,4.55,-8.75},{5.30,0,0},{0,0,1.05},{0,-1,0},white,false);
    }else throw std::runtime_error("unknown scene mode: "+mode);
    build_scene_bvh(scene);
    return scene;
}

struct Ray{Vec3 origin{},direction{};};
struct Hit{bool valid=false;int primitive=-1;double t=0;Vec3 position{},normal{},geometric_normal{};bool front=true;};
struct TraceStats{
    std::uint64_t primary=0,secondary=0,shadow=0,mirror=0,metal=0,dielectric=0,specular_caustic=0;
    std::uint64_t prism_bottom_events=0,prism_top_events=0;
    std::uint64_t direct_atlas_gathers=0,direct_exact_calls=0,specular_area_calls=0,specular_memo_hits=0;
    std::uint64_t surface_radiance_memo_hits=0,surface_radiance_memo_stores=0;
    std::uint64_t primary_specular_area_calls=0,secondary_specular_area_calls=0;
    std::uint64_t camera_specular_gathers=0;
    std::uint64_t secondary_specular_weight_lt_1e4=0,secondary_specular_weight_lt_1e3=0;
    std::uint64_t secondary_specular_weight_lt_1e2=0;
    std::uint64_t emitter_rows=0,emitter_intervals=0,emitter_quadrature_samples=0;
    std::uint64_t feedback_loops=0,feedback_returns=0,sealed_feedback_tails=0,maximum_optical_packets=0;
    double sealed_residual_weight=0;
};

Hit intersect_primitive(const Primitive& p,int id,const Ray& ray,double t_max=std::numeric_limits<double>::infinity()){
    Hit h;h.primitive=id;double t=0;Vec3 gn{};
    if(p.shape==Shape::Rectangle){const double den=dot(p.normal,ray.direction);if(std::abs(den)<1e-10)return h;
        t=dot(p.origin-ray.origin,p.normal)/den;if(t<=1e-5||t>=t_max)return h;const Vec3 q=ray.origin+ray.direction*t-p.origin;
        const double qu=dot(q,p.u),qv=dot(q,p.v);if(std::abs(p.gram_det)<1e-16)return h;
        const double a=(qu*p.gram_vv-qv*p.gram_uv)/p.gram_det,b=(qv*p.gram_uu-qu*p.gram_uv)/p.gram_det;
        if(a<0||a>1||b<0||b>1)return h;gn=p.normal;
    }else if(p.shape==Shape::Sphere){const Vec3 rel=ray.origin-p.center;const double b=dot(ray.direction,rel);
        const double c=dot(rel,rel)-p.radius2,disc=b*b-c;if(disc<0)return h;const double r=std::sqrt(disc);
        t=-b-r;if(t<=1e-5)t=-b+r;if(t<=1e-5||t>=t_max)return h;gn=unit(ray.origin+ray.direction*t-p.center);
    }else{const Vec3 q=cross(ray.direction,p.edge2);const double det=dot(p.edge1,q);
        if(std::abs(det)<1e-10)return h;const double inv=1/det;const Vec3 s=ray.origin-p.a;const double u=dot(s,q)*inv;
        if(u<0||u>1)return h;const Vec3 side=cross(s,p.edge1);const double v=dot(ray.direction,side)*inv;
        if(v<0||u+v>1)return h;t=dot(p.edge2,side)*inv;if(t<=1e-5||t>=t_max)return h;gn=p.normal;}
    h.valid=true;h.t=t;h.position=ray.origin+ray.direction*t;h.geometric_normal=gn;h.front=dot(ray.direction,gn)<0;h.normal=h.front?gn:-gn;return h;
}

bool intersect_bounds(const Bounds3& bounds,const Ray& ray,double t_limit,double& near_distance){double near=0,far=t_limit;
    for(int axis=0;axis<3;++axis){const double origin=coordinate(ray.origin,axis),direction=coordinate(ray.direction,axis);
        const double lower=coordinate(bounds.lower,axis),upper=coordinate(bounds.upper,axis);
        if(std::abs(direction)<1e-15){if(origin<lower||origin>upper)return false;continue;}
        double a=(lower-origin)/direction,b=(upper-origin)/direction;if(a>b)std::swap(a,b);
        near=std::max(near,a);far=std::min(far,b);if(near>far)return false;}
    near_distance=near;return far>1e-5;}

Hit first_hit(const Scene& scene,const Ray& ray,double t_max=std::numeric_limits<double>::infinity(),int ignore=-1){
    Hit best;best.t=t_max;
    auto consider=[&](int primitive){if(primitive==ignore)return;const double epsilon=1e-11*std::max(1.0,best.t);
        const Hit hit=intersect_primitive(scene.primitives[primitive],primitive,ray,best.t+epsilon);if(!hit.valid)return;
        if(!best.valid||hit.t<best.t-epsilon||(std::abs(hit.t-best.t)<=epsilon&&primitive<best.primitive))best=hit;};
    if(!scene.use_bvh||scene.bvh_nodes.empty()){for(int i=0;i<static_cast<int>(scene.primitives.size());++i)consider(i);return best;}
    struct StackEntry{int node=-1;double near=0;};std::array<StackEntry,128> stack{};int size=0;double root_near=0;
    if(!intersect_bounds(scene.bvh_nodes.front().bounds,ray,best.t,root_near))return best;stack[size++]={0,root_near};
    while(size){const StackEntry entry=stack[--size];if(entry.near>best.t)continue;const BvhNode& node=scene.bvh_nodes[entry.node];
        if(node.count){for(int i=0;i<node.count;++i)consider(scene.bvh_primitives[node.begin+i]);continue;}
        double left_near=0,right_near=0;const bool left=intersect_bounds(scene.bvh_nodes[node.left].bounds,ray,best.t,left_near);
        const bool right=intersect_bounds(scene.bvh_nodes[node.right].bounds,ray,best.t,right_near);
        if(left&&right){if(left_near<right_near){stack[size++]={node.right,right_near};stack[size++]={node.left,left_near};}
            else{stack[size++]={node.left,left_near};stack[size++]={node.right,right_near};}}
        else if(left)stack[size++]={node.left,left_near};else if(right)stack[size++]={node.right,right_near};
    }
    return best;
}

#if defined(__clang__) || defined(__GNUC__)
#define PHOTONIC_RESTRICT __restrict__
#else
#define PHOTONIC_RESTRICT
#endif

// Coherent camera rays share an origin.  Traverse one shape-specialized
// primitive across the whole scanline so direction and nearest-hit arrays are
// contiguous and the compiler can SIMD the independent pixels. Secondary and
// incoherent rays retain first_hit(). Once the packet has selected a winner,
// assemble the Hit from its saved distance instead of intersecting it twice.
void first_hit_camera_packet(const Scene& scene,Vec3 ray_origin,const Ray* PHOTONIC_RESTRICT rays,
    const double* PHOTONIC_RESTRICT dx,const double* PHOTONIC_RESTRICT dy,
    const double* PHOTONIC_RESTRICT dz,Hit* PHOTONIC_RESTRICT hits,
    double* PHOTONIC_RESTRICT best_t,int* PHOTONIC_RESTRICT best_id,int count){
    if(scene.use_bvh||scene.bvh_nodes.empty()){for(int i=0;i<count;++i)hits[i]=first_hit(scene,rays[i]);return;}
    std::fill_n(best_t,count,std::numeric_limits<double>::infinity());std::fill_n(best_id,count,-1);
    for(int id=0;id<static_cast<int>(scene.primitives.size());++id){const Primitive& p=scene.primitives[id];
        if(p.shape==Shape::Rectangle){const Vec3 rel=ray_origin-p.origin;
#pragma clang loop vectorize(enable) interleave(enable)
            for(int i=0;i<count;++i){const double den=p.normal.x*dx[i]+p.normal.y*dy[i]+p.normal.z*dz[i];
                if(std::abs(den)<1e-10)continue;const double t=-(rel.x*p.normal.x+rel.y*p.normal.y+rel.z*p.normal.z)/den;
                if(t<=1e-5)continue;const double qx=rel.x+dx[i]*t,qy=rel.y+dy[i]*t,qz=rel.z+dz[i]*t;
                const double qu=qx*p.u.x+qy*p.u.y+qz*p.u.z,qv=qx*p.v.x+qy*p.v.y+qz*p.v.z;
                const double a=(qu*p.gram_vv-qv*p.gram_uv)/p.gram_det;
                const double b=(qv*p.gram_uu-qu*p.gram_uv)/p.gram_det;if(a<0||a>1||b<0||b>1)continue;
                const double epsilon=1e-11*std::max(1.0,best_t[i]);if(best_id[i]<0||t<best_t[i]-epsilon||
                    (std::abs(t-best_t[i])<=epsilon&&id<best_id[i])){best_t[i]=t;best_id[i]=id;}}
        }else if(p.shape==Shape::Sphere){const Vec3 rel=ray_origin-p.center;const double c=dot(rel,rel)-p.radius2;
#pragma clang loop vectorize(enable) interleave(enable)
            for(int i=0;i<count;++i){const double b=dx[i]*rel.x+dy[i]*rel.y+dz[i]*rel.z,disc=b*b-c;
                if(disc<0)continue;const double root=std::sqrt(disc);double t=-b-root;if(t<=1e-5)t=-b+root;if(t<=1e-5)continue;
                const double epsilon=1e-11*std::max(1.0,best_t[i]);if(best_id[i]<0||t<best_t[i]-epsilon||
                    (std::abs(t-best_t[i])<=epsilon&&id<best_id[i])){best_t[i]=t;best_id[i]=id;}}
        }else{const Vec3 s=ray_origin-p.a,side=cross(s,p.edge1);const double numerator=dot(p.edge2,side);
#pragma clang loop vectorize(enable) interleave(enable)
            for(int i=0;i<count;++i){const double qx=dy[i]*p.edge2.z-dz[i]*p.edge2.y;
                const double qy=dz[i]*p.edge2.x-dx[i]*p.edge2.z,qz=dx[i]*p.edge2.y-dy[i]*p.edge2.x;
                const double determinant=p.edge1.x*qx+p.edge1.y*qy+p.edge1.z*qz;if(std::abs(determinant)<1e-10)continue;
                const double inverse=1/determinant,u=(s.x*qx+s.y*qy+s.z*qz)*inverse;
                if(u<0||u>1)continue;const double v=(dx[i]*side.x+dy[i]*side.y+dz[i]*side.z)*inverse;
                if(v<0||u+v>1)continue;const double t=numerator*inverse;if(t<=1e-5)continue;
                const double epsilon=1e-11*std::max(1.0,best_t[i]);if(best_id[i]<0||t<best_t[i]-epsilon||
                    (std::abs(t-best_t[i])<=epsilon&&id<best_id[i])){best_t[i]=t;best_id[i]=id;}}}}
    for(int i=0;i<count;++i){const int id=best_id[i];if(id<0){hits[i]=Hit{};continue;}
        const Primitive& p=scene.primitives[id];Hit hit;hit.valid=true;hit.primitive=id;hit.t=best_t[i];
        hit.position=ray_origin+rays[i].direction*hit.t;
        hit.geometric_normal=p.shape==Shape::Sphere?unit(hit.position-p.center):p.normal;
        hit.front=dot(rays[i].direction,hit.geometric_normal)<0;
        hit.normal=hit.front?hit.geometric_normal:-hit.geometric_normal;hits[i]=hit;}
}

#undef PHOTONIC_RESTRICT

bool bounds_overlap(const Bounds3& a,const Bounds3& b){return a.lower.x<=b.upper.x&&a.upper.x>=b.lower.x&&
    a.lower.y<=b.upper.y&&a.upper.y>=b.lower.y&&a.lower.z<=b.upper.z&&a.upper.z>=b.lower.z;}

void query_scene_bounds(const Scene& scene,const Bounds3& query,std::vector<int>& candidates){candidates.clear();
    if(!scene.use_bvh||scene.bvh_nodes.empty()){candidates.resize(scene.primitives.size());
        for(int i=0;i<static_cast<int>(scene.primitives.size());++i)candidates[i]=i;return;}
    std::array<int,128> stack{};int size=0;stack[size++]=0;while(size){const BvhNode& node=scene.bvh_nodes[stack[--size]];
        if(!bounds_overlap(node.bounds,query))continue;if(node.count){for(int i=0;i<node.count;++i)
                candidates.push_back(scene.bvh_primitives[node.begin+i]);}
        else{stack[size++]=node.left;stack[size++]=node.right;}}
}

struct SurfaceSample{Vec3 position{},normal{};double weight=0;};
std::vector<SurfaceSample> surface_samples(const Primitive& p){std::vector<SurfaceSample> out;
    if(p.shape==Shape::Rectangle){constexpr int n=3;for(int y=0;y<n;++y)for(int x=0;x<n;++x)
        out.push_back({p.origin+p.u*((x+.5)/n)+p.v*((y+.5)/n),p.normal,p.area/(n*n)});
    }else if(p.shape==Shape::Sphere){constexpr int n=24;constexpr double golden=2.3999632297286533;
        for(int i=0;i<n;++i){const double y=1-2*(i+.5)/n,r=std::sqrt(std::max(0.0,1-y*y)),a=golden*i;
            const Vec3 normal{r*std::cos(a),y,r*std::sin(a)};out.push_back({p.center+normal*p.radius,normal,p.area/n});}}
    else{constexpr std::array<std::array<double,3>,7> bary{{{{1./3,1./3,1./3}},{{.72,.14,.14}},{{.14,.72,.14}},
            {{.14,.14,.72}},{{.5,.5,0}},{{0,.5,.5}},{{.5,0,.5}}}};
        for(const auto& w:bary)out.push_back({p.a*w[0]+p.b*w[1]+p.c*w[2],p.normal,p.area/bary.size()});}
    return out;
}

double schlick(double cosine,double eta_i,double eta_t){const double r=(eta_i-eta_t)/(eta_i+eta_t),r2=r*r;
    return r2+(1-r2)*std::pow(1-std::clamp(cosine,0.0,1.0),5);}

double spectral_ior(const Material& material,double coordinate){
    // Equal-width RGB response bands span a continuous dispersion coordinate.
    // End bands use the adjacent measured slope rather than clamping to a
    // single red or blue wavelength.
    if(coordinate<=1)return material.ior_rgb[0]+coordinate*(material.ior_rgb[1]-material.ior_rgb[0]);
    return material.ior_rgb[1]+(coordinate-1)*(material.ior_rgb[2]-material.ior_rgb[1]);
}

RGB segment_transmittance(const Scene& scene,Vec3 a,Vec3 b,int ignore_a=-1,int ignore_b=-1){RGB throughput{1,1,1};
    Vec3 delta=b-a;double remaining=norm(delta);if(remaining<1e-8)return throughput;const Vec3 d=delta/remaining;
    Ray ray{a+d*2e-4,d};remaining-=2e-4;int previous=ignore_a;
    for(int step=0;step<16&&remaining>2e-4;++step){const Hit h=first_hit(scene,ray,remaining+1e-4,previous);if(!h.valid)break;
        if(h.primitive==ignore_b)break;const Material& m=scene.materials[scene.primitives[h.primitive].material];
        if(m.kind!=MaterialKind::Dielectric)return {};
        const double path=m.thin?.08:.16;for(int c=0;c<3;++c)
            throughput[c]*=m.base[c]*(1-schlick(std::abs(dot(d,h.normal)),1,m.ior_rgb[c]))*std::exp(-m.absorption[c]*path);
        remaining-=h.t+3e-4;ray.origin=h.position+d*3e-4;previous=h.primitive;}
    return throughput;
}

struct SegmentProgram{std::array<int,16> primitive{};int count=0;};

SegmentProgram discover_segment_program(const Scene& scene,Vec3 a,Vec3 b,int ignore_a=-1,int ignore_b=-1){
    SegmentProgram program;Vec3 delta=b-a;double remaining=norm(delta);if(remaining<1e-8)return program;
    const Vec3 d=delta/remaining;Ray ray{a+d*2e-4,d};remaining-=2e-4;int previous=ignore_a;
    for(int step=0;step<16&&remaining>2e-4;++step){const Hit hit=first_hit(scene,ray,remaining+1e-4,previous);
        if(!hit.valid||hit.primitive==ignore_b)break;program.primitive[program.count++]=hit.primitive;
        remaining-=hit.t+3e-4;ray.origin=hit.position+d*3e-4;previous=hit.primitive;}
    return program;
}

RGB evaluate_segment_program(const Scene& scene,const SegmentProgram& program,Vec3 a,Vec3 b,
                             int ignore_a=-1,int ignore_b=-1){
    RGB throughput{1,1,1};const Vec3 delta=b-a;const double distance=norm(delta);if(distance<1e-8)return throughput;
    const Vec3 d=delta/distance;const Ray ray{a+d*2e-4,d};
    for(int i=0;i<program.count;++i){const int primitive=program.primitive[i];if(primitive==ignore_a||primitive==ignore_b)continue;
        const Hit hit=intersect_primitive(scene.primitives[primitive],primitive,ray,distance+1e-4);
        if(!hit.valid)return segment_transmittance(scene,a,b,ignore_a,ignore_b);const Material& material=
            scene.materials[scene.primitives[hit.primitive].material];if(material.kind!=MaterialKind::Dielectric)return {};
        const double path=material.thin?.08:.16;
        for(int channel=0;channel<3;++channel)throughput[channel]*=material.base[channel]*
            (1-schlick(std::abs(dot(d,hit.normal)),1,material.ior_rgb[channel]))*
            std::exp(-material.absorption[channel]*path);}
    return throughput;
}

struct ProjectedPolygon{std::array<Vec2,4> vertex{};int count=0,primitive=-1;};
struct EmitterPartition{std::vector<ProjectedPolygon> polygons;std::vector<int> spheres;};

bool emitter_coordinates(const Primitive& light,Vec3 point,Vec2& uv){
    const Vec3 q=point-light.origin;const double uu=dot(light.u,light.u),cross_uv=dot(light.u,light.v),vv=dot(light.v,light.v);
    const double det=uu*vv-cross_uv*cross_uv;if(std::abs(det)<1e-18)return false;
    const double qu=dot(q,light.u),qv=dot(q,light.v);
    uv={(qu*vv-qv*cross_uv)/det,(qv*uu-qu*cross_uv)/det};return true;
}

bool project_vertex_to_emitter(const Primitive& light,Vec3 target,Vec3 vertex,Vec2& uv){
    const Vec3 direction=vertex-target;const double denominator=dot(light.normal,direction);
    if(std::abs(denominator)<1e-12)return false;const double t=dot(light.normal,light.origin-target)/denominator;
    if(t<=0)return false;return emitter_coordinates(light,target+direction*t,uv);
}

double primitive_bound_radius(const Primitive& primitive){
    if(primitive.shape==Shape::Sphere)return primitive.radius;
    if(primitive.shape==Shape::Rectangle)return .5*std::sqrt(norm2(primitive.u)+norm2(primitive.v));
    return std::max({norm(primitive.a-primitive.center),norm(primitive.b-primitive.center),
        norm(primitive.c-primitive.center)});
}

bool blocker_may_overlap_emitter(const Primitive& light,const Primitive& blocker,Vec3 target){
    // A conservative angular-disk rejection.  The disks enclose both the
    // rectangular emitter and every analytic blocker, so rejecting disjoint
    // disks cannot discard a real source interval.
    const Vec3 source_axis=light.center-target;const double source_distance=norm(source_axis);
    if(source_distance<=1e-12)return true;const Vec3 source_direction=source_axis/source_distance;
    const double source_radius=primitive_bound_radius(light),blocker_radius=primitive_bound_radius(blocker);
    const Vec3 blocker_axis=blocker.center-target;const double blocker_distance=norm(blocker_axis);
    if(blocker_distance<=blocker_radius*(1+1e-12))return true;
    if(dot(blocker_axis,source_direction)+blocker_radius<=0)return false;
    if(blocker_distance-blocker_radius>=source_distance+source_radius)return false;
    const double source_angle=source_distance<=source_radius?pi*.5:
        std::asin(std::clamp(source_radius/source_distance,0.0,1.0));
    const double blocker_angle=std::asin(std::clamp(blocker_radius/blocker_distance,0.0,1.0));
    const double separation=std::acos(std::clamp(dot(source_direction,blocker_axis/blocker_distance),-1.0,1.0));
    return separation<=source_angle+blocker_angle;
}

EmitterPartition build_emitter_partition(const Scene& scene,const Primitive& light,Vec3 target,int receiver,int emitter){
    EmitterPartition partition;Bounds3 source_pyramid;expand(source_pyramid,target);expand(source_pyramid,light.origin);
    expand(source_pyramid,light.origin+light.u);expand(source_pyramid,light.origin+light.v);
    expand(source_pyramid,light.origin+light.u+light.v);std::vector<int> candidates;
    query_scene_bounds(scene,source_pyramid,candidates);
    for(int id:candidates){if(id==receiver||id==emitter)continue;
        const Primitive& blocker=scene.primitives[id];if(!blocker_may_overlap_emitter(light,blocker,target))continue;
        if(blocker.shape==Shape::Sphere){partition.spheres.push_back(id);continue;}
        ProjectedPolygon polygon;polygon.count=blocker.shape==Shape::Rectangle?4:3;polygon.primitive=id;
        const std::array<Vec3,4> vertices=blocker.shape==Shape::Rectangle?
            std::array<Vec3,4>{{blocker.origin,blocker.origin+blocker.u,blocker.origin+blocker.u+blocker.v,blocker.origin+blocker.v}}:
            std::array<Vec3,4>{{blocker.a,blocker.b,blocker.c,blocker.c}};
        bool valid=true;double u0=std::numeric_limits<double>::infinity(),u1=-u0;
        double v0=std::numeric_limits<double>::infinity(),v1=-v0;
        for(int i=0;i<polygon.count;++i){valid=valid&&project_vertex_to_emitter(light,target,vertices[i],polygon.vertex[i]);
            if(!valid)break;u0=std::min(u0,polygon.vertex[i].x);u1=std::max(u1,polygon.vertex[i].x);
            v0=std::min(v0,polygon.vertex[i].y);v1=std::max(v1,polygon.vertex[i].y);}
        if(valid&&u1>=0&&u0<=1&&v1>=0&&v0<=1)partition.polygons.push_back(polygon);
    }
    return partition;
}

void append_sphere_conic_roots(const Primitive& light,const Primitive& sphere,Vec3 target,double v,std::vector<double>& cuts){
    const Vec3 base=light.origin+light.v*v-target,w=sphere.center-target,U=light.u;
    const double k=dot(w,w)-sphere.radius*sphere.radius,wu=dot(w,U),wb=dot(w,base);
    const double A=wu*wu-k*dot(U,U),B=2*(wb*wu-k*dot(base,U)),C=wb*wb-k*dot(base,base);
    auto append=[&](double u){if(u>1e-10&&u<1-1e-10&&std::isfinite(u))cuts.push_back(u);};
    if(std::abs(A)<1e-14){if(std::abs(B)>1e-14)append(-C/B);return;}
    const double discriminant=B*B-4*A*C;if(discriminant<0)return;const double root=std::sqrt(std::max(0.0,discriminant));
    const double q=-.5*(B+std::copysign(root,B));if(std::abs(q)>1e-18){append(q/A);append(C/q);}else append(-B/(2*A));
}

void append_sphere_vertical_extrema(const Primitive& light,const Primitive& sphere,Vec3 target,std::vector<double>& cuts){
    const Vec3 P=light.origin-target,U=light.u,V=light.v,w=sphere.center-target;
    const double k=dot(w,w)-sphere.radius*sphere.radius;
    const double wu=dot(w,U),wv=dot(w,V),wp=dot(w,P);
    const double A=wu*wu-k*dot(U,U),B=2*(wu*wv-k*dot(U,V)),C=wv*wv-k*dot(V,V);
    const double D=2*(wp*wu-k*dot(P,U)),E=2*(wp*wv-k*dot(P,V)),F=wp*wp-k*dot(P,P);
    const double qa=B*B-4*A*C,qb=2*B*D-4*A*E,qc=D*D-4*A*F;
    auto append=[&](double v){if(v>1e-10&&v<1-1e-10&&std::isfinite(v))cuts.push_back(v);};
    if(std::abs(qa)<1e-14){if(std::abs(qb)>1e-14)append(-qc/qb);return;}
    const double discriminant=qb*qb-4*qa*qc;if(discriminant<0)return;
    const double root=std::sqrt(std::max(0.0,discriminant)),q=-.5*(qb+std::copysign(root,qb));
    if(std::abs(q)>1e-18){append(q/qa);append(qc/q);}else append(-qb/(2*qa));
}

std::vector<double> emitter_vertical_cuts(const Scene& scene,const Primitive& light,const EmitterPartition& partition,
                                          Vec3 target){std::vector<double> cuts{0,1};
    cuts.reserve(2+4*partition.polygons.size()+2*partition.spheres.size());
    for(const ProjectedPolygon& polygon:partition.polygons)for(int i=0;i<polygon.count;++i){
        const double v=polygon.vertex[i].y;if(v>1e-10&&v<1-1e-10&&std::isfinite(v))cuts.push_back(v);}
    for(int id:partition.spheres)append_sphere_vertical_extrema(light,scene.primitives[id],target,cuts);
    std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end(),
        [](double a,double b){return std::abs(a-b)<1e-10;}),cuts.end());return cuts;
}

std::vector<double> emitter_row_cuts(const Scene& scene,const Primitive& light,const EmitterPartition& partition,
                                     Vec3 target,double v){std::vector<double> cuts{0,1};
    // Silhouette events are linear in the row population. Depth-order events
    // are appended only for actually overlapping spans below; never reserve a
    // quadratic array for pairs that may not exist.
    cuts.reserve(2+3*(partition.polygons.size()+partition.spheres.size()));
    for(int id:partition.spheres)append_sphere_conic_roots(light,scene.primitives[id],target,v,cuts);
    struct RowSpan{std::size_t polygon=0;double lower=0,upper=0;};std::vector<RowSpan> spans;spans.reserve(partition.polygons.size());
    for(std::size_t polygon_index=0;polygon_index<partition.polygons.size();++polygon_index){
        const ProjectedPolygon& polygon=partition.polygons[polygon_index];std::array<double,4> intersection{};int count=0;
        for(int i=0;i<polygon.count;++i){const Vec2 a=polygon.vertex[i],b=polygon.vertex[(i+1)%polygon.count];
            if((a.y<=v&&v<b.y)||(b.y<=v&&v<a.y)){const double u=a.x+(v-a.y)*(b.x-a.x)/(b.y-a.y);
                if(std::isfinite(u)){intersection[count++]=u;if(u>1e-10&&u<1-1e-10)cuts.push_back(u);}}}
        if(count>=2){const auto limits=std::minmax_element(intersection.begin(),intersection.begin()+count);
            if(*limits.second>=0&&*limits.first<=1)spans.push_back({polygon_index,*limits.first,*limits.second});}}
    // Two overlapping planar blockers may exchange front-to-back order without
    // either silhouette ending.  Their ray depths are ai/(bi+ci*u), so the
    // equality has one analytic root along this source row.  Cutting there
    // makes the ordered blocker program invariant inside every interval.
    const Vec3 row_base=light.origin+light.v*v-target;
    std::sort(spans.begin(),spans.end(),[](const RowSpan& a,const RowSpan& b){return a.lower<b.lower;});
    std::vector<RowSpan> active;active.reserve(spans.size());for(const RowSpan& current:spans){
        active.erase(std::remove_if(active.begin(),active.end(),[&](const RowSpan& span){return span.upper<current.lower;}),active.end());
        for(const RowSpan& previous:active){const double overlap_lower=std::max({0.0,current.lower,previous.lower});
        const double overlap_upper=std::min({1.0,current.upper,previous.upper});if(overlap_upper-overlap_lower<=1e-12)continue;
        const Primitive& first=scene.primitives[partition.polygons[previous.polygon].primitive];
        const Primitive& second=scene.primitives[partition.polygons[current.polygon].primitive];
        const Vec3 first_point=first.shape==Shape::Rectangle?first.origin:first.a;
        const Vec3 second_point=second.shape==Shape::Rectangle?second.origin:second.a;
        const double ai=dot(first.normal,first_point-target),aj=dot(second.normal,second_point-target);
        const double bi=dot(first.normal,row_base),bj=dot(second.normal,row_base);
        const double ci=dot(first.normal,light.u),cj=dot(second.normal,light.u);
        const double denominator=ai*cj-aj*ci;if(std::abs(denominator)<1e-14)continue;
        const double u=(aj*bi-ai*bj)/denominator;if(u<=overlap_lower+1e-10||u>=overlap_upper-1e-10||!std::isfinite(u))continue;
        const double ray_denominator=bi+ci*u;if(std::abs(ray_denominator)<1e-14)continue;
        const double depth=ai/ray_denominator;if(depth>1e-10&&depth<1-1e-10)cuts.push_back(u);}
        active.push_back(current);}
    std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end(),
        [](double a,double b){return std::abs(a-b)<1e-10;}),cuts.end());return cuts;
}

static constexpr std::array<double,2> emitter_gauss2_x{{-.5773502691896258,.5773502691896258}};
static constexpr std::array<double,2> emitter_gauss2_w{{1,1}};
static constexpr std::array<double,3> emitter_gauss3_x{{-.7745966692414834,0,.7745966692414834}};
static constexpr std::array<double,3> emitter_gauss3_w{{5./9,8./9,5./9}};
static constexpr std::array<double,4> emitter_gauss4_x{{
    -.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526}};
static constexpr std::array<double,4> emitter_gauss4_w{{
    .3478548451374539,.6521451548625461,.6521451548625461,.3478548451374539}};

template<class Kernel>
RGB integrate_rectangular_emitter(const Scene& scene,const AreaLight& area,Vec3 point,Vec3 normal,int receiver,
                                  Kernel&& kernel,TraceStats* stats=nullptr,int quadrature_order=2){RGB result{};
    const double* gauss_x=emitter_gauss2_x.data();const double* gauss_w=emitter_gauss2_w.data();
    if(quadrature_order>=4){quadrature_order=4;gauss_x=emitter_gauss4_x.data();gauss_w=emitter_gauss4_w.data();}
    else if(quadrature_order==3){gauss_x=emitter_gauss3_x.data();gauss_w=emitter_gauss3_w.data();}
    else quadrature_order=2;
    const Primitive& light=scene.primitives[area.primitive];const EmitterPartition partition=
        build_emitter_partition(scene,light,point,receiver,area.primitive);
    const auto vertical_cuts=emitter_vertical_cuts(scene,light,partition,point);
    for(std::size_t band=1;band<vertical_cuts.size();++band){const double vlo=vertical_cuts[band-1],vhi=vertical_cuts[band];
        if(vhi-vlo<=1e-12)continue;
        for(int j=0;j<quadrature_order;++j){const double v=.5*((vhi-vlo)*gauss_x[j]+vhi+vlo);
            const auto cuts=emitter_row_cuts(scene,light,partition,point,v);if(stats)++stats->emitter_rows;
            for(std::size_t interval=1;interval<cuts.size();++interval){const double lo=cuts[interval-1],hi=cuts[interval];
                if(hi-lo<=1e-12)continue;if(stats)++stats->emitter_intervals;RGB row{};
                // Every blocker boundary is already a cut, hence the ordered
                // opacity/dielectric membership is constant inside this open
                // interval.  Discover it once at the interval centroid; the
                // four Gaussian nodes integrate only the smooth geometric/BRDF
                // kernel.  The retained program names the crossed dielectric
                // faces, so their angle-dependent Fresnel terms remain exact at
                // every ordinate without repeating a whole-scene ray walk.
                const Vec3 midpoint=light.origin+light.u*((lo+hi)*.5)+light.v*v;
                const SegmentProgram program=partition.polygons.empty()&&partition.spheres.empty()?SegmentProgram{}:
                    discover_segment_program(scene,point,midpoint,receiver,area.primitive);
                if(stats)++stats->shadow;
                for(int i=0;i<quadrature_order;++i){const double u=.5*((hi-lo)*gauss_x[i]+hi+lo);
                const Vec3 q=light.origin+light.u*u+light.v*v,delta=q-point;const double r2=norm2(delta);if(r2<1e-12)continue;
                const Vec3 d=delta/std::sqrt(r2);const double cr=std::max(dot(normal,d),0.0),cl=std::max(dot(light.normal,-d),0.0);
                if(cr<=0||cl<=0)continue;if(stats)++stats->emitter_quadrature_samples;
                const RGB trans=evaluate_segment_program(scene,program,point,q,receiver,area.primitive);
                row+=kernel(q,d,r2,cr,cl,trans)*gauss_w[i];}
                result+=row*(.25*gauss_w[j]*(vhi-vlo)*(hi-lo)*light.area);}}}
    return result;
}

RGB area_irradiance(const Scene& scene,Vec3 point,Vec3 normal,int receiver,TraceStats* stats=nullptr){RGB result{};
    for(const AreaLight& area:scene.area_lights)result+=integrate_rectangular_emitter(scene,area,point,normal,receiver,
        [&](Vec3,Vec3,double r2,double cr,double cl,const RGB& trans){return multiply(area.radiance,trans)*(cr*cl/r2);},stats);
    return result;
}

struct CausticDeposit{int primitive=-1;Vec3 position{};RGB power{};double radius=.12;};
struct OrientedBeamSheet{
    int primitive=-1,channel=0;std::uint64_t path=0,sample_count=0;Vec3 position{},normal{},tangent_u{},tangent_v{},direction{};
    RGB power{};std::array<double,16> phase_covariance{};std::array<double,4> inverse_position_covariance{};
    double position_determinant=0,anisotropy=1,cross_coupling=0;
};
struct BeamField{std::vector<CausticDeposit> deposits;std::vector<std::vector<int>> by_primitive;
    std::vector<OrientedBeamSheet> oriented_sheets;std::vector<std::vector<int>> oriented_by_primitive;bool oriented=false;
    std::vector<std::uint64_t> dielectric_by_primitive;
    std::uint64_t launched=0,dielectric_crossings=0,mirror_crossings=0;};

struct DepositAccumulator{Vec3 weighted{};RGB power{};double mass=0;};
struct BeamCompileConfig{double cell=.095,deposit_radius=.14;};
BeamField compile_beam_field(const Scene& scene,BeamCompileConfig config={}){BeamField field;field.by_primitive.resize(scene.primitives.size());
    field.dielectric_by_primitive.resize(scene.primitives.size());
    std::map<std::tuple<int,int,int,int>,DepositAccumulator> bins;const double cell=config.cell;
    for(const BeamBundle& beam:scene.beams){const int n=beam.fibres;int accepted=0;
        for(int y=0;y<n;++y)for(int x=0;x<n;++x){const double u=2*(x+.5)/n-1,v=2*(y+.5)/n-1;if(u*u+v*v>1)continue;++accepted;}
        for(int channel=0;channel<3;++channel)for(int y=0;y<n;++y)for(int x=0;x<n;++x){
            const double u=2*(x+.5)/n-1,v=2*(y+.5)/n-1;if(u*u+v*v>1)continue;++field.launched;
            const Vec3 aperture=beam.axis_u*(u*beam.radius)+beam.axis_v*(v*beam.radius);
            Vec3 d=unit(beam.direction+beam.axis_u*(u*beam.spread)+beam.axis_v*(v*beam.spread));
            Ray ray{beam.origin+aperture,d};double power=beam.power[channel]/accepted;int previous=-1;
            for(int depth=0;depth<12&&power>1e-7;++depth){const Hit h=first_hit(scene,ray,std::numeric_limits<double>::infinity(),previous);
                if(!h.valid)break;const Material& m=scene.materials[scene.primitives[h.primitive].material];
                if(m.kind==MaterialKind::Dielectric){++field.dielectric_crossings;++field.dielectric_by_primitive[h.primitive];
                    const double eta_i=h.front?1:m.ior_rgb[channel];
                    const double eta_t=h.front?m.ior_rgb[channel]:1;const double F=schlick(std::abs(dot(ray.direction,h.normal)),eta_i,eta_t);
                    power*=m.base[channel]*(1-F)*std::exp(-m.absorption[channel]*(m.thin?.08:.18));Vec3 next=ray.direction;
                    if(!m.thin&&!refract(ray.direction,h.normal,eta_i/eta_t,next))next=reflect(ray.direction,h.normal);
                    ray={h.position+next*3e-4,next};previous=h.primitive;continue;}
                if(m.kind==MaterialKind::Mirror){++field.mirror_crossings;d=reflect(ray.direction,h.normal);power*=m.base[channel];
                    ray={h.position+d*3e-4,d};previous=h.primitive;continue;}
                const int bx=static_cast<int>(std::floor(h.position.x/cell)),by=static_cast<int>(std::floor(h.position.y/cell));
                const int bz=static_cast<int>(std::floor(h.position.z/cell));auto& a=bins[{h.primitive,bx,by,bz}];
                a.weighted=a.weighted+h.position*power;a.power[channel]+=power;a.mass+=power;break;}}}
    for(const auto& [key,a]:bins){if(a.mass<=0)continue;CausticDeposit d;d.primitive=std::get<0>(key);d.position=a.weighted/a.mass;
        d.power=a.power;d.radius=config.deposit_radius;field.by_primitive[d.primitive].push_back(static_cast<int>(field.deposits.size()));field.deposits.push_back(d);}
    return field;
}

struct BeamEndpointSample{Vec3 position{},normal{},direction{};double u=0,v=0,power=0;};

std::uint64_t beam_path_push(std::uint64_t path,int primitive,int event){std::uint64_t value=
    static_cast<std::uint64_t>(primitive+1)*0x9e3779b97f4a7c15ULL+static_cast<std::uint64_t>(event+17);
    value^=value>>30;value*=0xbf58476d1ce4e5b9ULL;value^=value>>27;return path^(value+0x9e3779b97f4a7c15ULL+(path<<6)+(path>>2));}

bool march_beam_endpoint(const Scene& scene,const BeamBundle& beam,double u,double v,int channel,double initial_power,
    BeamField& field,BeamEndpointSample& sample,int& primitive,std::uint64_t& path){const Vec3 aperture=
        beam.axis_u*(u*beam.radius)+beam.axis_v*(v*beam.radius);Vec3 direction=unit(beam.direction+
        beam.axis_u*(u*beam.spread)+beam.axis_v*(v*beam.spread));Ray ray{beam.origin+aperture,direction};
    double power=initial_power;int previous=-1;path=1469598103934665603ULL;
    for(int depth=0;depth<12&&power>1e-7;++depth){const Hit hit=first_hit(scene,ray,std::numeric_limits<double>::infinity(),previous);
        if(!hit.valid)return false;const Material& material=scene.materials[scene.primitives[hit.primitive].material];
        if(material.kind==MaterialKind::Dielectric){++field.dielectric_crossings;++field.dielectric_by_primitive[hit.primitive];
            path=beam_path_push(path,hit.primitive,1);const double eta_i=hit.front?1:material.ior_rgb[channel];
            const double eta_t=hit.front?material.ior_rgb[channel]:1;const double fresnel=
                schlick(std::abs(dot(ray.direction,hit.normal)),eta_i,eta_t);power*=material.base[channel]*(1-fresnel)*
                std::exp(-material.absorption[channel]*(material.thin?.08:.18));Vec3 next=ray.direction;
            if(!material.thin&&!refract(ray.direction,hit.normal,eta_i/eta_t,next)){
                next=reflect(ray.direction,hit.normal);path=beam_path_push(path,hit.primitive,2);}
            ray={hit.position+next*3e-4,next};previous=hit.primitive;continue;}
        if(material.kind==MaterialKind::Mirror){++field.mirror_crossings;path=beam_path_push(path,hit.primitive,3);
            direction=reflect(ray.direction,hit.normal);power*=material.base[channel];ray={hit.position+direction*3e-4,direction};
            previous=hit.primitive;continue;}
        path=beam_path_push(path,hit.primitive,4);primitive=hit.primitive;sample={hit.position,hit.normal,ray.direction,u,v,power};return true;}
    return false;
}

Vec3 beam_tangent(Vec3 normal){return unit(cross(normal,std::abs(normal.y)<.9?Vec3{0,1,0}:Vec3{1,0,0}));}

BeamField compile_oriented_beam_field(const Scene& scene,double spatial_blur=.14){BeamField field;field.oriented=true;
    field.by_primitive.resize(scene.primitives.size());field.oriented_by_primitive.resize(scene.primitives.size());
    field.dielectric_by_primitive.resize(scene.primitives.size());
    using Group=std::tuple<int,int,std::uint64_t>;std::map<Group,std::vector<BeamEndpointSample>> groups;
    for(const BeamBundle& beam:scene.beams){const int count=beam.fibres;int accepted=0;
        for(int y=0;y<count;++y)for(int x=0;x<count;++x){const double u=2*(x+.5)/count-1,v=2*(y+.5)/count-1;
            if(u*u+v*v<=1)++accepted;}
        for(int channel=0;channel<3;++channel)for(int y=0;y<count;++y)for(int x=0;x<count;++x){const double u=
            2*(x+.5)/count-1,v=2*(y+.5)/count-1;if(u*u+v*v>1)continue;++field.launched;BeamEndpointSample sample;
            int primitive=-1;std::uint64_t path=0;if(march_beam_endpoint(scene,beam,u,v,channel,beam.power[channel]/accepted,
                    field,sample,primitive,path))groups[{channel,primitive,path}].push_back(sample);}}
    for(const auto& [group,samples]:groups){if(samples.empty())continue;OrientedBeamSheet sheet;sheet.channel=std::get<0>(group);
        sheet.primitive=std::get<1>(group);sheet.path=std::get<2>(group);sheet.sample_count=samples.size();double mass=0;
        for(const BeamEndpointSample& sample:samples){mass+=sample.power;sheet.position=sheet.position+sample.position*sample.power;
            sheet.normal=sheet.normal+sample.normal*sample.power;sheet.direction=sheet.direction+sample.direction*sample.power;}
        if(mass<=0)continue;sheet.position=sheet.position/mass;sheet.normal=unit(sheet.normal);sheet.direction=unit(sheet.direction);
        sheet.tangent_u=beam_tangent(sheet.normal);sheet.tangent_v=unit(cross(sheet.normal,sheet.tangent_u));sheet.power[sheet.channel]=mass;
        std::array<double,4> mean{};for(const BeamEndpointSample& sample:samples){const Vec3 offset=sample.position-sheet.position;
            const double denominator=std::max(std::abs(dot(sample.direction,sheet.normal)),1e-5);const std::array<double,4> value{{
                dot(offset,sheet.tangent_u),dot(offset,sheet.tangent_v),dot(sample.direction,sheet.tangent_u)/denominator,
                dot(sample.direction,sheet.tangent_v)/denominator}};for(int i=0;i<4;++i)mean[i]+=sample.power*value[i]/mass;}
        for(const BeamEndpointSample& sample:samples){const Vec3 offset=sample.position-sheet.position;
            const double denominator=std::max(std::abs(dot(sample.direction,sheet.normal)),1e-5);std::array<double,4> value{{
                dot(offset,sheet.tangent_u),dot(offset,sheet.tangent_v),dot(sample.direction,sheet.tangent_u)/denominator,
                dot(sample.direction,sheet.tangent_v)/denominator}};for(int i=0;i<4;++i)value[i]-=mean[i];
            for(int i=0;i<4;++i)for(int j=0;j<4;++j)sheet.phase_covariance[i*4+j]+=sample.power*value[i]*value[j]/mass;}
        sheet.phase_covariance[0]+=spatial_blur*spatial_blur;sheet.phase_covariance[5]+=spatial_blur*spatial_blur;
        const double a=sheet.phase_covariance[0],b=sheet.phase_covariance[1],d=sheet.phase_covariance[5];
        sheet.position_determinant=std::max(a*d-b*b,1e-18);sheet.inverse_position_covariance=
            {d/sheet.position_determinant,-b/sheet.position_determinant,-b/sheet.position_determinant,a/sheet.position_determinant};
        const double trace=a+d,disc=std::sqrt(std::max(0.0,(a-d)*(a-d)+4*b*b));const double major=.5*(trace+disc),minor=
            std::max(.5*(trace-disc),1e-18);sheet.anisotropy=std::sqrt(major/minor);double cross=0;
        for(int i=0;i<2;++i)for(int j=2;j<4;++j)cross+=sheet.phase_covariance[i*4+j]*sheet.phase_covariance[i*4+j];
        const double position_variance=sheet.phase_covariance[0]+sheet.phase_covariance[5];
        const double angular_variance=sheet.phase_covariance[10]+sheet.phase_covariance[15];
        sheet.cross_coupling=std::sqrt(cross/std::max(position_variance*angular_variance,1e-30));
        field.oriented_by_primitive[sheet.primitive].push_back(
            static_cast<int>(field.oriented_sheets.size()));field.oriented_sheets.push_back(sheet);}
    return field;
}

RGB beam_irradiance(const BeamField& field,int primitive,Vec3 point){RGB result{};if(primitive<0)return result;
    if(field.oriented){for(int index:field.oriented_by_primitive[primitive]){const OrientedBeamSheet& sheet=field.oriented_sheets[index];
            const Vec3 offset=point-sheet.position;const double x=dot(offset,sheet.tangent_u),y=dot(offset,sheet.tangent_v);
            const auto& inverse=sheet.inverse_position_covariance;const double q=.5*(inverse[0]*x*x+(inverse[1]+inverse[2])*x*y+
                inverse[3]*y*y);if(q>12)continue;result+=sheet.power*(std::exp(-q)/(2*pi*std::sqrt(sheet.position_determinant)));}
        return result;}
    for(int index:field.by_primitive[primitive]){const CausticDeposit& d=field.deposits[index];const double q=norm2(point-d.position)/(2*d.radius*d.radius);
        if(q>12)continue;result+=d.power*(std::exp(-q)/(2*pi*d.radius*d.radius));}return result;
}

std::uint8_t tone_byte(double linear);

struct SurfaceIrradianceAtlas{
    int primitive=-1,width=0,height=0,refine_side=5;bool sphere=false;std::vector<RGB> value,refined_value;
    std::vector<int> refined_offset;
};

bool rectangle_coordinates(const Primitive& primitive,Vec3 point,double& u,double& v){
    const Vec3 q=point-primitive.origin;const double uu=dot(primitive.u,primitive.u),uv=dot(primitive.u,primitive.v);
    const double vv=dot(primitive.v,primitive.v),qu=dot(q,primitive.u),qv=dot(q,primitive.v),det=uu*vv-uv*uv;
    if(std::abs(det)<1e-18)return false;u=(qu*vv-qv*uv)/det;v=(qv*uu-qu*uv)/det;return true;
}

RGB sample_surface_atlas(const SurfaceIrradianceAtlas& atlas,const Primitive& primitive,Vec3 point){
    double u=0,v=0;if(atlas.sphere){const Vec3 normal=unit(point-primitive.center);
        u=(std::atan2(normal.z,normal.x)+pi)/(2*pi);v=(1-normal.y)*.5;
    }else if(!rectangle_coordinates(primitive,point,u,v))return {};
    const double x=std::clamp(u,0.0,1.0)*(atlas.width-1),y=std::clamp(v,0.0,1.0)*(atlas.height-1);
    const int x0=std::clamp(static_cast<int>(std::floor(x)),0,atlas.width-1),x1=std::min(x0+1,atlas.width-1);
    const int y0=std::clamp(static_cast<int>(std::floor(y)),0,atlas.height-1),y1=std::min(y0+1,atlas.height-1);
    double tx=x-x0,ty=y-y0;int local_width=atlas.width,offset=0;const std::vector<RGB>* values=&atlas.value;
    int sx0=x0,sx1=x1,sy0=y0,sy1=y1;
    if(x0+1<atlas.width&&y0+1<atlas.height&&!atlas.refined_offset.empty()){const int refined=
            atlas.refined_offset[static_cast<std::size_t>(y0)*(atlas.width-1)+x0];
        if(refined>=0){const double rx=tx*(atlas.refine_side-1),ry=ty*(atlas.refine_side-1);
            sx0=std::min(static_cast<int>(std::floor(rx)),atlas.refine_side-2);sx1=sx0+1;
            sy0=std::min(static_cast<int>(std::floor(ry)),atlas.refine_side-2);sy1=sy0+1;
            tx=rx-sx0;ty=ry-sy0;local_width=atlas.refine_side;offset=refined;values=&atlas.refined_value;}}
    auto at=[&](int sx,int sy)->const RGB&{return (*values)[offset+static_cast<std::size_t>(sy)*local_width+sx];};RGB result{};
    for(int channel=0;channel<3;++channel){const double a=at(sx0,sy0)[channel]*(1-tx)+at(sx1,sy0)[channel]*tx;
        const double b=at(sx0,sy1)[channel]*(1-tx)+at(sx1,sy1)[channel]*tx;result[channel]=a*(1-ty)+b*ty;}
    return result;
}

struct TransportCoupling{int source=-1;double weight=0;};
struct TransportRecipient{int receiver=-1;double weight=0;};
struct TransportField{std::vector<int> node_primitives,index_by_primitive;std::vector<RGB> direct,bounce,radiance;
    std::vector<TransportCoupling> coupling;std::vector<std::size_t> incoming_offset;
    std::vector<TransportRecipient> outgoing;std::vector<std::size_t> outgoing_offset;
    std::vector<SurfaceIrradianceAtlas> direct_atlas;std::vector<int> direct_atlas_by_primitive;
    std::uint64_t direct_atlas_samples=0,nonzeros=0,propagated_edges=0;int iterations=0;double final_residual=0;};

TransportField compile_transport_field(const Scene& scene,const BeamField& beams){TransportField field;
    field.index_by_primitive.assign(scene.primitives.size(),-1);
    for(int p=0;p<static_cast<int>(scene.primitives.size());++p){const Primitive& primitive=scene.primitives[p];
        const Material& m=scene.materials[primitive.material];if(!primitive.transport||m.diffuse<=0||m.kind==MaterialKind::Emissive||
            m.kind==MaterialKind::Mirror||m.kind==MaterialKind::Dielectric)continue;
        field.index_by_primitive[p]=static_cast<int>(field.node_primitives.size());field.node_primitives.push_back(p);}
    const int count=static_cast<int>(field.node_primitives.size());field.direct.resize(count);field.bounce.resize(count);
    field.radiance.resize(count);field.incoming_offset.resize(static_cast<std::size_t>(count)+1);
    field.direct_atlas_by_primitive.assign(scene.primitives.size(),-1);
    std::vector<std::vector<SurfaceSample>> samples(count);
    for(int i=0;i<count;++i){const int p=field.node_primitives[i];samples[i]=surface_samples(scene.primitives[p]);double area=0;
        for(const SurfaceSample& s:samples[i]){field.direct[i]+=(area_irradiance(scene,s.position+s.normal*2e-4,s.normal,p)+
                beam_irradiance(beams,p,s.position))*s.weight;area+=s.weight;}if(area>0)field.direct[i]=field.direct[i]*(1/area);}
    std::vector<double> source_sum(count);for(int receiver=0;receiver<count;++receiver){field.incoming_offset[receiver]=field.coupling.size();
        for(int source=0;source<count;++source){if(receiver==source)continue;double g=0,receiver_area=0;
        const int rp=field.node_primitives[receiver],sp=field.node_primitives[source];
        for(const SurfaceSample& r:samples[receiver]){receiver_area+=r.weight;for(const SurfaceSample& q:samples[source]){
            const Vec3 delta=r.position-q.position;const double d2=norm2(delta);if(d2<1e-7)continue;const Vec3 direction=delta/std::sqrt(d2);
            const double cs=std::max(dot(q.normal,direction),0.0),cr=std::max(dot(r.normal,-direction),0.0);if(cs<=0||cr<=0)continue;
            const RGB trans=segment_transmittance(scene,q.position,r.position,sp,rp);const double scalar=(trans[0]+trans[1]+trans[2])/3;
            g+=r.weight*q.weight*cs*cr*scalar/d2;}}
        if(receiver_area>0)g/=receiver_area;if(g>1e-8){field.coupling.push_back({source,g});source_sum[source]+=g;}}
    }field.incoming_offset[count]=field.coupling.size();field.nonzeros=field.coupling.size();
    for(int receiver=0;receiver<count;++receiver)for(std::size_t edge=field.incoming_offset[receiver];
            edge<field.incoming_offset[receiver+1];++edge){TransportCoupling& coupling=field.coupling[edge];
        if(source_sum[coupling.source]>.82)coupling.weight*=.82/source_sum[coupling.source];}
    field.outgoing_offset.assign(static_cast<std::size_t>(count)+1,0);
    for(const TransportCoupling& coupling:field.coupling)++field.outgoing_offset[coupling.source+1];
    for(int source=0;source<count;++source)field.outgoing_offset[source+1]+=field.outgoing_offset[source];
    field.outgoing.resize(field.coupling.size());
    std::vector<std::size_t> cursor=field.outgoing_offset;for(int receiver=0;receiver<count;++receiver)
        for(std::size_t edge=field.incoming_offset[receiver];edge<field.incoming_offset[receiver+1];++edge){
            const TransportCoupling& coupling=field.coupling[edge];field.outgoing[cursor[coupling.source]++]={receiver,coupling.weight};}
    auto energy=[](const RGB& value){return value[0]+value[1]+value[2];};
    std::vector<RGB> response(count),delta(count);for(int i=0;i<count;++i){const Material& material=
            scene.materials[scene.primitives[field.node_primitives[i]].material];response[i]=material.base*(material.diffuse/pi);
        field.radiance[i]=delta[i]=multiply(field.direct[i],response[i]);}
    std::vector<int> active,next_active;active.reserve(count);next_active.reserve(count);for(int i=0;i<count;++i)
        if(energy(delta[i])>1e-14)active.push_back(i);
    for(int iteration=0;iteration<24;++iteration){std::vector<RGB> next(count);std::vector<std::uint8_t> marked(count);next_active.clear();
        for(int source:active)for(std::size_t edge=field.outgoing_offset[source];edge<field.outgoing_offset[source+1];++edge){
            const TransportRecipient& recipient=field.outgoing[edge];next[recipient.receiver]+=multiply(
                delta[source]*recipient.weight,response[recipient.receiver]);++field.propagated_edges;
            if(!marked[recipient.receiver]){marked[recipient.receiver]=1;next_active.push_back(recipient.receiver);}}
        double residual=0;active.clear();for(int receiver:next_active){const RGB value=next[receiver];residual+=energy(value);
            field.radiance[receiver]+=value;if(energy(value)>1e-14)active.push_back(receiver);}
        delta.swap(next);field.iterations=iteration+1;field.final_residual=residual;if(residual<1e-5)break;}
    for(int i=0;i<count;++i){RGB incident{};for(std::size_t edge=field.incoming_offset[i];edge<field.incoming_offset[i+1];++edge){
            const TransportCoupling& coupling=field.coupling[edge];incident+=field.radiance[coupling.source]*coupling.weight;}
        field.bounce[i]=incident;}
    for(int primitive_id=0;primitive_id<static_cast<int>(scene.primitives.size());++primitive_id){
        const Primitive& primitive=scene.primitives[primitive_id];const Material& material=scene.materials[primitive.material];
        if(material.kind==MaterialKind::Emissive||material.kind==MaterialKind::Mirror||
            material.kind==MaterialKind::Dielectric||material.diffuse<=0||
            (primitive.shape!=Shape::Rectangle&&primitive.shape!=Shape::Sphere))continue;
        if(primitive.shape==Shape::Rectangle){const double a=norm(primitive.u),b=norm(primitive.v);
            if(std::max(a,b)/std::max(std::min(a,b),1e-12)>10)continue;}
        SurfaceIrradianceAtlas atlas;atlas.primitive=primitive_id;atlas.sphere=primitive.shape==Shape::Sphere;
        atlas.width=atlas.sphere?33:33;atlas.height=atlas.sphere?17:33;
        atlas.value.resize(static_cast<std::size_t>(atlas.width)*atlas.height);
        auto surface_point=[&](double u,double v,Vec3& position,Vec3& normal){
            if(atlas.sphere){const double ny=1-2*v,r=std::sqrt(std::max(0.0,1-ny*ny)),angle=2*pi*u-pi;
                normal={r*std::cos(angle),ny,r*std::sin(angle)};position=primitive.center+normal*primitive.radius;}
            else{normal=primitive.normal;position=primitive.origin+primitive.u*u+primitive.v*v;}};
        auto evaluate=[&](double u,double v){Vec3 position,normal;surface_point(u,v,position,normal);++field.direct_atlas_samples;
            return area_irradiance(scene,position+normal*2e-4,normal,primitive_id);};
        constexpr int refinement=4;const int fine_width=(atlas.width-1)*refinement;
        const int fine_height=(atlas.height-1)*refinement;std::map<std::pair<int,int>,RGB> fine_samples;
        auto evaluate_grid=[&](int gx,int gy){const int key_x=atlas.sphere&&gx==fine_width?0:gx;
            const std::pair<int,int> key{key_x,gy};const auto existing=fine_samples.find(key);
            if(existing!=fine_samples.end())return existing->second;const RGB value=evaluate(
                double(key_x)/fine_width,double(gy)/fine_height);fine_samples.emplace(key,value);return value;};
        for(int y=0;y<atlas.height;++y)for(int x=0;x<atlas.width;++x)
            atlas.value[static_cast<std::size_t>(y)*atlas.width+x]=evaluate_grid(x*refinement,y*refinement);
        atlas.refined_offset.assign(static_cast<std::size_t>(atlas.width-1)*(atlas.height-1),-1);
        const RGB response=material.base*(material.diffuse/pi);for(int y=0;y+1<atlas.height;++y)for(int x=0;x+1<atlas.width;++x){
            const RGB centre=evaluate_grid(x*refinement+refinement/2,y*refinement+refinement/2);RGB predicted{};
            for(int channel=0;channel<3;++channel)predicted[channel]=.25*(atlas.value[static_cast<std::size_t>(y)*atlas.width+x][channel]+
                atlas.value[static_cast<std::size_t>(y)*atlas.width+x+1][channel]+
                atlas.value[static_cast<std::size_t>(y+1)*atlas.width+x][channel]+
                atlas.value[static_cast<std::size_t>(y+1)*atlas.width+x+1][channel]);
            int display_error=0;for(int channel=0;channel<3;++channel)display_error=std::max(display_error,std::abs(
                int(tone_byte(centre[channel]*response[channel]))-int(tone_byte(predicted[channel]*response[channel]))));
            if(display_error<=1)continue;const int offset=static_cast<int>(atlas.refined_value.size());
            atlas.refined_offset[static_cast<std::size_t>(y)*(atlas.width-1)+x]=offset;
            for(int ry=0;ry<atlas.refine_side;++ry)for(int rx=0;rx<atlas.refine_side;++rx)
                atlas.refined_value.push_back(evaluate_grid(x*refinement+rx,y*refinement+ry));}
        field.direct_atlas_by_primitive[primitive_id]=static_cast<int>(field.direct_atlas.size());
        field.direct_atlas.push_back(std::move(atlas));}
    return field;
}

double rgb_energy(const RGB& v){return v[0]+v[1]+v[2];}
std::uint8_t tone_byte(double linear){const double mapped=std::pow(std::clamp(1-std::exp(-.75*std::max(linear,0.0)),0.0,1.0),1/2.2);
    return static_cast<std::uint8_t>(std::lround(255*mapped));}

int count_sheet_shadow_samples(const Scene& scene,int receiver){if(receiver<0||scene.area_lights.empty())return 0;int count=0;
    const auto samples=surface_samples(scene.primitives[receiver]);const Primitive& light=scene.primitives[scene.area_lights[0].primitive];
    for(const SurfaceSample& s:samples)for(int y=0;y<3;++y)for(int x=0;x<3;++x){const Vec3 q=light.origin+light.u*((x+.5)/3)+light.v*((y+.5)/3);
        const Vec3 delta=q-s.position;const double distance=norm(delta);if(distance<1e-8||dot(s.normal,delta/distance)<=0)continue;
        const Hit h=first_hit(scene,{s.position+s.normal*2e-4,delta/distance},distance,receiver);if(h.valid&&h.primitive==scene.glass_sheet)++count;}
    return count;}

struct SurfaceRadianceMemoEntry{int primitive;Vec3 position,normal,view;RGB value;bool directional=false;};
struct SpecularMemo{
    std::array<SurfaceRadianceMemoEntry,32> surface_entry;
    std::array<std::uint64_t,32> surface_tag{};
};

struct Camera{Vec3 origin{},forward{},right{},up{};double scale=0,aspect=1;};
struct CameraSpecularField{
    std::vector<SurfaceIrradianceAtlas> atlas;std::vector<int> by_primitive;
    TraceStats construction{};std::uint64_t samples=0;Vec3 origin{};
};

struct TraceContext{
    const Scene& scene;const BeamField& beams;const TransportField& field;TraceStats* stats=nullptr;
    bool sealed_optics=true;double optical_cutoff=1e-5;SpecularMemo* specular_memo=nullptr;
    const CameraSpecularField* camera_specular=nullptr;
};

bool is_area_emitter(const Scene& scene,int primitive){
    return std::any_of(scene.area_lights.begin(),scene.area_lights.end(),[&](const AreaLight& light){
        return light.primitive==primitive;});
}

RGB emitted_radiance(const Scene& scene,const Hit& hit,Vec3 outgoing){
    const Material& material=scene.materials[scene.primitives[hit.primitive].material];
    if(hit.primitive==scene.projector_lens&&!scene.beams.empty()){
        // The lens is the geometric aperture of a collimated bundle, not an
        // omnidirectional point lamp.  A soft angular gate avoids introducing
        // another binary boundary at the edge of its very narrow source cone.
        const BeamBundle& beam=scene.beams.front();const double angle=std::acos(std::clamp(
            dot(unit(outgoing),beam.direction),-1.0,1.0));const double inner=std::max(beam.spread,.0005);
        const double outer=std::max(3*beam.spread,.0025);if(angle>=outer)return {};
        const double t=angle<=inner?1:1-(angle-inner)/(outer-inner);const double gate=t*t*(3-2*t);
        return material.emission*gate;
    }
    if(is_area_emitter(scene,hit.primitive)&&dot(scene.primitives[hit.primitive].normal,outgoing)<=0)return {};
    return material.emission;
}

struct RoughTerminalRelation{double coverage=1;bool area_emitter=false;Hit terminal{};};

RoughTerminalRelation rough_terminal_relation(const Scene& scene,const Ray& ray,int ignore,double roughness){
    const Hit terminal=first_hit(scene,ray,std::numeric_limits<double>::infinity(),ignore);
    RoughTerminalRelation relation;relation.terminal=terminal;if(!terminal.valid)return relation;
    const Primitive& primitive=scene.primitives[terminal.primitive];
    relation.area_emitter=is_area_emitter(scene,terminal.primitive);
    // A rough BRDF receives a finite angular region.  The old single delta ray
    // assigned full ownership at the first infinitesimal tangent hit, creating
    // isolated dots and hard rectangular islands.  This is the analytic
    // inside-edge mass of that angular footprint; broadening outside the region
    // is supplied separately by the compiled area-emitter integral.
    const double alpha=std::max(1e-5,roughness*roughness),range=std::max(terminal.t,1e-6);double margin=alpha;
    if(primitive.shape==Shape::Sphere){const Vec3 delta=primitive.center-ray.origin;
        const double along=dot(delta,ray.direction);const double perpendicular=std::sqrt(std::max(
            0.0,norm2(delta)-along*along));margin=(primitive.radius-perpendicular)/range;
    }else if(primitive.shape==Shape::Rectangle){const Vec3 q=terminal.position-primitive.origin;
        if(std::abs(primitive.gram_det)>1e-16){const double qu=dot(q,primitive.u),qv=dot(q,primitive.v);
            const double a=(qu*primitive.gram_vv-qv*primitive.gram_uv)/primitive.gram_det;
            const double b=(qv*primitive.gram_uu-qu*primitive.gram_uv)/primitive.gram_det;const double edge=std::min({
                a*std::sqrt(primitive.gram_uu),(1-a)*std::sqrt(primitive.gram_uu),b*std::sqrt(primitive.gram_vv),
                (1-b)*std::sqrt(primitive.gram_vv)});margin=edge/range;}
    }else{const Vec3 q=terminal.position-primitive.a;const double d00=dot(primitive.edge1,primitive.edge1);
        const double d01=dot(primitive.edge1,primitive.edge2),d11=dot(primitive.edge2,primitive.edge2);
        const double d20=dot(q,primitive.edge1),d21=dot(q,primitive.edge2);
        const double det=d00*d11-d01*d01;if(std::abs(det)>1e-16){const double b=(d11*d20-d01*d21)/det;
            const double c=(d00*d21-d01*d20)/det,a=1-b-c;const double scale=std::min({
                norm(primitive.edge1),norm(primitive.edge2),norm(primitive.c-primitive.b)});
            margin=std::min({a,b,c})*scale/range;}}
    const double t=std::clamp(margin/alpha,0.0,1.0);relation.coverage=t*t*(3-2*t);return relation;
}

RGB specular_area(const TraceContext& ctx,const Hit& hit,Vec3 view,const Material& material,bool camera_primary=false,
    double path_weight=1){RGB result{};
    if(ctx.stats){++ctx.stats->specular_area_calls;
        if(camera_primary)++ctx.stats->primary_specular_area_calls;else{++ctx.stats->secondary_specular_area_calls;
            if(path_weight<1e-4)++ctx.stats->secondary_specular_weight_lt_1e4;
            if(path_weight<1e-3)++ctx.stats->secondary_specular_weight_lt_1e3;
            if(path_weight<1e-2)++ctx.stats->secondary_specular_weight_lt_1e2;}}
    const double alpha=std::max(.035,material.roughness),exponent=std::min(900.0,std::max(2.0,2/(alpha*alpha)-2));
    const RGB f0=material.kind==MaterialKind::Metal?material.base:RGB{.045,.045,.045};
    for(const AreaLight& area:ctx.scene.area_lights)result+=integrate_rectangular_emitter(ctx.scene,area,
        hit.position+hit.normal*2e-4,hit.normal,hit.primitive,[&](Vec3,Vec3 l,double r2,double nl,double ll,const RGB& trans){
            const Vec3 half=unit(l+view);const double nh=std::max(dot(hit.normal,half),0.0),vh=std::max(dot(view,half),0.0);
            const double lobe=(exponent+2)/(2*pi)*std::pow(nh,exponent);RGB fresnel{};
            for(int c=0;c<3;++c)fresnel[c]=f0[c]+(1-f0[c])*std::pow(1-vh,5);
            return multiply(multiply(area.radiance,trans),fresnel)*(lobe*nl*ll/r2);},ctx.stats,
        material.roughness<=.18?4:3);
    return result;
}

RGB compiled_specular_area(const TraceContext& ctx,const Hit& hit,Vec3 view,const Material& material,bool camera_primary,
    double path_weight){
    const bool camera_direction=ctx.camera_specular&&(camera_primary||dot(unit(ctx.camera_specular->origin-hit.position),view)>1-1e-11);
    const int atlas_index=camera_direction&&hit.primitive>=0&&
        hit.primitive<static_cast<int>(ctx.camera_specular->by_primitive.size())?
        ctx.camera_specular->by_primitive[hit.primitive]:-1;
    if(atlas_index<0)return specular_area(ctx,hit,view,material,camera_primary,path_weight);
    if(ctx.stats)++ctx.stats->camera_specular_gathers;
    return sample_surface_atlas(ctx.camera_specular->atlas[atlas_index],ctx.scene.primitives[hit.primitive],hit.position);
}

RGB compiled_area_irradiance(const TraceContext& ctx,const Hit& hit){const int atlas_index=
        hit.primitive>=0&&hit.primitive<static_cast<int>(ctx.field.direct_atlas_by_primitive.size())?
        ctx.field.direct_atlas_by_primitive[hit.primitive]:-1;
    if(atlas_index<0){if(ctx.stats)++ctx.stats->direct_exact_calls;
        return area_irradiance(ctx.scene,hit.position+hit.normal*2e-4,hit.normal,hit.primitive,ctx.stats);}
    const Primitive& primitive=ctx.scene.primitives[hit.primitive];
    if(primitive.shape==Shape::Rectangle&&dot(hit.normal,primitive.normal)<0){if(ctx.stats)++ctx.stats->direct_exact_calls;
        return area_irradiance(ctx.scene,hit.position+hit.normal*2e-4,hit.normal,hit.primitive,ctx.stats);
    }
    if(ctx.stats)++ctx.stats->direct_atlas_gathers;
    return sample_surface_atlas(ctx.field.direct_atlas[atlas_index],primitive,hit.position);
}

std::uint64_t radiance_memo_hash(int primitive,Vec3 position,Vec3 normal,Vec3 view,bool directional){
    std::uint64_t hash=1469598103934665603ULL;auto mix=[&](std::uint64_t value){value^=value>>30;
        value*=0xbf58476d1ce4e5b9ULL;value^=value>>27;value*=0x94d049bb133111ebULL;value^=value>>31;
        hash^=value+0x9e3779b97f4a7c15ULL+(hash<<6)+(hash>>2);};
    mix(static_cast<std::uint64_t>(primitive+1));mix(std::bit_cast<std::uint64_t>(position.x));
    mix(std::bit_cast<std::uint64_t>(position.y));mix(std::bit_cast<std::uint64_t>(position.z));
    mix(std::bit_cast<std::uint64_t>(normal.x));mix(std::bit_cast<std::uint64_t>(normal.y));
    mix(std::bit_cast<std::uint64_t>(normal.z));if(directional){mix(std::bit_cast<std::uint64_t>(view.x));
        mix(std::bit_cast<std::uint64_t>(view.y));mix(std::bit_cast<std::uint64_t>(view.z));}
    return hash|(1ULL<<63);
}

RGB base_radiance(const TraceContext& ctx,const Hit& hit,Vec3 view,bool camera_primary=false,double path_weight=1){const Primitive& p=ctx.scene.primitives[hit.primitive];
    const Material& m=ctx.scene.materials[p.material];if(m.kind==MaterialKind::Emissive)return emitted_radiance(ctx.scene,hit,view);
    const bool directional=m.kind==MaterialKind::Glossy||m.kind==MaterialKind::Metal;
    const std::uint64_t memo_tag=ctx.specular_memo?radiance_memo_hash(hit.primitive,hit.position,hit.normal,view,directional):0;
    const std::size_t memo_mask=ctx.specular_memo?ctx.specular_memo->surface_entry.size()-1:0;
    const std::size_t memo_home=memo_tag&memo_mask;std::size_t memo_slot=memo_home;
    if(ctx.specular_memo)for(int probe=0;probe<4;++probe){const std::size_t slot=(memo_home+probe)&memo_mask;
        const std::uint64_t tag=ctx.specular_memo->surface_tag[slot];if(tag==0){memo_slot=slot;break;}
        if(tag!=memo_tag)continue;const SurfaceRadianceMemoEntry& entry=ctx.specular_memo->surface_entry[slot];
        const bool same_view=!entry.directional||(entry.view.x==view.x&&entry.view.y==view.y&&entry.view.z==view.z);
        if(entry.primitive==hit.primitive&&same_view&&entry.position.x==hit.position.x&&entry.position.y==hit.position.y&&
            entry.position.z==hit.position.z&&entry.normal.x==hit.normal.x&&entry.normal.y==hit.normal.y&&
            entry.normal.z==hit.normal.z){if(ctx.stats)++ctx.stats->surface_radiance_memo_hits;return entry.value;}}
    RGB incident=compiled_area_irradiance(ctx,hit)+
        beam_irradiance(ctx.beams,hit.primitive,hit.position);
    const int node=ctx.field.index_by_primitive[hit.primitive];if(node>=0)incident+=ctx.field.bounce[node];
    RGB result=multiply(incident,m.base)*(m.diffuse/pi);
    if(m.kind==MaterialKind::Glossy||m.kind==MaterialKind::Metal)result+=compiled_specular_area(
        ctx,hit,view,m,camera_primary,path_weight);
    if(ctx.specular_memo){ctx.specular_memo->surface_entry[memo_slot]=
            {hit.primitive,hit.position,hit.normal,view,result,directional};ctx.specular_memo->surface_tag[memo_slot]=memo_tag;
        if(ctx.stats)++ctx.stats->surface_radiance_memo_stores;}
    return result;
}

CameraSpecularField compile_viewer_origin_specular_field(const TraceContext& context,Vec3 viewer_origin){CameraSpecularField field;
    // This field is attached to the eye point, not its sensor chart. Rotation,
    // roll, FOV, aspect, and resolution leave every surface-to-eye direction
    // unchanged and must not invalidate it.
    field.origin=viewer_origin;
    field.by_primitive.assign(context.scene.primitives.size(),-1);TraceContext evaluator=context;
    evaluator.stats=&field.construction;evaluator.specular_memo=nullptr;evaluator.camera_specular=nullptr;
    for(int primitive_id=0;primitive_id<static_cast<int>(context.scene.primitives.size());++primitive_id){
        const Primitive& primitive=context.scene.primitives[primitive_id];const Material& material=
            context.scene.materials[primitive.material];if((material.kind!=MaterialKind::Glossy&&material.kind!=MaterialKind::Metal)||
            (primitive.shape!=Shape::Rectangle&&primitive.shape!=Shape::Sphere))continue;
        SurfaceIrradianceAtlas atlas;atlas.primitive=primitive_id;atlas.sphere=primitive.shape==Shape::Sphere;
        atlas.width=atlas.sphere?33:33;atlas.height=atlas.sphere?17:33;
        atlas.value.resize(static_cast<std::size_t>(atlas.width)*atlas.height);
        auto surface_point=[&](double u,double v,Vec3& position,Vec3& normal){
            if(atlas.sphere){const double ny=1-2*v,r=std::sqrt(std::max(0.0,1-ny*ny)),angle=2*pi*u-pi;
                normal={r*std::cos(angle),ny,r*std::sin(angle)};position=primitive.center+normal*primitive.radius;}
            else{normal=primitive.normal;position=primitive.origin+primitive.u*u+primitive.v*v;}
            const Vec3 view=unit(viewer_origin-position);if(!atlas.sphere&&dot(normal,view)<0)normal=-normal;};
        auto evaluate=[&](double u,double v){Vec3 position,normal;surface_point(u,v,position,normal);const Vec3 view=unit(viewer_origin-position);
            if(atlas.sphere&&dot(normal,view)<=0)return RGB{};
            Hit hit;hit.valid=true;hit.primitive=primitive_id;hit.position=position;hit.normal=normal;hit.geometric_normal=
                primitive.shape==Shape::Sphere?unit(position-primitive.center):primitive.normal;hit.front=dot(view,hit.geometric_normal)>0;
            ++field.samples;return specular_area(evaluator,hit,view,material,true);};
        constexpr int refinement=4;const int fine_width=(atlas.width-1)*refinement;
        const int fine_height=(atlas.height-1)*refinement;std::map<std::pair<int,int>,RGB> fine_samples;
        auto evaluate_grid=[&](int gx,int gy){const int key_x=atlas.sphere&&gx==fine_width?0:gx;
            const std::pair<int,int> key{key_x,gy};const auto existing=fine_samples.find(key);
            if(existing!=fine_samples.end())return existing->second;const RGB value=evaluate(
                double(key_x)/fine_width,double(gy)/fine_height);fine_samples.emplace(key,value);return value;};
        for(int y=0;y<atlas.height;++y)for(int x=0;x<atlas.width;++x)
            atlas.value[static_cast<std::size_t>(y)*atlas.width+x]=evaluate_grid(x*refinement,y*refinement);
        atlas.refined_offset.assign(static_cast<std::size_t>(atlas.width-1)*(atlas.height-1),-1);
        for(int y=0;y+1<atlas.height;++y)for(int x=0;x+1<atlas.width;++x){const RGB centre=
                evaluate_grid(x*refinement+refinement/2,y*refinement+refinement/2);RGB predicted{};
            for(int channel=0;channel<3;++channel)predicted[channel]=.25*(atlas.value[static_cast<std::size_t>(y)*atlas.width+x][channel]+
                atlas.value[static_cast<std::size_t>(y)*atlas.width+x+1][channel]+
                atlas.value[static_cast<std::size_t>(y+1)*atlas.width+x][channel]+
                atlas.value[static_cast<std::size_t>(y+1)*atlas.width+x+1][channel]);
            int display_error=0;for(int channel=0;channel<3;++channel){display_error=std::max(display_error,std::abs(
                int(tone_byte(centre[channel]))-int(tone_byte(predicted[channel]))));}
            if(display_error<=1)continue;
            const int offset=static_cast<int>(atlas.refined_value.size());atlas.refined_offset[
                static_cast<std::size_t>(y)*(atlas.width-1)+x]=offset;
            for(int ry=0;ry<atlas.refine_side;++ry)for(int rx=0;rx<atlas.refine_side;++rx)
                atlas.refined_value.push_back(evaluate_grid(x*refinement+rx,y*refinement+ry));}
        field.by_primitive[primitive_id]=static_cast<int>(field.atlas.size());field.atlas.push_back(std::move(atlas));}
    return field;
}

void signature_push(std::uint64_t* signature,int primitive,int channel){if(!signature)return;
    std::uint64_t x=static_cast<std::uint64_t>(primitive+1)*0x9e3779b97f4a7c15ULL+static_cast<std::uint64_t>(channel+3);
    x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;
    *signature=(*signature^(x+0x9e3779b97f4a7c15ULL+(*signature<<6)+(*signature>>2)));
}

double trace_channel_recursive(const TraceContext& ctx,const Ray& ray,int depth,int channel,std::uint64_t* signature,
    int ignore=-1,double spectral_coordinate=std::numeric_limits<double>::quiet_NaN(),bool exclude_area_emitters=false,
    const Hit* prefetched=nullptr){
    if(!std::isfinite(spectral_coordinate))spectral_coordinate=channel;
    if(depth<=0)return 0;if(ctx.stats)++ctx.stats->secondary;const Hit hit=prefetched?*prefetched:
        first_hit(ctx.scene,ray,std::numeric_limits<double>::infinity(),ignore);
    if(!hit.valid)return 0;signature_push(signature,hit.primitive,channel);const Material& m=ctx.scene.materials[ctx.scene.primitives[hit.primitive].material];
    if(m.kind==MaterialKind::Emissive)return exclude_area_emitters&&is_area_emitter(ctx.scene,hit.primitive)?0:
        emitted_radiance(ctx.scene,hit,-ray.direction)[channel];
    if(m.kind==MaterialKind::Dielectric){if(ctx.stats){++ctx.stats->dielectric;
        if(hit.primitive==ctx.scene.prism_bottom)++ctx.stats->prism_bottom_events;
        if(hit.primitive==ctx.scene.prism_top)++ctx.stats->prism_top_events;}
        const double index=spectral_ior(m,spectral_coordinate),ni=hit.front?1:index,nt=hit.front?index:1;
        const double F=schlick(std::abs(dot(ray.direction,hit.normal)),ni,nt);const Vec3 rd=unit(reflect(ray.direction,hit.normal));
        const double reflected=trace_channel_recursive(ctx,{hit.position+rd*3e-4,rd},depth-1,channel,signature,
            hit.primitive,spectral_coordinate,exclude_area_emitters);
        Vec3 td=ray.direction;double transmitted=0;if(m.thin||refract(ray.direction,hit.normal,ni/nt,td)){
            if(m.thin)td=ray.direction;transmitted=trace_channel_recursive(ctx,{hit.position+td*3e-4,td},depth-1,
                channel,signature,hit.primitive,spectral_coordinate,exclude_area_emitters);}
        const double absorb=std::exp(-m.absorption[channel]*(m.thin?.10:.24));return m.base[channel]*absorb*((1-F)*transmitted+F*reflected);}
    if(m.kind==MaterialKind::Mirror){if(ctx.stats)++ctx.stats->mirror;const Vec3 d=unit(reflect(ray.direction,hit.normal));
        if(ctx.stats){const Hit next=first_hit(ctx.scene,{hit.position+d*3e-4,d},std::numeric_limits<double>::infinity(),hit.primitive);
            if(next.valid&&rgb_energy(beam_irradiance(ctx.beams,next.primitive,next.position))>1e-5)++ctx.stats->specular_caustic;}
        return m.base[channel]*trace_channel_recursive(ctx,{hit.position+d*3e-4,d},depth-1,channel,signature,
            hit.primitive,spectral_coordinate,exclude_area_emitters);}
    const RGB base=base_radiance(ctx,hit,-ray.direction);if(m.kind==MaterialKind::Metal){if(ctx.stats)++ctx.stats->metal;
        // The former five-point cross was not an integral of the rough lobe: it
        // produced five coherent copies of every reflected object.  Until the
        // directional source regions are compiled, retain one exact terminal
        // relation.  The continuous area-emitter lobe is already integrated by
        // specular_area(), so this removes false images and four recursive paths.
        const Vec3 d=unit(reflect(ray.direction,hit.normal));const Ray continuation{hit.position+d*3e-4,d};
        const RoughTerminalRelation terminal=rough_terminal_relation(ctx.scene,continuation,hit.primitive,m.roughness);
        const double reflected=terminal.area_emitter?0:terminal.coverage*
            trace_channel_recursive(ctx,continuation,depth-1,channel,signature,hit.primitive,spectral_coordinate,true,&terminal.terminal);
        return .12*base[channel]+.88*m.base[channel]*reflected;}
    if(m.kind==MaterialKind::Glossy){const Vec3 d=unit(reflect(ray.direction,hit.normal));const Ray continuation{hit.position+d*3e-4,d};
        const RoughTerminalRelation terminal=rough_terminal_relation(ctx.scene,continuation,hit.primitive,m.roughness);
        const double reflected=terminal.area_emitter?0:terminal.coverage*
            trace_channel_recursive(ctx,continuation,depth-1,channel,signature,hit.primitive,spectral_coordinate,true,&terminal.terminal);
        return base[channel]+(.04+.20*(1-m.roughness))*reflected;}
    return base[channel];
}

struct OpticalPacket{
    Ray ray{};double weight=1;int interactions=0,ignore=-1;
    int previous=-1,previous_previous=-1;double previous_weight=1,previous_previous_weight=1;
    bool feedback=false;double feedback_ratio=0;bool exclude_area_emitters=false;
    Vec3 previous_direction{},previous_previous_direction{};Hit prefetched{};bool has_prefetched=false;
};

double trace_channel_sealed(const TraceContext& ctx,const Ray& initial,int depth,int channel,
    std::uint64_t* signature,int ignore=-1,double spectral_coordinate=std::numeric_limits<double>::quiet_NaN(),
    bool exclude_area_emitters=false,const Hit* prefetched=nullptr){
    if(!std::isfinite(spectral_coordinate))spectral_coordinate=channel;
    // A dielectric cavity is a residual-energy series, not a recursive path
    // tree.  Each packet surrenders its exits once.  An A-B-A face return marks
    // a feedback relation; after that, the loop is marched as a local series
    // and sealed when its geometric tail is below the radiance budget.
    const double packet_cutoff=ctx.optical_cutoff;
    const int maximum_interactions=std::max(16,depth*4);double result=0;
    std::vector<OpticalPacket> packets;packets.reserve(24);
    OpticalPacket initial_packet;initial_packet.ray=initial;initial_packet.ignore=ignore;
    initial_packet.previous=ignore;initial_packet.exclude_area_emitters=exclude_area_emitters;
    if(prefetched){initial_packet.prefetched=*prefetched;initial_packet.has_prefetched=true;}
    packets.push_back(initial_packet);
    while(!packets.empty()){
        if(ctx.stats)ctx.stats->maximum_optical_packets=std::max<std::uint64_t>(
            ctx.stats->maximum_optical_packets,packets.size());
        OpticalPacket packet=packets.back();packets.pop_back();
        if(packet.weight<=packet_cutoff||packet.interactions>=maximum_interactions){
            if(ctx.stats&&packet.feedback){++ctx.stats->sealed_feedback_tails;
                ctx.stats->sealed_residual_weight+=packet.weight;}continue;}
        if(ctx.stats)++ctx.stats->secondary;const Hit hit=packet.has_prefetched?packet.prefetched:first_hit(ctx.scene,packet.ray,
            std::numeric_limits<double>::infinity(),packet.ignore);if(!hit.valid)continue;
        signature_push(signature,hit.primitive,channel);const Material& material=
            ctx.scene.materials[ctx.scene.primitives[hit.primitive].material];
        const bool direction_return=norm2(packet.previous_previous_direction)>.5&&
            dot(unit(packet.ray.direction),unit(packet.previous_previous_direction))>1-1e-9;
        if(packet.previous_previous==hit.primitive&&direction_return){const double ratio=packet.previous_previous_weight>0?
                packet.weight/packet.previous_previous_weight:0;
            if(ctx.stats){++ctx.stats->feedback_returns;if(!packet.feedback)++ctx.stats->feedback_loops;}
            packet.feedback=true;if(ratio>0&&ratio<1)packet.feedback_ratio=ratio;}
        auto continuation=[&](Vec3 direction,double weight,bool loop_remainder=false,bool exclude_emitter=false,
            const Hit* next_hit=nullptr){
            if(weight<=0)return;OpticalPacket child=packet;child.ray={hit.position+direction*3e-4,direction};
            child.weight=weight;child.ignore=hit.primitive;child.interactions=packet.interactions+1;
            child.previous_previous=packet.previous;child.previous_previous_weight=packet.previous_weight;
            child.previous=hit.primitive;child.previous_weight=packet.weight;
            child.previous_previous_direction=packet.previous_direction;
            child.previous_direction=packet.ray.direction;
            child.exclude_area_emitters=packet.exclude_area_emitters||exclude_emitter;
            child.has_prefetched=next_hit!=nullptr;if(next_hit)child.prefetched=*next_hit;
            if(loop_remainder&&child.feedback&&child.feedback_ratio>0&&child.feedback_ratio<1){const double bound=
                    child.weight/(1-child.feedback_ratio);
                if(bound<=packet_cutoff){if(ctx.stats){++ctx.stats->sealed_feedback_tails;
                        ctx.stats->sealed_residual_weight+=bound;}return;}}
            packets.push_back(child);
        };
        if(material.kind==MaterialKind::Emissive){
            if(!(packet.exclude_area_emitters&&is_area_emitter(ctx.scene,hit.primitive)))
                result+=packet.weight*emitted_radiance(ctx.scene,hit,-packet.ray.direction)[channel];
            continue;}
        if(material.kind==MaterialKind::Dielectric){if(ctx.stats){++ctx.stats->dielectric;
                if(hit.primitive==ctx.scene.prism_bottom)++ctx.stats->prism_bottom_events;
                if(hit.primitive==ctx.scene.prism_top)++ctx.stats->prism_top_events;}
            const double index=spectral_ior(material,spectral_coordinate),ni=hit.front?1:index,nt=hit.front?index:1;
            const double fresnel=schlick(std::abs(dot(packet.ray.direction,hit.normal)),ni,nt);
            const double common=packet.weight*material.base[channel]*
                std::exp(-material.absorption[channel]*(material.thin?.10:.24));
            const Vec3 reflected=unit(reflect(packet.ray.direction,hit.normal));
            continuation(reflected,common*fresnel,packet.feedback);Vec3 transmitted=packet.ray.direction;
            if(material.thin||refract(packet.ray.direction,hit.normal,ni/nt,transmitted)){
                if(material.thin)transmitted=packet.ray.direction;continuation(transmitted,common*(1-fresnel));}
            continue;}
        if(material.kind==MaterialKind::Mirror){if(ctx.stats)++ctx.stats->mirror;
            const Vec3 direction=unit(reflect(packet.ray.direction,hit.normal));continuation(direction,
                packet.weight*material.base[channel],packet.feedback);continue;}
        const RGB base=base_radiance(ctx,hit,-packet.ray.direction,false,packet.weight);
        if(material.kind==MaterialKind::Metal){if(ctx.stats)++ctx.stats->metal;result+=packet.weight*.12*base[channel];
            const Vec3 direction=unit(reflect(packet.ray.direction,hit.normal));const RoughTerminalRelation terminal=
                rough_terminal_relation(ctx.scene,{hit.position+direction*3e-4,direction},hit.primitive,material.roughness);
            if(!terminal.area_emitter){continuation(direction,
                packet.weight*.88*material.base[channel]*terminal.coverage,packet.feedback,true,&terminal.terminal);}continue;}
        if(material.kind==MaterialKind::Glossy){result+=packet.weight*base[channel];const Vec3 direction=
            unit(reflect(packet.ray.direction,hit.normal));const RoughTerminalRelation terminal=
                rough_terminal_relation(ctx.scene,{hit.position+direction*3e-4,direction},hit.primitive,material.roughness);
            if(!terminal.area_emitter){continuation(direction,
                packet.weight*(.04+.20*(1-material.roughness))*terminal.coverage,packet.feedback,true,&terminal.terminal);}continue;}
        result+=packet.weight*base[channel];
    }
    return result;
}

double trace_channel(const TraceContext& ctx,const Ray& ray,int depth,int channel,std::uint64_t* signature,int ignore=-1,
    double spectral_coordinate=std::numeric_limits<double>::quiet_NaN(),bool exclude_area_emitters=false,
    const Hit* prefetched=nullptr){
    return ctx.sealed_optics?trace_channel_sealed(ctx,ray,depth,channel,signature,ignore,spectral_coordinate,
        exclude_area_emitters,prefetched):trace_channel_recursive(ctx,ray,depth,channel,signature,ignore,spectral_coordinate,
        exclude_area_emitters,prefetched);
}

bool append_visible_path_signature(const TraceContext&,const Ray&,const Hit&,int,int,std::uint64_t&,
    std::uint64_t*,int);

struct SpectralTraceSample{double coordinate=0,value=0;std::uint64_t signature=0;};

void merge_signature(std::uint64_t* destination,std::uint64_t source){if(!destination)return;
    *destination^=source+0x9e3779b97f4a7c15ULL+(*destination<<6)+(*destination>>2);}

SpectralTraceSample trace_primary_dielectric_sample(const TraceContext& ctx,const Ray& ray,const Hit& hit,
    const Material& material,int channel,double coordinate){
    if(ctx.stats){++ctx.stats->dielectric;if(hit.primitive==ctx.scene.prism_bottom)++ctx.stats->prism_bottom_events;
        if(hit.primitive==ctx.scene.prism_top)++ctx.stats->prism_top_events;}
    SpectralTraceSample sample;sample.coordinate=coordinate;const double index=spectral_ior(material,coordinate);
    const double ni=hit.front?1:index,nt=hit.front?index:1;
    const double fresnel=schlick(std::abs(dot(ray.direction,hit.normal)),ni,nt);
    const Vec3 reflected_direction=unit(reflect(ray.direction,hit.normal));const double reflected=trace_channel(ctx,
        {hit.position+reflected_direction*3e-4,reflected_direction},7,channel,&sample.signature,hit.primitive,coordinate);
    Vec3 transmitted_direction=ray.direction;double transmitted=0;
    if(material.thin||refract(ray.direction,hit.normal,ni/nt,transmitted_direction)){if(material.thin)transmitted_direction=ray.direction;
        transmitted=trace_channel(ctx,{hit.position+transmitted_direction*3e-4,transmitted_direction},7,channel,
            &sample.signature,hit.primitive,coordinate);}
    sample.value=material.base[channel]*std::exp(-material.absorption[channel]*(material.thin?.10:.24))*
        ((1-fresnel)*transmitted+fresnel*reflected);return sample;
}

std::uint64_t spectral_path_signature(const TraceContext& ctx,const Ray& primary_ray,const Hit& primary_hit,
    double coordinate){
    std::uint64_t signature=0;signature_push(&signature,primary_hit.primitive,-73);Ray ray=primary_ray;Hit hit=primary_hit;
    for(int depth=0;depth<8;++depth){const Material& material=
            ctx.scene.materials[ctx.scene.primitives[hit.primitive].material];Vec3 direction{};
        if(material.kind==MaterialKind::Mirror||material.kind==MaterialKind::Metal||material.kind==MaterialKind::Glossy)
            direction=unit(reflect(ray.direction,hit.normal));
        else if(material.kind==MaterialKind::Dielectric){const Vec3 reflected=unit(reflect(ray.direction,hit.normal));
            const Hit reflected_hit=first_hit(ctx.scene,{hit.position+reflected*3e-4,reflected},
                std::numeric_limits<double>::infinity(),hit.primitive);
            if(reflected_hit.valid)signature_push(&signature,reflected_hit.primitive,129+depth);
            if(material.thin)direction=ray.direction;else{const double index=spectral_ior(material,coordinate);
                const double ni=hit.front?1:index,nt=hit.front?index:1;
                if(!refract(ray.direction,hit.normal,ni/nt,direction))direction=reflected;}}
        else break;ray={hit.position+direction*3e-4,direction};
        hit=first_hit(ctx.scene,ray,std::numeric_limits<double>::infinity(),hit.primitive);if(!hit.valid)break;
        signature_push(&signature,hit.primitive,151+depth);}
    return signature;
}

double integrate_spectral_band(const TraceContext& ctx,const Ray& ray,const Hit& hit,const Material& material,
    int channel,std::uint64_t* signature){
    struct Interval{double begin=0,end=0;std::uint64_t topology=0;};std::vector<Interval> leaves;
    auto subdivide=[&](auto&& self,double begin,std::uint64_t begin_signature,double end,
        std::uint64_t end_signature,int depth)->void{const double middle=(begin+end)*.5;
        const std::uint64_t middle_signature=spectral_path_signature(ctx,ray,hit,middle);
        if((begin_signature==middle_signature&&middle_signature==end_signature)||depth>=10){
            leaves.push_back({begin,end,middle_signature});return;}
        self(self,begin,begin_signature,middle,middle_signature,depth+1);
        self(self,middle,middle_signature,end,end_signature,depth+1);};
    const double begin=channel-.5,end=channel+.5;constexpr int seed_intervals=8;
    std::array<std::uint64_t,seed_intervals+1> topology{};for(int i=0;i<=seed_intervals;++i)
        topology[i]=spectral_path_signature(ctx,ray,hit,begin+(end-begin)*i/seed_intervals);
    for(int i=0;i<seed_intervals;++i)subdivide(subdivide,begin+(end-begin)*i/seed_intervals,topology[i],
        begin+(end-begin)*(i+1)/seed_intervals,topology[i+1],0);
    std::vector<Interval> regions;for(const Interval& leaf:leaves){if(!regions.empty()&&regions.back().topology==leaf.topology)
            regions.back().end=leaf.end;else regions.push_back(leaf);}
    double integral=0;for(const Interval& region:regions){const SpectralTraceSample sample=
            trace_primary_dielectric_sample(ctx,ray,hit,material,channel,(region.begin+region.end)*.5);
        merge_signature(signature,sample.signature);integral+=sample.value*(region.end-region.begin);}
    return integral/(end-begin);
}

RGB trace_primary(const TraceContext& ctx,const Ray& ray,const Hit& hit,std::uint64_t* signature){if(!hit.valid)return {};
    if(ctx.stats)++ctx.stats->primary;const Material& m=ctx.scene.materials[ctx.scene.primitives[hit.primitive].material];
    signature_push(signature,hit.primitive,-1);if(m.kind==MaterialKind::Diffuse||m.kind==MaterialKind::Emissive)
        return base_radiance(ctx,hit,-ray.direction);
    SpecularMemo memo;TraceContext local=ctx;if(!local.specular_memo)local.specular_memo=&memo;
    const bool prism_face=m.kind==MaterialKind::Dielectric&&
        hit.primitive>=local.scene.prism_volume&&hit.primitive<=local.scene.prism_top;
    std::array<std::uint64_t,3> spectral_paths{};if(prism_face)for(int band=0;band<3;++band)
        spectral_paths[band]=spectral_path_signature(local,ray,hit,band);
    const bool mixed_spectrum=prism_face&&
        (spectral_paths[0]!=spectral_paths[1]||spectral_paths[1]!=spectral_paths[2]);
    const bool shared_base=m.kind==MaterialKind::Metal||m.kind==MaterialKind::Glossy;
    const RGB primary_base=shared_base?base_radiance(local,hit,-ray.direction,true):RGB{};
    const bool reflected_primary=m.kind==MaterialKind::Mirror||m.kind==MaterialKind::Metal||m.kind==MaterialKind::Glossy;
    const Vec3 reflected_direction=reflected_primary?unit(reflect(ray.direction,hit.normal)):Vec3{};
    const Ray reflected_continuation=reflected_primary?Ray{hit.position+reflected_direction*3e-4,reflected_direction}:Ray{};
    RoughTerminalRelation rough_terminal{};if(shared_base)rough_terminal=
        rough_terminal_relation(local.scene,reflected_continuation,hit.primitive,m.roughness);
    // Geometry is shared by all spectral channels. One reflected hit supplies
    // both the caustic diagnostic and the first event of all three RGB marches.
    Hit mirror_terminal;const bool mirror_primary=m.kind==MaterialKind::Mirror;
    if(mirror_primary){mirror_terminal=first_hit(local.scene,reflected_continuation,
            std::numeric_limits<double>::infinity(),hit.primitive);
        if(local.stats&&mirror_terminal.valid&&rgb_energy(beam_irradiance(
                local.beams,mirror_terminal.primitive,mirror_terminal.position))>1e-5)
            ++local.stats->specular_caustic;}
    RGB result{};for(int c=0;c<3;++c){if(m.kind==MaterialKind::Dielectric){
            if(mixed_spectrum)result[c]=integrate_spectral_band(local,ray,hit,m,c,signature);
            else{const SpectralTraceSample sample=trace_primary_dielectric_sample(local,ray,hit,m,c,c);
                merge_signature(signature,sample.signature);result[c]=sample.value;}
        }else if(m.kind==MaterialKind::Mirror){
            result[c]=m.base[c]*trace_channel(local,reflected_continuation,7,c,signature,hit.primitive,
                std::numeric_limits<double>::quiet_NaN(),false,&mirror_terminal);
        }else if(m.kind==MaterialKind::Metal){
            const double reflected=rough_terminal.area_emitter?0:rough_terminal.coverage*
                trace_channel(local,reflected_continuation,6,c,signature,hit.primitive,
                    std::numeric_limits<double>::quiet_NaN(),true,&rough_terminal.terminal);
            result[c]=.12*primary_base[c]+.88*m.base[c]*reflected;
        }else{const double reflected=
                rough_terminal.area_emitter?0:rough_terminal.coverage*trace_channel(local,reflected_continuation,5,c,signature,hit.primitive,
                    std::numeric_limits<double>::quiet_NaN(),true,&rough_terminal.terminal);
            result[c]=primary_base[c]+(.04+.20*(1-m.roughness))*reflected;}}
    return clamp_rgb(result);
}

Camera make_look_camera(int width,int height,Vec3 origin,Vec3 target,Vec3 up_hint,double half_fov_degrees){Camera c;
    c.origin=origin;c.forward=unit(target-origin);c.right=unit(cross(c.forward,up_hint));c.up=cross(c.right,c.forward);
    c.scale=std::tan(half_fov_degrees*pi/180);c.aspect=double(width)/height;return c;}
Camera make_camera(int width,int height){return make_look_camera(width,height,{0,2.62,8.15},{0,2.02,-4.75},{0,1,0},26);}
Camera make_original_variant_camera(int width,int height,const std::string& mode){
    if(mode=="default-high")return make_look_camera(width,height,{0,5.35,8.15},{0,1.85,-4.75},{0,1,0},26);
    if(mode=="default-low")return make_look_camera(width,height,{0,.35,8.15},{0,1.55,-4.75},{0,1,0},26);
    if(mode=="default-closer")return make_look_camera(width,height,{0,2.32,1.70},{0,2.02,-4.75},{0,1,0},26);
    throw std::runtime_error("unknown original camera variant: "+mode);}
Camera make_prism_overhead_camera(int width,int height){const Vec3 centre{-2.193,0,-3.960};
    return make_look_camera(width,height,{centre.x,5.70,centre.z},{centre.x,.72,centre.z},{0,0,-1},18);}
Camera make_prism_orbit_camera(int width,int height,double azimuth_degrees){const Vec3 centre{-2.193,.78,-3.960};
    const double azimuth=azimuth_degrees*pi/180,radius=2.75;const Vec3 origin{
        centre.x+radius*std::sin(azimuth),4.35,centre.z+radius*std::cos(azimuth)};
    return make_look_camera(width,height,origin,centre,{0,1,0},22);}
Camera make_camera_mode(int width,int height,const std::string& mode){if(mode=="default")return make_camera(width,height);
    if(mode=="default-high"||mode=="default-low"||mode=="default-closer")
        return make_original_variant_camera(width,height,mode);
    if(mode=="prism-overhead")return make_prism_overhead_camera(width,height);
    if(mode=="prism-front")return make_prism_orbit_camera(width,height,0);
    if(mode=="prism-right")return make_prism_orbit_camera(width,height,90);
    if(mode=="prism-back")return make_prism_orbit_camera(width,height,180);
    if(mode=="prism-left")return make_prism_orbit_camera(width,height,270);
    throw std::runtime_error("unknown camera mode: "+mode);}

struct JourneyKey{Vec3 position{},target{};double half_fov_degrees=27;};
struct JourneyPose{Vec3 position{},target{};double half_fov_degrees=27;};

const std::array<JourneyKey,12>& journey_keys(){static const std::array<JourneyKey,12> keys{{
    // The loop stays in the scene's open front volume or above its object
    // envelope.  Earlier keys threaded between the analytic sheets: although
    // the eye point remained collision-free, a wide frustum made those sheets
    // fill whole frames.  These poses preserve close studies without grazing.
    {{0.00,2.80,7.20},{0.00,1.40,-4.85},27.0},
    {{2.10,2.55,3.15},{-0.40,1.20,-4.85},27.0},
    {{3.75,2.60,-0.40},{0.20,1.10,-5.10},26.0},
    {{4.15,3.20,-3.20},{4.02,1.00,-8.10},24.5},
    {{3.25,4.85,-3.35},{1.35,1.00,-5.70},25.0},
    {{0.75,5.15,-2.35},{0.20,.95,-5.25},25.0},
    {{-1.65,5.10,-1.70},{-2.18,.85,-3.85},24.5},
    {{-3.85,3.45,-1.05},{-2.18,.90,-3.90},24.5},
    {{-3.30,2.00,.85},{-1.15,1.00,-4.65},25.0},
    {{-1.65,1.85,2.55},{-.40,1.00,-5.00},26.0},
    {{-.35,2.25,3.55},{1.45,1.00,-5.35},26.0},
    {{-.80,3.30,5.20},{-.55,1.20,-4.70},27.0}
}};return keys;}

Vec3 catmull_rom(Vec3 p0,Vec3 p1,Vec3 p2,Vec3 p3,double t){const double t2=t*t,t3=t2*t;
    return (p1*2+(p2-p0)*t+(p0*2-p1*5+p2*4-p3)*t2+(-p0+p1*3-p2*3+p3)*t3)*.5;}
double catmull_rom(double p0,double p1,double p2,double p3,double t){const double t2=t*t,t3=t2*t;
    return .5*(2*p1+(p2-p0)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3);}

JourneyPose journey_pose(double parameter,const std::string& scene_mode="standard"){const auto& keys=journey_keys();const int count=static_cast<int>(keys.size());
    parameter-=std::floor(parameter);const double scaled=parameter*count;const int i=static_cast<int>(std::floor(scaled));
    const double t=scaled-i;auto key=[&](int offset)->const JourneyKey&{return keys[(i+offset+count)%count];};
    JourneyPose pose{catmull_rom(key(-1).position,key(0).position,key(1).position,key(2).position,t),
        catmull_rom(key(-1).target,key(0).target,key(1).target,key(2).target,t),
        catmull_rom(key(-1).half_fov_degrees,key(0).half_fov_degrees,key(1).half_fov_degrees,
            key(2).half_fov_degrees,t)};
    if(scene_mode=="aperture-canyon"){
        pose.target=pose.target*.32+Vec3{-.35,1.35,-7.60}*.68;pose.half_fov_degrees=24.5;
    }else if(scene_mode=="mirror-relay"){
        pose.target=pose.target*.30+Vec3{0,1.60,-8.10}*.70;pose.half_fov_degrees=25.0;
    }else if(scene_mode=="occlusion-garden"){
        pose.target=pose.target*.25+Vec3{-.10,.82,-8.05}*.75;pose.half_fov_degrees=25.5;
    }else if(scene_mode!="standard")throw std::runtime_error("unknown scene mode: "+scene_mode);
    return pose;}

std::vector<double> journey_arc_parameters(int frame_count,const std::string& scene_mode="standard"){
    constexpr int samples=8192;std::array<double,samples+1> length{};
    Vec3 previous=journey_pose(0,scene_mode).position;for(int i=1;i<=samples;++i){const Vec3 current=journey_pose(double(i)/samples,scene_mode).position;
        length[i]=length[i-1]+norm(current-previous);previous=current;}
    std::vector<double> parameter(frame_count);for(int frame=0;frame<frame_count;++frame){const double target=
            length.back()*frame/frame_count;const auto upper=std::lower_bound(length.begin(),length.end(),target);
        const int hi=static_cast<int>(upper-length.begin()),lo=std::max(0,hi-1);const double span=length[hi]-length[lo];
        const double fraction=span>1e-12?(target-length[lo])/span:0;parameter[frame]=(lo+fraction)/samples;}
    return parameter;}

Ray camera_ray(const Camera& c,int width,int height,double x,double y){const double px=(2*(x+.5)/width-1)*c.aspect*c.scale;
    const double py=(1-2*(y+.5)/height)*c.scale;return {c.origin,unit(c.forward+c.right*px+c.up*py)};}

bool append_visible_path_signature(const TraceContext& ctx,const Ray& primary_ray,const Hit& primary_hit,
    int channel,int tag_base,std::uint64_t& signature,std::uint64_t* paired_signature,int paired_tag_base){
    auto push=[&](int primitive,int tag){signature_push(&signature,primitive,tag);
        if(paired_signature)signature_push(paired_signature,primitive,tag-tag_base+paired_tag_base);};
    bool dielectric=false;Ray ray=primary_ray;Hit hit=primary_hit;for(int depth=0;depth<8;++depth){
        const Material& material=ctx.scene.materials[ctx.scene.primitives[hit.primitive].material];Vec3 direction{};
        if(material.kind==MaterialKind::Mirror||material.kind==MaterialKind::Metal||material.kind==MaterialKind::Glossy)
            direction=unit(reflect(ray.direction,hit.normal));
        else if(material.kind==MaterialKind::Dielectric){dielectric=true;const Vec3 reflected=
                unit(reflect(ray.direction,hit.normal));const Hit reflected_hit=first_hit(ctx.scene,
                {hit.position+reflected*3e-4,reflected},std::numeric_limits<double>::infinity(),hit.primitive);
            if(reflected_hit.valid)push(reflected_hit.primitive,tag_base+96+32*channel+depth);
            if(material.thin)direction=ray.direction;else{
            const double ni=hit.front?1:material.ior_rgb[channel],nt=hit.front?material.ior_rgb[channel]:1;
            if(!refract(ray.direction,hit.normal,ni/nt,direction))direction=unit(reflect(ray.direction,hit.normal));}}
        else break;ray={hit.position+direction*3e-4,direction};
        hit=first_hit(ctx.scene,ray,std::numeric_limits<double>::infinity(),hit.primitive);if(!hit.valid)break;
        push(hit.primitive,tag_base+32*channel+depth);}
    return dielectric;
}

std::uint64_t terminal_topology_signature(const TraceContext& ctx,const Ray& primary_ray,const Hit& primary_hit,
    std::uint64_t* visible_signature=nullptr){
    if(!primary_hit.valid)return 0;std::uint64_t signature=0;signature_push(&signature,primary_hit.primitive,-11);
    std::uint64_t visible=0;if(visible_signature)signature_push(&visible,primary_hit.primitive,-41);
    // Area-light visibility already lives in field.direct_atlas. Recasting a
    // 3x3 emitter stencil from every camera sample merely rediscovers that
    // compiled field; the radiance certificate below the ownership pass is
    // the correct test for whether it can be interpolated. Keep only topology
    // absent from the diffuse atlas: caustics and specular/dielectric paths.
    if(primary_hit.primitive>=0){if(ctx.beams.oriented){for(int index:ctx.beams.oriented_by_primitive[primary_hit.primitive]){
            const OrientedBeamSheet& sheet=ctx.beams.oriented_sheets[index];const Vec3 offset=primary_hit.position-sheet.position;
            const double x=dot(offset,sheet.tangent_u),y=dot(offset,sheet.tangent_v);const auto& inverse=
                sheet.inverse_position_covariance;const double q=.5*(inverse[0]*x*x+(inverse[1]+inverse[2])*x*y+inverse[3]*y*y);
            if(q<=12)signature_push(&signature,static_cast<int>(ctx.scene.primitives.size())+index,-29);}}
        else for(int index:ctx.beams.by_primitive[primary_hit.primitive]){const CausticDeposit& deposit=ctx.beams.deposits[index];
            if(norm2(primary_hit.position-deposit.position)<=24*deposit.radius*deposit.radius)
                signature_push(&signature,static_cast<int>(ctx.scene.primitives.size())+index,-23);}}
    const bool dielectric=append_visible_path_signature(ctx,primary_ray,primary_hit,1,31,signature,
        visible_signature?&visible:nullptr,47);
    if(dielectric){append_visible_path_signature(ctx,primary_ray,primary_hit,0,31,signature,
            visible_signature?&visible:nullptr,47);
        append_visible_path_signature(ctx,primary_ray,primary_hit,2,31,signature,
            visible_signature?&visible:nullptr,47);}
    if(visible_signature)*visible_signature=visible;return signature;
}

std::uint64_t visible_path_signature(const TraceContext& ctx,const Ray& primary_ray,const Hit& primary_hit){
    if(!primary_hit.valid)return 0;std::uint64_t signature=0;signature_push(&signature,primary_hit.primitive,-41);
    const bool dielectric=append_visible_path_signature(ctx,primary_ray,primary_hit,1,47,signature,nullptr,0);
    if(dielectric){append_visible_path_signature(ctx,primary_ray,primary_hit,0,47,signature,nullptr,0);
        append_visible_path_signature(ctx,primary_ray,primary_hit,2,47,signature,nullptr,0);}
    return signature;
}

struct TerminalLabel{int primitive=-1;std::uint64_t signature=0;};
bool operator==(const TerminalLabel& a,const TerminalLabel& b){return a.primitive==b.primitive&&a.signature==b.signature;}
struct RenderStats{std::uint64_t exact_samples=0,topology_queries=0,topology_runs=0,certificate_samples=0,subdivisions=0,exact_leaves=0,interpolated=0,exact_pixels=0;
    std::uint64_t conv2d_regions=0,conv2d_accepted=0,boundary_pixels=0,boundary_samples=0,
        boundary_topology_samples=0,boundary_radiance_samples=0,primary_packet_rays=0;
    std::uint64_t edge_candidate_segments=0,visible_edge_segments=0,edge_refinements=0;
    std::uint64_t prism_top_edge_pixels=0,prism_top_mixed_pixels=0,analytic_edge_pixels=0,filtered_edge_pixels=0;
    std::uint64_t camera_specular_samples=0,camera_specular_quadrature_samples=0;
    TraceStats trace{};double camera_specular_build_ms=0,adaptive_ms=0,edge_discovery_ms=0,boundary_reconstruction_ms=0,
        raster_ms=0;bool viewer_origin_field_reused=false;};

std::vector<std::uint8_t> render_exact(const TraceContext& context,int width,int height,RenderStats& stats,
    const Camera* camera_override=nullptr){
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width)*height*3);const Camera camera=
        camera_override?*camera_override:make_camera(width,height);
    const unsigned workers=std::max(1u,std::thread::hardware_concurrency());std::atomic<int> next_row{0};std::vector<std::thread> threads;
    std::vector<TraceStats> worker_stats(workers);std::vector<std::uint64_t> exact_counts(workers);const auto start=Clock::now();
    for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&,worker]{TraceContext ctx=context;
        ctx.stats=&worker_stats[worker];ctx.specular_memo=nullptr;
        for(;;){const int y=next_row.fetch_add(1);if(y>=height)break;for(int x=0;x<width;++x){const Ray ray=camera_ray(camera,width,height,x,y);
            const Hit hit=first_hit(ctx.scene,ray);const RGB value=trace_primary(ctx,ray,hit,nullptr);++exact_counts[worker];
            const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;for(int c=0;c<3;++c)image[offset+c]=tone_byte(value[c]);}}});
    for(auto& thread:threads)thread.join();for(unsigned i=0;i<workers;++i){stats.exact_samples+=exact_counts[i];const auto& s=worker_stats[i];
        stats.trace.primary+=s.primary;stats.trace.secondary+=s.secondary;
        stats.trace.shadow+=s.shadow;stats.trace.mirror+=s.mirror;stats.trace.metal+=s.metal;stats.trace.dielectric+=s.dielectric;
        stats.trace.specular_caustic+=s.specular_caustic;stats.trace.prism_bottom_events+=s.prism_bottom_events;
        stats.trace.prism_top_events+=s.prism_top_events;stats.trace.direct_atlas_gathers+=s.direct_atlas_gathers;
        stats.trace.direct_exact_calls+=s.direct_exact_calls;stats.trace.specular_area_calls+=s.specular_area_calls;
        stats.trace.specular_memo_hits+=s.specular_memo_hits;
        stats.trace.surface_radiance_memo_hits+=s.surface_radiance_memo_hits;
        stats.trace.surface_radiance_memo_stores+=s.surface_radiance_memo_stores;
        stats.trace.primary_specular_area_calls+=s.primary_specular_area_calls;
        stats.trace.secondary_specular_area_calls+=s.secondary_specular_area_calls;
        stats.trace.camera_specular_gathers+=s.camera_specular_gathers;
        stats.trace.secondary_specular_weight_lt_1e4+=s.secondary_specular_weight_lt_1e4;
        stats.trace.secondary_specular_weight_lt_1e3+=s.secondary_specular_weight_lt_1e3;
        stats.trace.secondary_specular_weight_lt_1e2+=s.secondary_specular_weight_lt_1e2;
        stats.trace.emitter_rows+=s.emitter_rows;
        stats.trace.emitter_intervals+=s.emitter_intervals;stats.trace.emitter_quadrature_samples+=s.emitter_quadrature_samples;
        stats.trace.feedback_loops+=s.feedback_loops;stats.trace.feedback_returns+=s.feedback_returns;
        stats.trace.sealed_feedback_tails+=s.sealed_feedback_tails;stats.trace.maximum_optical_packets=
            std::max(stats.trace.maximum_optical_packets,s.maximum_optical_packets);
        stats.trace.sealed_residual_weight+=s.sealed_residual_weight;}
    stats.raster_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return image;
}

std::vector<std::uint8_t> render_adaptive(const TraceContext& context,int width,int height,int terminal_error,RenderStats& stats,
    std::vector<TerminalLabel>* ownership_field=nullptr,const Camera* camera_override=nullptr){
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width)*height*3);const Camera camera=
        camera_override?*camera_override:make_camera(width,height);
    if(ownership_field)ownership_field->assign(static_cast<std::size_t>(width)*height,{});
    const unsigned workers=std::max(1u,std::thread::hardware_concurrency());std::atomic<int> next_row{0};std::vector<std::thread> threads;
    struct Counts{std::uint64_t exact=0,topology=0,runs=0,certificate=0,split=0,leaves=0,interpolated=0,exact_pixels=0,packet_rays=0;};
    std::vector<Counts> counts(workers);std::vector<TraceStats> worker_stats(workers);const auto start=Clock::now();
    for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&,worker]{TraceContext ctx=context;
        ctx.stats=&worker_stats[worker];ctx.specular_memo=nullptr;
        std::vector<Ray> rays(width);std::vector<Hit> hits(width);std::vector<RGB> exact(width);
        std::vector<double> direction_x(width),direction_y(width),direction_z(width),best_t(width);
        std::vector<int> best_id(width);
        std::vector<std::array<std::uint8_t,3>> exact_bytes(width);std::vector<TerminalLabel> labels(width),ownership(width);
        std::vector<std::uint8_t> known(width);std::vector<float> synthesized;
        for(;;){const int y=next_row.fetch_add(1);if(y>=height)break;std::fill(known.begin(),known.end(),0);
            for(int x=0;x<width;++x){rays[x]=camera_ray(camera,width,height,x,y);direction_x[x]=rays[x].direction.x;
                direction_y[x]=rays[x].direction.y;direction_z[x]=rays[x].direction.z;}
            first_hit_camera_packet(ctx.scene,camera.origin,rays.data(),direction_x.data(),direction_y.data(),direction_z.data(),
                hits.data(),best_t.data(),best_id.data(),width);counts[worker].packet_rays+=width;
            for(int x=0;x<width;++x){
                std::uint64_t visible_signature=0;if(hits[x].valid){ownership[x]={hits[x].primitive,
                    terminal_topology_signature(ctx,rays[x],hits[x],ownership_field?&visible_signature:nullptr)};++counts[worker].topology;}
                else ownership[x]={-1,0};if(ownership_field)(*ownership_field)[static_cast<std::size_t>(y)*width+x]=hits[x].valid?
                    TerminalLabel{hits[x].primitive,visible_signature}:TerminalLabel{-1,0};}
            auto evaluate=[&](int x)->const RGB&{if(known[x])return exact[x];known[x]=1;++counts[worker].exact;
                if(!hits[x].valid){labels[x]={-1,0};exact[x]={};return exact[x];}std::uint64_t signature=0;
                exact[x]=trace_primary(ctx,rays[x],hits[x],&signature);labels[x]={hits[x].primitive,signature^ownership[x].signature};
                for(int c=0;c<3;++c)exact_bytes[x][c]=tone_byte(exact[x][c]);return exact[x];};
            int begin=0;while(begin<width){const int primitive=ownership[begin].primitive;int end=begin;
                while(end+1<width&&ownership[end+1]==ownership[begin])++end;++counts[worker].runs;
                if(primitive<0){begin=end+1;continue;}
                auto exact_segment=[&](int a,int b){++counts[worker].leaves;for(int x=a;x<=b;++x){evaluate(x);
                    const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;for(int c=0;c<3;++c)image[offset+c]=exact_bytes[x][c];
                    ++counts[worker].exact_pixels;}};
                auto approximate=[&](auto&& self,int a,int b)->void{const int length=b-a+1;if(length<5){exact_segment(a,b);return;}
                    constexpr std::array<double,5> controls{{0,.25,.5,.75,1}};
                    constexpr std::array<double,12> probes{{1./16,2./16,3./16,5./16,6./16,7./16,9./16,10./16,11./16,13./16,14./16,15./16}};
                    std::array<int,5> sites{};std::array<float,15> source{};for(int i=0;i<5;++i){sites[i]=std::clamp(
                        static_cast<int>(std::lround(a+controls[i]*(b-a))),a,b);const RGB& value=evaluate(sites[i]);
                        for(int c=0;c<3;++c)source[i*3+c]=static_cast<float>(value[c]);}
                    bool reject=false;const TerminalLabel label=labels[sites[0]];for(int i=1;i<5;++i)reject=reject||!(labels[sites[i]]==label);
                    std::array<int,12> probe{};std::array<float,12> positions{};std::array<float,36> predicted{};
                    for(int i=0;!reject&&i<12;++i){probe[i]=std::clamp(static_cast<int>(std::lround(a+probes[i]*(b-a))),a,b);
                        ++counts[worker].certificate;evaluate(probe[i]);if(!(labels[probe[i]]==label)){reject=true;break;}
                        positions[i]=static_cast<float>(4.0*(probe[i]-a)/(b-a));}
                    std::array<float,60> prepared{};if(!reject&&conv_prepare_profile_f32(source.data(),5,3,prepared.data())!=0)
                        throw std::runtime_error("CONV profile preparation failed");
                    if(!reject&&conv_evaluate_prepared_profile_f32(source.data(),prepared.data(),5,3,
                            positions.data(),predicted.data(),12)!=0)throw std::runtime_error("CONV certificate failed");
                    const int tolerance=std::max(0,terminal_error-1);for(int i=0;!reject&&i<12;++i)for(int c=0;c<3;++c)
                        if(std::abs(int(exact_bytes[probe[i]][c])-int(tone_byte(predicted[i*3+c])))>tolerance){reject=true;break;}
                    if(reject){++counts[worker].split;const int middle=(a+b)/2;if(middle<=a||middle>=b){exact_segment(a,b);return;}
                        self(self,a,middle);self(self,middle+1,b);return;}
                    synthesized.resize(static_cast<std::size_t>(length)*3);if(conv_resize_prepared_lines_f32(
                            source.data(),prepared.data(),5,3,synthesized.data(),length)!=0)
                        throw std::runtime_error("CONV regional synthesis failed");
                    for(int x=a;x<=b;++x){const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;bool control=false;
                        for(int site:sites)control=control||x==site;if(control){evaluate(x);
                            for(int c=0;c<3;++c)image[offset+c]=exact_bytes[x][c];++counts[worker].exact_pixels;}
                        else{const std::size_t local=static_cast<std::size_t>(x-a)*3;for(int c=0;c<3;++c)image[offset+c]=tone_byte(synthesized[local+c]);
                            ++counts[worker].interpolated;}}};
                approximate(approximate,begin,end);begin=end+1;}}
    });
    for(auto& thread:threads)thread.join();for(unsigned i=0;i<workers;++i){stats.exact_samples+=counts[i].exact;
        stats.topology_queries+=counts[i].topology;stats.topology_runs+=counts[i].runs;stats.primary_packet_rays+=counts[i].packet_rays;
        stats.certificate_samples+=counts[i].certificate;stats.subdivisions+=counts[i].split;stats.exact_leaves+=counts[i].leaves;
        stats.interpolated+=counts[i].interpolated;stats.exact_pixels+=counts[i].exact_pixels;const auto& s=worker_stats[i];
        stats.trace.primary+=s.primary;stats.trace.secondary+=s.secondary;stats.trace.shadow+=s.shadow;stats.trace.mirror+=s.mirror;
        stats.trace.metal+=s.metal;stats.trace.dielectric+=s.dielectric;stats.trace.specular_caustic+=s.specular_caustic;
        stats.trace.prism_bottom_events+=s.prism_bottom_events;stats.trace.prism_top_events+=s.prism_top_events;
        stats.trace.direct_atlas_gathers+=s.direct_atlas_gathers;stats.trace.direct_exact_calls+=s.direct_exact_calls;
        stats.trace.specular_area_calls+=s.specular_area_calls;
        stats.trace.specular_memo_hits+=s.specular_memo_hits;
        stats.trace.surface_radiance_memo_hits+=s.surface_radiance_memo_hits;
        stats.trace.surface_radiance_memo_stores+=s.surface_radiance_memo_stores;
        stats.trace.primary_specular_area_calls+=s.primary_specular_area_calls;
        stats.trace.secondary_specular_area_calls+=s.secondary_specular_area_calls;
        stats.trace.camera_specular_gathers+=s.camera_specular_gathers;
        stats.trace.secondary_specular_weight_lt_1e4+=s.secondary_specular_weight_lt_1e4;
        stats.trace.secondary_specular_weight_lt_1e3+=s.secondary_specular_weight_lt_1e3;
        stats.trace.secondary_specular_weight_lt_1e2+=s.secondary_specular_weight_lt_1e2;
        stats.trace.emitter_rows+=s.emitter_rows;stats.trace.emitter_intervals+=s.emitter_intervals;
        stats.trace.emitter_quadrature_samples+=s.emitter_quadrature_samples;
        stats.trace.feedback_loops+=s.feedback_loops;stats.trace.feedback_returns+=s.feedback_returns;
        stats.trace.sealed_feedback_tails+=s.sealed_feedback_tails;stats.trace.maximum_optical_packets=
            std::max(stats.trace.maximum_optical_packets,s.maximum_optical_packets);
        stats.trace.sealed_residual_weight+=s.sealed_residual_weight;}
    stats.raster_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return image;
}

void accumulate_trace_stats(TraceStats& total,const TraceStats& value){
    total.primary+=value.primary;total.secondary+=value.secondary;total.shadow+=value.shadow;
    total.mirror+=value.mirror;total.metal+=value.metal;total.dielectric+=value.dielectric;
    total.specular_caustic+=value.specular_caustic;total.prism_bottom_events+=value.prism_bottom_events;
    total.prism_top_events+=value.prism_top_events;total.emitter_rows+=value.emitter_rows;
    total.direct_atlas_gathers+=value.direct_atlas_gathers;total.direct_exact_calls+=value.direct_exact_calls;
    total.specular_area_calls+=value.specular_area_calls;total.specular_memo_hits+=value.specular_memo_hits;
    total.surface_radiance_memo_hits+=value.surface_radiance_memo_hits;
    total.surface_radiance_memo_stores+=value.surface_radiance_memo_stores;
    total.primary_specular_area_calls+=value.primary_specular_area_calls;
    total.secondary_specular_area_calls+=value.secondary_specular_area_calls;
    total.camera_specular_gathers+=value.camera_specular_gathers;
    total.secondary_specular_weight_lt_1e4+=value.secondary_specular_weight_lt_1e4;
    total.secondary_specular_weight_lt_1e3+=value.secondary_specular_weight_lt_1e3;
    total.secondary_specular_weight_lt_1e2+=value.secondary_specular_weight_lt_1e2;
    total.emitter_intervals+=value.emitter_intervals;
    total.emitter_quadrature_samples+=value.emitter_quadrature_samples;
    total.feedback_loops+=value.feedback_loops;total.feedback_returns+=value.feedback_returns;
    total.sealed_feedback_tails+=value.sealed_feedback_tails;total.maximum_optical_packets=
        std::max(total.maximum_optical_packets,value.maximum_optical_packets);
    total.sealed_residual_weight+=value.sealed_residual_weight;
}

bool project_to_sensor(const Camera& camera,int width,int height,Vec3 point,Vec2& sensor){
    const Vec3 q=point-camera.origin;const double forward=dot(q,camera.forward);if(forward<=1e-8)return false;
    const double nx=dot(q,camera.right)/(forward*camera.aspect*camera.scale);
    const double ny=dot(q,camera.up)/(forward*camera.scale);
    sensor={(nx+1)*.5*width-.5,(1-ny)*.5*height-.5};return std::isfinite(sensor.x)&&std::isfinite(sensor.y);
}

double maximum_plane_value(const Bounds3& bounds,Vec3 origin,Vec3 normal){const Vec3 point{
    normal.x>=0?bounds.upper.x:bounds.lower.x,normal.y>=0?bounds.upper.y:bounds.lower.y,
    normal.z>=0?bounds.upper.z:bounds.lower.z};return dot(point-origin,normal);}

bool bounds_may_reach_camera(const Bounds3& bounds,const Camera& camera){const double horizontal=camera.aspect*camera.scale;
    const std::array<Vec3,5> inward{{camera.forward,camera.forward*horizontal+camera.right,
        camera.forward*horizontal-camera.right,camera.forward*camera.scale+camera.up,
        camera.forward*camera.scale-camera.up}};
    for(Vec3 plane:inward)if(maximum_plane_value(bounds,camera.origin,plane)<-1e-8)return false;return true;}

void query_camera_primitives(const Scene& scene,const Camera& camera,std::vector<int>& candidates){candidates.clear();
    if(!scene.use_bvh||scene.bvh_nodes.empty()){candidates.resize(scene.primitives.size());
        for(int i=0;i<static_cast<int>(scene.primitives.size());++i)candidates[i]=i;return;}
    std::array<int,128> stack{};int size=0;stack[size++]=0;while(size){const BvhNode& node=scene.bvh_nodes[stack[--size]];
        if(!bounds_may_reach_camera(node.bounds,camera))continue;if(node.count){for(int i=0;i<node.count;++i)
                candidates.push_back(scene.bvh_primitives[node.begin+i]);}
        else{stack[size++]=node.left;stack[size++]=node.right;}}
}

struct ProjectedEdgeLine{double a=0,b=0,c=0;std::uint8_t count=0;};

struct VisibleEdgeField{
    // bit 0: primary, bit 1: topology, bit 2: shallow line,
    // bit 3: topology reconstruction, bits 4/5: x/y topology crossings
    std::vector<std::uint8_t> kind;
    std::vector<ProjectedEdgeLine> straight;
    std::uint64_t candidate_segments=0,visible_segments=0,label_queries=0,refinements=0;
};

struct PixelRegion{double area=0;Vec2 centroid{};};

PixelRegion clipped_pixel_region(int px,int py,const ProjectedEdgeLine& line,bool positive){
    std::vector<Vec2> polygon{{px-.5,py-.5},{px+.5,py-.5},{px+.5,py+.5},{px-.5,py+.5}},clipped;
    auto value=[&](Vec2 point){const double v=line.a*point.x+line.b*point.y+line.c;return positive?v:-v;};
    for(std::size_t i=0;i<polygon.size();++i){const Vec2 current=polygon[i],next=polygon[(i+1)%polygon.size()];
        const double fc=value(current),fn=value(next);const bool inside_current=fc>=0,inside_next=fn>=0;
        if(inside_current)clipped.push_back(current);if(inside_current!=inside_next){const double t=fc/(fc-fn);
            clipped.push_back({current.x+(next.x-current.x)*t,current.y+(next.y-current.y)*t});}}
    if(clipped.size()<3)return {};double twice_area=0,cx=0,cy=0;
    for(std::size_t i=0;i<clipped.size();++i){const Vec2 p=clipped[i],q=clipped[(i+1)%clipped.size()];
        const double cross_value=p.x*q.y-q.x*p.y;twice_area+=cross_value;cx+=(p.x+q.x)*cross_value;
        cy+=(p.y+q.y)*cross_value;}
    if(std::abs(twice_area)<1e-14)return {};PixelRegion result;result.area=std::abs(twice_area)*.5;
    result.centroid={cx/(3*twice_area),cy/(3*twice_area)};return result;
}

VisibleEdgeField discover_visible_edges(const TraceContext& context,const Camera& camera,int width,int height,
    const std::vector<TerminalLabel>& ownership){
    VisibleEdgeField field;field.kind.assign(static_cast<std::size_t>(width)*height,0);
    field.straight.resize(static_cast<std::size_t>(width)*height);
    std::vector<int> visible_curved_owner(static_cast<std::size_t>(width)*height,-1);
    auto mark=[&](double x,double y,std::uint8_t kind){const int px=static_cast<int>(std::floor(x+.5));
        const int py=static_cast<int>(std::floor(y+.5));if(px>=0&&px<width&&py>=0&&py<height)
            field.kind[static_cast<std::size_t>(py)*width+px]|=kind;};
    auto mark_straight=[&](double x,double y,Vec2 a,Vec2 b){const int px=static_cast<int>(std::floor(x+.5));
        const int py=static_cast<int>(std::floor(y+.5));if(px<0||px>=width||py<0||py>=height)return;
        const double dx=b.x-a.x,dy=b.y-a.y,length=std::hypot(dx,dy);if(length<1e-12)return;
        ProjectedEdgeLine candidate{-dy/length,dx/length,0,1};candidate.c=-(candidate.a*a.x+candidate.b*a.y);
        if(candidate.a<0||(std::abs(candidate.a)<1e-14&&candidate.b<0)){
            candidate.a=-candidate.a;candidate.b=-candidate.b;candidate.c=-candidate.c;}
        ProjectedEdgeLine& stored=field.straight[static_cast<std::size_t>(py)*width+px];
        if(stored.count==0)stored=candidate;else if(stored.count==1&&(std::abs(stored.a-candidate.a)>1e-7||
                std::abs(stored.b-candidate.b)>1e-7||std::abs(stored.c-candidate.c)>1e-4))stored.count=2;};
    auto label_at=[&](double x,double y,bool topology){++field.label_queries;const Ray ray=camera_ray(camera,width,height,x,y);
        const Hit hit=first_hit(context.scene,ray);if(!hit.valid)return TerminalLabel{-1,0};
        return TerminalLabel{hit.primitive,topology?visible_path_signature(context,ray,hit):0};};
    auto refine=[&](Vec2 a,Vec2 b,TerminalLabel left,TerminalLabel right,bool topology){
        // This result is consumed only by mark(), which rounds it back to one
        // of the two adjacent sensor pixels.  One midpoint classification
        // determines that rounded basin; deeper bisection was discarded.
        for(int iteration=0;iteration<1;++iteration){const Vec2 middle{(a.x+b.x)*.5,(a.y+b.y)*.5};
            const TerminalLabel value=label_at(middle.x,middle.y,topology);++field.refinements;
            if(value==left)a=middle;else{b=middle;right=value;}}
        (void)right;return Vec2{(a.x+b.x)*.5,(a.y+b.y)*.5};};

    // Every sampled ownership transition marks the adjacent pixel basin that
    // contains its first change. This includes reflected/refracted terminal
    // changes already present at sensor resolution, rather than only
    // first-surface silhouettes. Analytic contours carry subpixel coverage.
    for(int y=0;y<height;++y)for(int x=0;x+1<width;++x){const TerminalLabel a=ownership[static_cast<std::size_t>(y)*width+x];
        const TerminalLabel b=ownership[static_cast<std::size_t>(y)*width+x+1];if(a==b)continue;
        const bool topology=a.primitive==b.primitive;const TerminalLabel la=topology?a:TerminalLabel{a.primitive,0};
        const TerminalLabel lb=topology?b:TerminalLabel{b.primitive,0};const Vec2 crossing=refine({double(x),double(y)},
            {double(x+1),double(y)},la,lb,topology);mark(crossing.x,crossing.y,topology?18:1);}
    for(int y=0;y+1<height;++y)for(int x=0;x<width;++x){const TerminalLabel a=ownership[static_cast<std::size_t>(y)*width+x];
        const TerminalLabel b=ownership[static_cast<std::size_t>(y+1)*width+x];if(a==b)continue;
        const bool topology=a.primitive==b.primitive;const TerminalLabel la=topology?a:TerminalLabel{a.primitive,0};
        const TerminalLabel lb=topology?b:TerminalLabel{b.primitive,0};const Vec2 crossing=refine({double(x),double(y)},
            {double(x),double(y+1)},la,lb,topology);mark(crossing.x,crossing.y,topology?34:1);}

    // Analytic primitive contours catch visible geometry thinner than a sensor
    // sample.  Straight world edges remain straight under pinhole projection;
    // sphere contours are exact camera-tangent circles sampled below 1/8 pixel.
    auto clip_to_sensor=[&](Vec2& a,Vec2& b){constexpr double margin=1;const double xmin=-margin,xmax=width-1+margin;
        const double ymin=-margin,ymax=height-1+margin;auto code=[&](Vec2 p){return (p.x<xmin?1:0)|(p.x>xmax?2:0)|
            (p.y<ymin?4:0)|(p.y>ymax?8:0);};int ca=code(a),cb=code(b);for(int iteration=0;iteration<8;++iteration){
            if(!(ca|cb))return true;if(ca&cb)return false;const int outside=ca?ca:cb;Vec2 q{};
            if(outside&4){if(std::abs(b.y-a.y)<1e-15)return false;q={a.x+(b.x-a.x)*(ymin-a.y)/(b.y-a.y),ymin};}
            else if(outside&8){if(std::abs(b.y-a.y)<1e-15)return false;q={a.x+(b.x-a.x)*(ymax-a.y)/(b.y-a.y),ymax};}
            else if(outside&2){if(std::abs(b.x-a.x)<1e-15)return false;q={xmax,a.y+(b.y-a.y)*(xmax-a.x)/(b.x-a.x)};}
            else{if(std::abs(b.x-a.x)<1e-15)return false;q={xmin,a.y+(b.y-a.y)*(xmin-a.x)/(b.x-a.x)};}
            if(outside==ca){a=q;ca=code(a);}else{b=q;cb=code(b);}}return false;};
    auto visible_segment=[&](int primitive,Vec2 a,Vec2 b,bool straight=false){if(!clip_to_sensor(a,b))return;
        const double dx=b.x-a.x,dy=b.y-a.y,length=std::hypot(dx,dy);
        if(length<1e-9)return;const int pieces=std::clamp(static_cast<int>(std::ceil(length*8)),1,200000);
        const double nx=-dy/length,ny=dx/length;int last_visible_pixel=-1;
        for(int piece=0;piece<pieces;++piece){++field.candidate_segments;const double t=(piece+.5)/pieces;
            const double x=a.x+dx*t,y=a.y+dy*t;const int px=static_cast<int>(std::floor(x+.5));
            const int py=static_cast<int>(std::floor(y+.5));const int pixel=px>=0&&px<width&&py>=0&&py<height?py*width+px:-1;
            if(pixel>=0&&(straight?pixel==last_visible_pixel:visible_curved_owner[pixel]==primitive))continue;
            const TerminalLabel left=label_at(x+nx*.04,y+ny*.04,false);
            const TerminalLabel right=label_at(x-nx*.04,y-ny*.04,false);
            if(left.primitive!=right.primitive&&(left.primitive==primitive||right.primitive==primitive)){
                ++field.visible_segments;mark(x,y,1);if(straight){mark_straight(x,y,a,b);last_visible_pixel=pixel;}
                else if(pixel>=0)visible_curved_owner[pixel]=primitive;}}};
    std::vector<int> camera_primitives;query_camera_primitives(context.scene,camera,camera_primitives);
    for(int id:camera_primitives){const Primitive& primitive=context.scene.primitives[id];
        if(primitive.shape==Shape::Sphere){const Vec3 axis=primitive.center-camera.origin;const double distance=norm(axis);
            if(distance<=primitive.radius)continue;const Vec3 direction=axis/distance;
            Vec3 tangent_u=unit(cross(direction,std::abs(direction.y)<.9?Vec3{0,1,0}:Vec3{1,0,0}));
            const Vec3 tangent_v=cross(direction,tangent_u);const double centre_distance=distance-primitive.radius*primitive.radius/distance;
            const double radius=primitive.radius*std::sqrt(distance*distance-primitive.radius*primitive.radius)/distance;
            const Vec3 circle= camera.origin+direction*centre_distance;Vec2 centre_screen,edge_screen;
            if(!project_to_sensor(camera,width,height,circle,centre_screen)||
                !project_to_sensor(camera,width,height,circle+tangent_u*radius,edge_screen))continue;
            const double radius_pixels=std::max(1.0,std::hypot(edge_screen.x-centre_screen.x,edge_screen.y-centre_screen.y));
            const int pieces=std::clamp(static_cast<int>(std::ceil(2*pi*radius_pixels*8)),64,20000);Vec2 previous{};
            for(int piece=0;piece<=pieces;++piece){const double angle=2*pi*(piece%pieces)/pieces;Vec2 current;
                if(!project_to_sensor(camera,width,height,circle+(tangent_u*std::cos(angle)+tangent_v*std::sin(angle))*radius,current))break;
                if(piece>0)visible_segment(id,previous,current);previous=current;}}
        else{const std::array<Vec3,4> vertex=primitive.shape==Shape::Rectangle?
                std::array<Vec3,4>{{primitive.origin,primitive.origin+primitive.u,primitive.origin+primitive.u+primitive.v,primitive.origin+primitive.v}}:
                std::array<Vec3,4>{{primitive.a,primitive.b,primitive.c,primitive.a}};
            const int count=primitive.shape==Shape::Rectangle?4:3;for(int edge=0;edge<count;++edge){Vec2 a,b;
                if(project_to_sensor(camera,width,height,vertex[edge],a)&&project_to_sensor(camera,width,height,vertex[(edge+1)%count],b))
                    visible_segment(id,a,b,true);}}
    }

    // Box reconstruction can leave coherent stairs on nearly horizontal or
    // vertical high-contrast seams even when their subpixel area is exact.
    // Carry only those straight lines into the immediately adjacent pixels;
    // they will receive a narrow sensor-footprint reconstruction below.
    const std::vector<ProjectedEdgeLine> original_straight=field.straight;
    auto merge_line=[](ProjectedEdgeLine& stored,const ProjectedEdgeLine& candidate){
        if(stored.count==0)stored=candidate;else if(stored.count==1&&(std::abs(stored.a-candidate.a)>1e-7||
                std::abs(stored.b-candidate.b)>1e-7||std::abs(stored.c-candidate.c)>1e-4))stored.count=2;};
    for(int y=0;y<height;++y)for(int x=0;x<width;++x){const ProjectedEdgeLine& line=
            original_straight[static_cast<std::size_t>(y)*width+x];
        const double minor=std::min(std::abs(line.a),std::abs(line.b));
        if(line.count!=1||minor<=.015||minor>=.35)continue;
        for(int oy=-1;oy<=1;++oy)for(int ox=-1;ox<=1;++ox){const int qx=x+ox,qy=y+oy;
            if(qx<0||qx>=width||qy<0||qy>=height)continue;const std::size_t q=static_cast<std::size_t>(qy)*width+qx;
            field.kind[q]|=5;merge_line(field.straight[q],line);}}
    const std::vector<std::uint8_t> original_kind=field.kind;
    for(int y=0;y<height;++y)for(int x=0;x<width;++x){const std::size_t index=static_cast<std::size_t>(y)*width+x;
        const int primary=ownership[index].primitive;if(!(original_kind[index]&2)||primary<0||
            context.scene.materials[context.scene.primitives[primary].material].kind!=MaterialKind::Mirror)continue;
        int x_crossings=0,y_crossings=0;
        for(int oy=-2;oy<=2;++oy)for(int ox=-2;ox<=2;++ox){const int qx=x+ox,qy=y+oy;
            if(qx<0||qx>=width||qy<0||qy>=height)continue;const std::uint8_t nearby=
                original_kind[static_cast<std::size_t>(qy)*width+qx];x_crossings+=(nearby&16)!=0;y_crossings+=(nearby&32)!=0;}
        const int major=std::max(x_crossings,y_crossings),minor=std::min(x_crossings,y_crossings);
        if(major<2||minor*3>major)continue;
        for(int oy=-1;oy<=1;++oy)for(int ox=-1;ox<=1;++ox){const int qx=x+ox,qy=y+oy;
            if(qx<0||qx>=width||qy<0||qy>=height)continue;field.kind[static_cast<std::size_t>(qy)*width+qx]|=10;}}

    return field;
}

std::vector<std::uint8_t> render_visible_edge_field(
    const TraceContext& context,int width,int height,int terminal_error,RenderStats& stats,const Camera* camera_override=nullptr,
    const CameraSpecularField* cached_viewer_origin_field=nullptr){
    const auto start=Clock::now();const Camera camera=camera_override?*camera_override:make_camera(width,height);
    const bool origin_matches=cached_viewer_origin_field&&cached_viewer_origin_field->origin.x==camera.origin.x&&
        cached_viewer_origin_field->origin.y==camera.origin.y&&cached_viewer_origin_field->origin.z==camera.origin.z;
    CameraSpecularField owned_viewer_origin_field;const CameraSpecularField* viewer_origin_field=cached_viewer_origin_field;
    if(!origin_matches){const auto camera_field_start=Clock::now();owned_viewer_origin_field=
            compile_viewer_origin_specular_field(context,camera.origin);stats.camera_specular_build_ms=
            std::chrono::duration<double,std::milli>(Clock::now()-camera_field_start).count();
        viewer_origin_field=&owned_viewer_origin_field;}else stats.viewer_origin_field_reused=true;
    stats.camera_specular_samples=viewer_origin_field->samples;stats.camera_specular_quadrature_samples=
        origin_matches?0:viewer_origin_field->construction.emitter_quadrature_samples;TraceContext camera_context=context;
    camera_context.camera_specular=viewer_origin_field;
    const auto adaptive_start=Clock::now();std::vector<TerminalLabel> ownership;
    std::vector<std::uint8_t> image=render_adaptive(camera_context,width,height,terminal_error,stats,&ownership,&camera);
    stats.adaptive_ms=std::chrono::duration<double,std::milli>(Clock::now()-adaptive_start).count();
    const auto edge_start=Clock::now();const VisibleEdgeField edges=
        discover_visible_edges(camera_context,camera,width,height,ownership);
    stats.edge_discovery_ms=std::chrono::duration<double,std::milli>(Clock::now()-edge_start).count();
    std::vector<int> pixels;
    for(std::size_t index=0;index<edges.kind.size();++index)if(edges.kind[index])pixels.push_back(static_cast<int>(index));
    stats.boundary_pixels=pixels.size();stats.edge_candidate_segments=edges.candidate_segments;
    stats.visible_edge_segments=edges.visible_segments;stats.edge_refinements=edges.refinements;
    stats.topology_queries+=edges.label_queries;

    const auto boundary_start=Clock::now();
    const unsigned workers=std::max(1u,std::thread::hardware_concurrency());std::atomic<std::size_t> next{0};
    std::vector<std::thread> threads;std::vector<TraceStats> trace_stats(workers);
    std::vector<std::uint64_t> label_counts(workers),topology_label_counts(workers),packet_counts(workers),radiance_counts(workers),
        prism_top_counts(workers),prism_mixed_counts(workers),analytic_counts(workers),filtered_counts(workers);
    for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&,worker]{TraceContext ctx=camera_context;
        ctx.stats=&trace_stats[worker];ctx.specular_memo=nullptr;
        struct Bucket{TerminalLabel label{};double weight=0,x=0,y=0,first_x=0,first_y=0;};
        std::array<Bucket,256> buckets{};
        std::array<Ray,256> packet_rays{};std::array<Hit,256> packet_hits{};
        std::array<double,256> packet_x{},packet_y{},packet_weight{},packet_dx{},packet_dy{},packet_dz{},packet_best_t{};
        std::array<int,256> packet_best_id{};
        for(;;){const std::size_t work=next.fetch_add(1);if(work>=pixels.size())break;const int index=pixels[work];
            const int px=index%width,py=index/width;
            const bool topology=(edges.kind[index]&2)!=0;int bucket_count=0;
            auto label_at=[&](double x,double y){const Ray ray=camera_ray(camera,width,height,x,y);
                const Hit hit=first_hit(ctx.scene,ray);TerminalLabel label{-1,0};if(hit.valid){label.primitive=hit.primitive;
                if(topology)label.signature=visible_path_signature(ctx,ray,hit);}++label_counts[worker];
                topology_label_counts[worker]+=topology;return label;};
            auto add_region=[&](TerminalLabel label,double x,double y,double weight){
                int found=-1;for(int bucket=0;bucket<bucket_count;++bucket)if(buckets[bucket].label==label){found=bucket;break;}
                if(found<0){if(bucket_count>=static_cast<int>(buckets.size()))throw std::runtime_error("edge bucket ceiling reached");
                    buckets[bucket_count++]={label,weight,x*weight,y*weight,x,y};}
                else{buckets[found].weight+=weight;buckets[found].x+=x*weight;buckets[found].y+=y*weight;}};
            auto add_packet_regions=[&](int count){first_hit_camera_packet(ctx.scene,camera.origin,packet_rays.data(),
                    packet_dx.data(),packet_dy.data(),packet_dz.data(),packet_hits.data(),packet_best_t.data(),
                    packet_best_id.data(),count);label_counts[worker]+=count;packet_counts[worker]+=count;
                if(topology)topology_label_counts[worker]+=count;
                for(int sample=0;sample<count;++sample){const Hit& hit=packet_hits[sample];TerminalLabel label{-1,0};
                    if(hit.valid){label.primitive=hit.primitive;if(topology)
                            label.signature=visible_path_signature(ctx,packet_rays[sample],hit);}
                    add_region(label,packet_x[sample],packet_y[sample],packet_weight[sample]);}};
            bool analytic=false;const ProjectedEdgeLine& straight=edges.straight[index];
            if(topology&&(edges.kind[index]&8)){constexpr int filter_side=8;double filter_weight=0;
                for(int sy=0;sy<filter_side;++sy)for(int sx=0;sx<filter_side;++sx){const double dx=-1+(sx+.5)*2/filter_side;
                    const double dy=-1+(sy+.5)*2/filter_side,weight=(1-std::abs(dx))*(1-std::abs(dy));
                    const int sample=sy*filter_side+sx;packet_x[sample]=px+dx;packet_y[sample]=py+dy;
                    packet_weight[sample]=weight;packet_rays[sample]=camera_ray(camera,width,height,packet_x[sample],packet_y[sample]);
                    packet_dx[sample]=packet_rays[sample].direction.x;packet_dy[sample]=packet_rays[sample].direction.y;
                    packet_dz[sample]=packet_rays[sample].direction.z;filter_weight+=weight;}
                add_packet_regions(filter_side*filter_side);
                analytic=filter_weight>0;if(analytic)++filtered_counts[worker];}
            if(!analytic&&(edges.kind[index]&4)&&straight.count==1){const double distance=
                    straight.a*px+straight.b*py+straight.c;constexpr double sigma=.42;
                const double positive_weight=.5*(1+std::erf(distance/(std::sqrt(2.0)*sigma)));
                const Vec2 boundary{px-straight.a*distance,py-straight.b*distance};const double offset=.08;
                const Vec2 positive_point=distance>offset?Vec2{double(px),double(py)}:
                    Vec2{boundary.x+straight.a*offset,boundary.y+straight.b*offset};
                const Vec2 negative_point=distance<-offset?Vec2{double(px),double(py)}:
                    Vec2{boundary.x-straight.a*offset,boundary.y-straight.b*offset};
                const TerminalLabel positive_label=label_at(positive_point.x,positive_point.y),negative_label=
                    label_at(negative_point.x,negative_point.y);if(!(positive_label==negative_label)){
                    if(positive_weight>1e-8)add_region(positive_label,positive_point.x,positive_point.y,positive_weight);
                    if(positive_weight<1-1e-8)add_region(negative_label,negative_point.x,negative_point.y,1-positive_weight);
                    analytic=true;++filtered_counts[worker];}}
            if(!analytic&&!topology&&straight.count==1){const PixelRegion positive=clipped_pixel_region(px,py,straight,true);
                const PixelRegion negative=clipped_pixel_region(px,py,straight,false);
                if(positive.area>1e-10&&negative.area>1e-10){const TerminalLabel positive_label=
                        label_at(positive.centroid.x,positive.centroid.y);const TerminalLabel negative_label=
                        label_at(negative.centroid.x,negative.centroid.y);
                    if(!(positive_label==negative_label)){add_region(positive_label,positive.centroid.x,
                            positive.centroid.y,positive.area);add_region(negative_label,negative.centroid.x,
                            negative.centroid.y,negative.area);analytic=true;++analytic_counts[worker];}}}
            constexpr int side=16;if(!analytic){for(int sy=0;sy<side;++sy)for(int sx=0;sx<side;++sx){
                    const int sample=sy*side+sx;const double x=px-.5+(sx+.5)/side,y=py-.5+(sy+.5)/side;
                    packet_x[sample]=x;packet_y[sample]=y;packet_weight[sample]=1;
                    packet_rays[sample]=camera_ray(camera,width,height,x,y);
                    packet_dx[sample]=packet_rays[sample].direction.x;packet_dy[sample]=packet_rays[sample].direction.y;
                    packet_dz[sample]=packet_rays[sample].direction.z;}
                add_packet_regions(side*side);}
            bool has_prism_top=false,has_other_owner=false;for(int bucket_index=0;bucket_index<bucket_count;++bucket_index){
                const Bucket& bucket=buckets[bucket_index];
                has_prism_top=has_prism_top||bucket.label.primitive==ctx.scene.prism_top;
                has_other_owner=has_other_owner||(bucket.label.primitive>=0&&bucket.label.primitive!=ctx.scene.prism_top);}
            if(has_prism_top){++prism_top_counts[worker];if(has_other_owner)++prism_mixed_counts[worker];}
            double total_weight=0;for(int bucket_index=0;bucket_index<bucket_count;++bucket_index)total_weight+=buckets[bucket_index].weight;
            RGB value{};for(int bucket_index=0;bucket_index<bucket_count;++bucket_index){const Bucket& bucket=buckets[bucket_index];
                if(bucket.label.primitive<0)continue;double x=bucket.x/bucket.weight,y=bucket.y/bucket.weight;
                Ray ray=camera_ray(camera,width,height,x,y);Hit hit=first_hit(ctx.scene,ray);TerminalLabel actual{hit.valid?hit.primitive:-1,0};
                if(hit.valid&&topology)actual.signature=visible_path_signature(ctx,ray,hit);if(!(actual==bucket.label)){
                    x=bucket.first_x;y=bucket.first_y;ray=camera_ray(camera,width,height,x,y);hit=first_hit(ctx.scene,ray);}
                value+=trace_primary(ctx,ray,hit,nullptr)*(bucket.weight/total_weight);++radiance_counts[worker];}
            const std::size_t offset=static_cast<std::size_t>(index)*3;for(int c=0;c<3;++c)image[offset+c]=tone_byte(value[c]);}}
    );
    for(auto& thread:threads)thread.join();for(unsigned worker=0;worker<workers;++worker){stats.boundary_samples+=label_counts[worker];
        stats.boundary_topology_samples+=topology_label_counts[worker];
        stats.boundary_radiance_samples+=radiance_counts[worker];
        stats.primary_packet_rays+=packet_counts[worker];
        stats.exact_samples+=radiance_counts[worker];stats.prism_top_edge_pixels+=prism_top_counts[worker];
        stats.prism_top_mixed_pixels+=prism_mixed_counts[worker];
        stats.analytic_edge_pixels+=analytic_counts[worker];
        stats.filtered_edge_pixels+=filtered_counts[worker];
        accumulate_trace_stats(stats.trace,trace_stats[worker]);}
    stats.boundary_reconstruction_ms=std::chrono::duration<double,std::milli>(Clock::now()-boundary_start).count();
    stats.raster_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return image;
}

struct TerminalRect{int x0=0,x1=0,y0=0,y1=0;};
struct TerminalPoint{RGB value{};TerminalLabel label{};};

[[maybe_unused]] std::vector<std::uint8_t> render_adaptive_2d(
    const TraceContext& context,int width,int height,int terminal_error,RenderStats& stats){
    const auto start=Clock::now();const Camera camera=make_camera(width,height);
    const std::size_t pixel_count=static_cast<std::size_t>(width)*height;
    std::vector<TerminalLabel> ownership(pixel_count);
    const unsigned workers=std::max(1u,std::thread::hardware_concurrency());
    std::atomic<int> next_row{0};std::vector<std::thread> topology_threads;
    std::vector<std::uint64_t> topology_counts(workers);
    for(unsigned worker=0;worker<workers;++worker)topology_threads.emplace_back([&,worker]{
        for(;;){const int y=next_row.fetch_add(1);if(y>=height)break;
            for(int x=0;x<width;++x){const Ray ray=camera_ray(camera,width,height,x,y);const Hit hit=first_hit(context.scene,ray);
                const std::size_t index=static_cast<std::size_t>(y)*width+x;
                if(hit.valid){ownership[index]={hit.primitive,terminal_topology_signature(context,ray,hit)};++topology_counts[worker];}
                else ownership[index]={-1,0};}}
    });
    for(auto& thread:topology_threads)thread.join();
    for(std::uint64_t count:topology_counts)stats.topology_queries+=count;

    auto uniform=[&](const TerminalRect& rectangle){const TerminalLabel label=
        ownership[static_cast<std::size_t>(rectangle.y0)*width+rectangle.x0];
        for(int y=rectangle.y0;y<=rectangle.y1;++y)for(int x=rectangle.x0;x<=rectangle.x1;++x)
            if(!(ownership[static_cast<std::size_t>(y)*width+x]==label))return false;
        return true;};
    std::vector<TerminalRect> regions,stack{{0,width-1,0,height-1}};
    while(!stack.empty()){const TerminalRect rectangle=stack.back();stack.pop_back();
        if(uniform(rectangle)){regions.push_back(rectangle);continue;}
        const int rw=rectangle.x1-rectangle.x0+1,rh=rectangle.y1-rectangle.y0+1;
        if(rw>=rh&&rw>1){const int middle=(rectangle.x0+rectangle.x1)/2;
            stack.push_back({rectangle.x0,middle,rectangle.y0,rectangle.y1});
            stack.push_back({middle+1,rectangle.x1,rectangle.y0,rectangle.y1});}
        else if(rh>1){const int middle=(rectangle.y0+rectangle.y1)/2;
            stack.push_back({rectangle.x0,rectangle.x1,rectangle.y0,middle});
            stack.push_back({rectangle.x0,rectangle.x1,middle+1,rectangle.y1});}
        else regions.push_back(rectangle);
    }
    stats.topology_runs=regions.size();stats.conv2d_regions=regions.size();

    std::vector<RGB> linear(pixel_count);std::atomic<std::size_t> next_region{0};
    struct Counts{std::uint64_t exact=0,certificate=0,split=0,leaves=0,interpolated=0,exact_pixels=0,accepted=0;};
    std::vector<Counts> counts(workers);std::vector<TraceStats> worker_stats(workers);std::vector<std::thread> render_threads;
    for(unsigned worker=0;worker<workers;++worker)render_threads.emplace_back([&,worker]{
        TraceContext ctx{context.scene,context.beams,context.field,&worker_stats[worker],context.sealed_optics,context.optical_cutoff};
        auto evaluate=[&](double x,double y){TerminalPoint result;const Ray ray=camera_ray(camera,width,height,x,y);
            const Hit hit=first_hit(ctx.scene,ray);++counts[worker].exact;if(!hit.valid){result.label={-1,0};return result;}
            const std::uint64_t topology=terminal_topology_signature(ctx,ray,hit);std::uint64_t signature=0;
            result.value=trace_primary(ctx,ray,hit,&signature);result.label={hit.primitive,signature^topology};return result;};
        auto exact_rectangle=[&](const TerminalRect& rectangle){++counts[worker].leaves;
            for(int y=rectangle.y0;y<=rectangle.y1;++y)for(int x=rectangle.x0;x<=rectangle.x1;++x){
                linear[static_cast<std::size_t>(y)*width+x]=evaluate(x,y).value;++counts[worker].exact_pixels;}};
        auto approximate=[&](auto&& self,const TerminalRect& rectangle)->void{
            const int rw=rectangle.x1-rectangle.x0+1,rh=rectangle.y1-rectangle.y0+1;
            if(rw<5||rh<5){exact_rectangle(rectangle);return;}
            std::array<float,75> source{};TerminalLabel common{};bool reject=false;
            for(int j=0;j<5;++j)for(int i=0;i<5;++i){const double x=rectangle.x0+.25*i*(rectangle.x1-rectangle.x0);
                const double y=rectangle.y0+.25*j*(rectangle.y1-rectangle.y0);const TerminalPoint sample=evaluate(x,y);
                if(i==0&&j==0)common=sample.label;else reject=reject||!(sample.label==common);
                for(int c=0;c<3;++c)source[(j*5+i)*3+c]=static_cast<float>(sample.value[c]);}
            static constexpr std::array<double,4> witness{{.125,.375,.625,.875}};
            std::array<float,16> qx{},qy{};std::array<float,48> predicted{};
            for(int j=0;j<4;++j)for(int i=0;i<4;++i){const int k=j*4+i;
                qx[k]=static_cast<float>(4*witness[i]);qy[k]=static_cast<float>(4*witness[j]);}
            if(!reject&&conv_evaluate_profile_2d_f32(
                    source.data(),5,5,3,qx.data(),qy.data(),predicted.data(),16)!=0)
                throw std::runtime_error("2-D CONV certificate failed");
            const int tolerance=std::max(0,terminal_error-1);
            for(int j=0;!reject&&j<4;++j)for(int i=0;!reject&&i<4;++i){const int k=j*4+i;
                const double x=rectangle.x0+witness[i]*(rectangle.x1-rectangle.x0);
                const double y=rectangle.y0+witness[j]*(rectangle.y1-rectangle.y0);
                ++counts[worker].certificate;const TerminalPoint exact=evaluate(x,y);
                if(!(exact.label==common)){reject=true;break;}for(int c=0;c<3;++c)
                    if(std::abs(int(tone_byte(exact.value[c]))-int(tone_byte(predicted[k*3+c])))>tolerance){reject=true;break;}}
            if(reject){++counts[worker].split;
                if(rw>=rh&&rw>1){const int middle=(rectangle.x0+rectangle.x1)/2;
                    self(self,{rectangle.x0,middle,rectangle.y0,rectangle.y1});
                    self(self,{middle+1,rectangle.x1,rectangle.y0,rectangle.y1});}
                else if(rh>1){const int middle=(rectangle.y0+rectangle.y1)/2;
                    self(self,{rectangle.x0,rectangle.x1,rectangle.y0,middle});
                    self(self,{rectangle.x0,rectangle.x1,middle+1,rectangle.y1});}
                else exact_rectangle(rectangle);return;}
            const int total=rw*rh;std::vector<float> xq(total),yq(total),output(static_cast<std::size_t>(total)*3);
            for(int y=0;y<rh;++y)for(int x=0;x<rw;++x){const int k=y*rw+x;
                xq[k]=rw==1?0:4.0f*x/(rw-1);yq[k]=rh==1?0:4.0f*y/(rh-1);}
            if(conv_evaluate_profile_2d_f32(source.data(),5,5,3,xq.data(),yq.data(),output.data(),total)!=0)
                throw std::runtime_error("2-D CONV regional synthesis failed");
            for(int y=0;y<rh;++y)for(int x=0;x<rw;++x){const int k=y*rw+x;
                linear[static_cast<std::size_t>(rectangle.y0+y)*width+rectangle.x0+x]=
                    {output[k*3],output[k*3+1],output[k*3+2]};++counts[worker].interpolated;}
            ++counts[worker].accepted;
        };
        for(;;){const std::size_t index=next_region.fetch_add(1);if(index>=regions.size())break;
            const TerminalRect rectangle=regions[index];const TerminalLabel label=
                ownership[static_cast<std::size_t>(rectangle.y0)*width+rectangle.x0];
            if(label.primitive<0)continue;approximate(approximate,rectangle);}
    });
    for(auto& thread:render_threads)thread.join();
    for(unsigned worker=0;worker<workers;++worker){stats.exact_samples+=counts[worker].exact;
        stats.certificate_samples+=counts[worker].certificate;stats.subdivisions+=counts[worker].split;
        stats.exact_leaves+=counts[worker].leaves;stats.interpolated+=counts[worker].interpolated;
        stats.exact_pixels+=counts[worker].exact_pixels;stats.conv2d_accepted+=counts[worker].accepted;
        accumulate_trace_stats(stats.trace,worker_stats[worker]);}

    std::vector<std::pair<int,int>> boundary;
    for(int y=0;y<height;++y)for(int x=0;x<width;++x){const TerminalLabel label=
        ownership[static_cast<std::size_t>(y)*width+x];bool mixed=false;
        for(int oy=-1;oy<=1&&!mixed;++oy)for(int ox=-1;ox<=1;++ox){if(!ox&&!oy)continue;
            const int qx=x+ox,qy=y+oy;if(qx<0||qx>=width||qy<0||qy>=height)continue;
            mixed=mixed||ownership[static_cast<std::size_t>(qy)*width+qx].primitive!=label.primitive;}
        if(mixed)boundary.push_back({x,y});}
    stats.boundary_pixels=boundary.size();std::atomic<std::size_t> next_boundary{0};
    std::vector<std::thread> boundary_threads;std::vector<TraceStats> boundary_stats(workers);
    std::vector<std::uint64_t> boundary_samples(workers);
    for(unsigned worker=0;worker<workers;++worker)boundary_threads.emplace_back([&,worker]{
        TraceContext ctx{context.scene,context.beams,context.field,&boundary_stats[worker],context.sealed_optics,context.optical_cutoff};
        auto sample=[&](double x,double y){TerminalPoint result;const Ray ray=camera_ray(camera,width,height,x,y);
            const Hit hit=first_hit(ctx.scene,ray);++boundary_samples[worker];if(!hit.valid){result.label={-1,0};return result;}
            const std::uint64_t topology=terminal_topology_signature(ctx,ray,hit);std::uint64_t signature=0;
            result.value=trace_primary(ctx,ray,hit,&signature);result.label={hit.primitive,signature^topology};return result;};
        auto integrate=[&](auto&& self,double cx,double cy,double width,int depth)->RGB{
            // Positive two-point Gauss integration over the pixel basin.  The
            // four weights are equal; subdivision preserves positivity and a
            // unit coefficient sum at every accepted depth.
            const double node=width/(2*std::sqrt(3.0));
            std::array<TerminalPoint,4> q{{sample(cx-node,cy-node),sample(cx+node,cy-node),
                sample(cx+node,cy+node),sample(cx-node,cy+node)}};bool same=true;
            for(int i=1;i<4;++i)same=same&&(q[i].label==q[0].label);RGB average{};
            for(const TerminalPoint& point:q)average+=point.value*.25;if(same||depth>=2)return average;
            const double offset=width*.25,child_width=width*.5;
            return (self(self,cx-offset,cy-offset,child_width,depth+1)+
                    self(self,cx+offset,cy-offset,child_width,depth+1)+
                    self(self,cx+offset,cy+offset,child_width,depth+1)+
                    self(self,cx-offset,cy+offset,child_width,depth+1))*.25;};
        for(;;){const std::size_t index=next_boundary.fetch_add(1);if(index>=boundary.size())break;
            const auto [x,y]=boundary[index];linear[static_cast<std::size_t>(y)*width+x]=integrate(integrate,x,y,1.0,0);}
    });
    for(auto& thread:boundary_threads)thread.join();
    for(unsigned worker=0;worker<workers;++worker){stats.boundary_samples+=boundary_samples[worker];
        stats.exact_samples+=boundary_samples[worker];accumulate_trace_stats(stats.trace,boundary_stats[worker]);}

    std::vector<std::uint8_t> image(pixel_count*3);
    for(std::size_t index=0;index<pixel_count;++index)for(int c=0;c<3;++c)image[index*3+c]=tone_byte(linear[index][c]);
    stats.raster_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return image;
}

void write_ppm(const std::string& path,const std::vector<std::uint8_t>& image,int width,int height){std::ofstream out(path,std::ios::binary);
    if(!out)throw std::runtime_error("cannot open output");out<<"P6\n"<<width<<" "<<height<<"\n255\n";
    out.write(reinterpret_cast<const char*>(image.data()),static_cast<std::streamsize>(image.size()));}

int run_intersection_benchmark(std::uint64_t primitive_count,std::uint64_t ray_count,const std::string& acceleration){
    const auto total_start=Clock::now();Scene scene;const int material=add_material(scene,
        {"benchmark diffuse",MaterialKind::Diffuse,{.7,.7,.7}});scene.primitives.reserve(primitive_count);
    const std::uint64_t side=static_cast<std::uint64_t>(std::ceil(std::sqrt(static_cast<double>(primitive_count))));
    for(std::uint64_t i=0;i<primitive_count;++i){const double x=(static_cast<double>(i%side)-.5*(side-1))*.82;
        const double y=(static_cast<double>(i/side)-.5*(side-1))*.82;add_sphere(scene,"",{x,y,-8-.07*(i%7)},.31,material,false);}
    const auto build_start=Clock::now();build_scene_bvh(scene);const double build_ms=
        std::chrono::duration<double,std::milli>(Clock::now()-build_start).count();scene.use_bvh=acceleration=="bvh"||
        (acceleration=="auto"&&primitive_count>=64);const std::string selected=scene.use_bvh?"bvh":"linear";
    const auto query_start=Clock::now();std::uint64_t checksum=0,hits=0;for(std::uint64_t i=0;i<ray_count;++i){
        const std::uint64_t target_index=(i*11400714819323198485ULL)%primitive_count;
        const Vec3 origin{0,0,15},target=scene.primitives[target_index].center;const Hit hit=first_hit(scene,{origin,unit(target-origin)});
        if(hit.valid){++hits;checksum=checksum*1099511628211ULL+static_cast<std::uint64_t>(hit.primitive+1);}}
    const double query_ms=std::chrono::duration<double,std::milli>(Clock::now()-query_start).count();const double total_ms=
        std::chrono::duration<double,std::milli>(Clock::now()-total_start).count();const std::size_t primitive_bytes=
        scene.primitives.capacity()*sizeof(Primitive),bvh_bytes=scene.bvh_nodes.capacity()*sizeof(BvhNode)+
        scene.bvh_primitives.capacity()*sizeof(int);
    std::cout<<std::fixed<<std::setprecision(3)<<"{\n  \"benchmark\": \"intersection_scaling\""
        <<",\n  \"primitives\": "<<primitive_count<<",\n  \"rays\": "<<ray_count
        <<",\n  \"acceleration\": \""<<selected<<"\""
        <<",\n  \"bvh_nodes\": "<<scene.bvh_nodes.size()<<",\n  \"primitive_stride_bytes\": "<<sizeof(Primitive)
        <<",\n  \"primitive_storage_bytes\": "<<primitive_bytes<<",\n  \"bvh_storage_bytes\": "<<bvh_bytes
        <<",\n  \"build_ms\": "<<build_ms<<",\n  \"query_ms\": "<<query_ms
        <<",\n  \"nanoseconds_per_ray\": "<<(query_ms*1e6/ray_count)<<",\n  \"hits\": "<<hits
        <<",\n  \"checksum\": "<<checksum<<",\n  \"total_ms\": "<<total_ms<<"\n}\n";return 0;
}

bool self_test(){const Scene scene=build_regime_scene();const BeamField beams=compile_beam_field(scene);const BeamField oriented=
    compile_oriented_beam_field(scene);const TransportField field=compile_transport_field(scene,beams);
    if(scene.primitives.size()<29||scene.materials.size()<13||beams.launched<800||beams.dielectric_crossings==0||beams.deposits.empty())return false;
    if(!oriented.oriented||oriented.launched!=beams.launched||oriented.oriented_sheets.empty())return false;
    for(const OrientedBeamSheet& sheet:oriented.oriented_sheets)if(!(sheet.position_determinant>0&&std::isfinite(sheet.anisotropy)&&
        std::isfinite(sheet.cross_coupling)&&sheet.sample_count>0))return false;
    if(scene.prism_bottom<0||scene.prism_top<0||scene.floor<0||
        std::abs(scene.primitives[scene.prism_bottom].center.y-scene.primitives[scene.floor].origin.y)>.00101)return false;
    if(scene.primitives[scene.prism_bottom].normal.y>-.999||scene.primitives[scene.prism_top].normal.y<.999)return false;
    Scene linear_scene=scene;linear_scene.use_bvh=false;const Camera acceleration_camera=make_camera(96,60);
    for(int y=0;y<60;y+=3)for(int x=0;x<96;x+=3){const Ray ray=camera_ray(acceleration_camera,96,60,x,y);
        const Hit accelerated=first_hit(scene,ray),linear=first_hit(linear_scene,ray);
        if(accelerated.valid!=linear.valid||accelerated.primitive!=linear.primitive||
            (accelerated.valid&&std::abs(accelerated.t-linear.t)>1e-9))return false;}
    std::vector<Ray> packet_rays(96);std::vector<Hit> packet_hits(96);std::vector<double> packet_x(96),packet_y(96),packet_z(96),packet_t(96);
    std::vector<int> packet_id(96);for(int y=0;y<60;y+=7){for(int x=0;x<96;++x){packet_rays[x]=camera_ray(acceleration_camera,96,60,x,y);
            packet_x[x]=packet_rays[x].direction.x;packet_y[x]=packet_rays[x].direction.y;packet_z[x]=packet_rays[x].direction.z;}
        first_hit_camera_packet(linear_scene,acceleration_camera.origin,packet_rays.data(),packet_x.data(),packet_y.data(),
            packet_z.data(),packet_hits.data(),packet_t.data(),packet_id.data(),96);
    for(int x=0;x<96;++x){const Hit scalar=first_hit(linear_scene,packet_rays[x]);if(packet_hits[x].valid!=scalar.valid||
                packet_hits[x].primitive!=scalar.primitive||(scalar.valid&&std::abs(packet_hits[x].t-scalar.t)>1e-12))return false;}}
    JourneyPose previous_pose=journey_pose(0);for(int sample=1;sample<=2048;++sample){const JourneyPose pose=
            journey_pose(double(sample)/2048);const Vec3 segment=pose.position-previous_pose.position;
        const double distance=norm(segment);if(distance<1e-8||norm(pose.target-pose.position)<.5||
            pose.position.y<.2||pose.position.y>5.7)return false;
        const Hit obstruction=first_hit(linear_scene,{previous_pose.position,segment/distance},distance-1e-6);
        if(obstruction.valid)return false;previous_pose=pose;}
    for(const std::string mode:{"aperture-canyon","mirror-relay","occlusion-garden"}){
        Scene demonstrator=build_demonstrator_scene(mode);demonstrator.use_bvh=false;
        if(demonstrator.primitives.size()<=scene.primitives.size()||demonstrator.primitives.size()>96||
            demonstrator.bvh_nodes.empty())return false;
        JourneyPose prior=journey_pose(0,mode);for(int sample=1;sample<=256;++sample){
            const JourneyPose pose=journey_pose(double(sample)/256,mode);const Vec3 segment=pose.position-prior.position;
            const double distance=norm(segment);if(distance<1e-8||norm(pose.target-pose.position)<.5)return false;
            const Hit obstruction=first_hit(demonstrator,{prior.position,segment/distance},distance-1e-6);
            if(obstruction.valid)return false;prior=pose;}}
    Scene aperture_test;const int matte=add_material(aperture_test,{"matte",MaterialKind::Diffuse,{.7,.7,.7}});
    const int emitter_material=add_material(aperture_test,{"emitter",MaterialKind::Emissive,{1,1,1},{1,1,1},0});
    const int emitter=add_rect(aperture_test,"emitter",{-1,-1,2},{2,0,0},{0,2,0},{0,0,-1},emitter_material,false);
    aperture_test.area_lights.push_back({emitter,{1,1,1}});
    const RGB clear=area_irradiance(aperture_test,{0,0,0},{0,0,1},-1);
    add_sphere(aperture_test,"blocker",{0,0,1},.25,matte,false);
    const RGB clipped=area_irradiance(aperture_test,{0,0,0},{0,0,1},-1);
    const Primitive& test_light=aperture_test.primitives[emitter];
    const EmitterPartition test_partition=build_emitter_partition(aperture_test,test_light,{0,0,0},-1,emitter);
    const auto centre_cuts=emitter_row_cuts(aperture_test,test_light,test_partition,{0,0,0},.5);
    if(!(clipped[0]>0&&clipped[0]<clear[0]&&centre_cuts.size()>=4))return false;
    if(field.node_primitives.size()<12||field.nonzeros<80||field.iterations<2)return false;const int cavity=field.index_by_primitive[scene.cavity_target];
    if(cavity<0||rgb_energy(field.direct[cavity])>1e-9||rgb_energy(field.bounce[cavity])<=0)return false;
    int metal=-1;for(int i=0;i<static_cast<int>(scene.primitives.size());++i)if(scene.primitives[i].name=="rough_metal_sphere")metal=i;
    if(count_sheet_shadow_samples(scene,metal)==0)return false;TraceContext context{scene,beams,field,nullptr};RenderStats exact_stats,adaptive_stats;
    // A close-view gold highlight has a delta direction which first meets the
    // smoke sheet and then reaches the rectangular emitter.  That source is
    // already present in specular_area(); its dielectric continuation must stay
    // sealed or the sheet edge becomes a bright floating duplicate.
    const Camera close_camera=make_original_variant_camera(1920,1280,"default-closer");
    const Ray close_ray=camera_ray(close_camera,1920,1280,1218,729);const Hit sheet_hit=first_hit(scene,close_ray);
    if(!sheet_hit.valid||sheet_hit.primitive!=scene.glass_sheet)return false;const Vec3 sheet_reflection=
        unit(reflect(close_ray.direction,sheet_hit.normal));const Hit gold_hit=first_hit(scene,
        {sheet_hit.position+sheet_reflection*3e-4,sheet_reflection},std::numeric_limits<double>::infinity(),sheet_hit.primitive);
    if(!gold_hit.valid||gold_hit.primitive!=metal)return false;const Vec3 gold_reflection=
        unit(reflect(sheet_reflection,gold_hit.normal));const Ray leaked_emitter_path{
            gold_hit.position+gold_reflection*3e-4,gold_reflection};const Hit nested_sheet=first_hit(scene,
                leaked_emitter_path,std::numeric_limits<double>::infinity(),gold_hit.primitive);
    if(!nested_sheet.valid||nested_sheet.primitive!=scene.glass_sheet)return false;const double unsealed_area=
        trace_channel(context,leaked_emitter_path,6,0,nullptr,gold_hit.primitive);const double sealed_area=
        trace_channel(context,leaked_emitter_path,6,0,nullptr,gold_hit.primitive,
            std::numeric_limits<double>::quiet_NaN(),true);
    if(unsealed_area<10||std::abs(sealed_area)>1e-10)return false;
    TraceStats sealed_stats,recursive_stats;TraceContext sealed_context{scene,beams,field,&sealed_stats,true};
    TraceContext recursive_context{scene,beams,field,&recursive_stats,false};const Camera overhead=make_prism_overhead_camera(160,100);
    const Ray loop_ray=camera_ray(overhead,160,100,80,50);const Hit loop_hit=first_hit(scene,loop_ray);
    const RGB sealed_value=trace_primary(sealed_context,loop_ray,loop_hit,nullptr);
    const RGB recursive_value=trace_primary(recursive_context,loop_ray,loop_hit,nullptr);double loop_difference=0;
    for(int channel=0;channel<3;++channel)loop_difference=std::max(loop_difference,
        std::abs(sealed_value[channel]-recursive_value[channel]));
    if(sealed_stats.feedback_loops==0||loop_difference>.01)return false;
    const auto exact=render_exact(context,160,100,exact_stats);const auto adaptive=render_adaptive(context,160,100,1,adaptive_stats);
    int maximum=0;for(std::size_t i=0;i<exact.size();++i)maximum=std::max(maximum,std::abs(int(exact[i])-int(adaptive[i])));
    if(maximum>1)return false;RenderStats edge_stats;const Camera edge_camera=make_camera(96,60);
    const CameraSpecularField viewer_origin_field=compile_viewer_origin_specular_field(context,edge_camera.origin);
    const auto edge_field=render_visible_edge_field(context,96,60,1,edge_stats,&edge_camera,&viewer_origin_field);
    return edge_field.size()==96u*60u*3u&&edge_stats.boundary_pixels>0&&edge_stats.visible_edge_segments>0&&
        edge_stats.boundary_samples>0&&edge_stats.analytic_edge_pixels>0&&edge_stats.viewer_origin_field_reused&&
        edge_stats.camera_specular_build_ms==0&&edge_stats.camera_specular_quadrature_samples==0;}

} // namespace

int main(int argc,char** argv)try{int width=960,height=640,terminal_error=1,animation_frames=0,animation_fps=30;
    bool test=false;std::string out="/tmp/regime_scene.ppm";
    std::uint64_t max_primitives=96;double max_build_seconds=60,oriented_blur=.14,optical_cutoff=1e-5,journey_position=-1;
    std::uint64_t benchmark_primitives=0,benchmark_rays=4096;
    std::string beam_mode="clustered",camera_mode="default",optical_mode="sealed",acceleration="auto",scene_mode="standard";
    for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--self-test"){test=true;continue;}
        if(i+1>=argc)throw std::runtime_error("missing argument value");if(arg=="--width")width=std::stoi(argv[++i]);
        else if(arg=="--height")height=std::stoi(argv[++i]);else if(arg=="--terminal-error")terminal_error=std::stoi(argv[++i]);
        else if(arg=="--out")out=argv[++i];else if(arg=="--max-primitives")max_primitives=std::stoull(argv[++i]);
        else if(arg=="--max-build-seconds")max_build_seconds=std::stod(argv[++i]);else if(arg=="--beam-mode")beam_mode=argv[++i];
        else if(arg=="--oriented-blur")oriented_blur=std::stod(argv[++i]);else if(arg=="--camera")camera_mode=argv[++i];
        else if(arg=="--optical-mode")optical_mode=argv[++i];
        else if(arg=="--optical-cutoff")optical_cutoff=std::stod(argv[++i]);
        else if(arg=="--acceleration")acceleration=argv[++i];
        else if(arg=="--scene")scene_mode=argv[++i];
        else if(arg=="--journey-position")journey_position=std::stod(argv[++i]);
        else if(arg=="--animation-frames")animation_frames=std::stoi(argv[++i]);
        else if(arg=="--animation-fps")animation_fps=std::stoi(argv[++i]);
        else if(arg=="--benchmark-primitives")benchmark_primitives=std::stoull(argv[++i]);
        else if(arg=="--benchmark-rays")benchmark_rays=std::stoull(argv[++i]);
        else throw std::runtime_error("unknown argument: "+arg);}
    if(test){if(!self_test())throw std::runtime_error("regime-scene invariants failed");
        std::cout<<"regime-scene invariants: ok\n";return 0;}if(width<16||height<16||terminal_error<0)throw std::runtime_error("invalid render configuration");
    if(animation_frames<0||animation_frames>1800||animation_fps<1||animation_fps>60)
        throw std::runtime_error("invalid animation configuration");
    if(journey_position!= -1&&!(journey_position>=0&&journey_position<1))
        throw std::runtime_error("journey position must be in [0,1)");
    if(animation_frames&&journey_position>=0)throw std::runtime_error("journey position is for still renders");
    if(animation_frames&&(width%2||height%2))throw std::runtime_error("animation dimensions must be even");
    if(acceleration!="auto"&&acceleration!="bvh"&&acceleration!="linear")
        throw std::runtime_error("acceleration must be auto, bvh, or linear");
    if(benchmark_primitives){if(benchmark_primitives>max_primitives)throw std::runtime_error("primitive ceiling reached");
        if(!benchmark_rays)throw std::runtime_error("benchmark rays must be positive");
        return run_intersection_benchmark(benchmark_primitives,benchmark_rays,acceleration);}
    const auto total_start=Clock::now(),scene_start=Clock::now();Scene scene=build_demonstrator_scene(scene_mode);
    const double scene_ms=std::chrono::duration<double,std::milli>(Clock::now()-scene_start).count();
    if(scene.primitives.size()>max_primitives)throw std::runtime_error("primitive ceiling reached");
    scene.use_bvh=acceleration=="bvh"||(acceleration=="auto"&&scene.primitives.size()>=64);
    const std::string selected_acceleration=scene.use_bvh?"bvh":"linear";
    if(beam_mode!="clustered"&&beam_mode!="oriented")throw std::runtime_error("beam mode must be clustered or oriented");
    if(optical_mode!="sealed"&&optical_mode!="recursive")throw std::runtime_error("optical mode must be sealed or recursive");
    if(!(optical_cutoff>0&&optical_cutoff<1e-2))throw std::runtime_error("optical cutoff must be in (0, 1e-2)");
    const auto beam_start=Clock::now();const BeamField beams=beam_mode=="oriented"?
        compile_oriented_beam_field(scene,oriented_blur):compile_beam_field(scene);
    const double beam_ms=std::chrono::duration<double,std::milli>(Clock::now()-beam_start).count();const auto field_start=Clock::now();
    const TransportField field=compile_transport_field(scene,beams);const double field_ms=std::chrono::duration<double,std::milli>(Clock::now()-field_start).count();
    if((beam_ms+field_ms)>max_build_seconds*1000)throw std::runtime_error("field build-time ceiling reached");
    TraceContext context{scene,beams,field,nullptr,optical_mode=="sealed",optical_cutoff};
    if(animation_frames){const auto parameters=journey_arc_parameters(animation_frames,scene_mode);double render_ms=0,adaptive_ms=0,
            edge_ms=0,boundary_ms=0,path_length=0;Vec3 previous=journey_pose(parameters.front(),scene_mode).position;
        const auto animation_start=Clock::now();const int progress_interval=std::max(1,animation_frames/40);
        std::ios::sync_with_stdio(false);for(int frame=0;frame<animation_frames;++frame){const JourneyPose pose=
                journey_pose(parameters[frame],scene_mode);if(frame)path_length+=norm(pose.position-previous);previous=pose.position;
            const Camera camera=make_look_camera(width,height,pose.position,pose.target,{0,1,0},pose.half_fov_degrees);
            RenderStats frame_stats;const std::vector<std::uint8_t> image=terminal_error>0?
                render_visible_edge_field(context,width,height,terminal_error,frame_stats,&camera):
                render_exact(context,width,height,frame_stats,&camera);
            std::cout.write(reinterpret_cast<const char*>(image.data()),static_cast<std::streamsize>(image.size()));
            if(!std::cout)throw std::runtime_error("raw animation stream failed");render_ms+=frame_stats.raster_ms;
            adaptive_ms+=frame_stats.adaptive_ms;edge_ms+=frame_stats.edge_discovery_ms;
            boundary_ms+=frame_stats.boundary_reconstruction_ms;
            if((frame+1)%progress_interval==0||frame+1==animation_frames)std::cerr<<"animation "<<(frame+1)<<"/"<<animation_frames
                <<" average-render-ms="<<(render_ms/(frame+1))<<"\n";}
        path_length+=norm(journey_pose(parameters.front(),scene_mode).position-previous);std::cout.flush();const double total_animation_ms=
            std::chrono::duration<double,std::milli>(Clock::now()-animation_start).count();std::cerr<<std::fixed<<std::setprecision(3)
            <<"{\n  \"scene\": \""<<scene_mode<<"\",\n  \"animation_frames\": "<<animation_frames<<",\n  \"animation_fps\": "<<animation_fps
            <<",\n  \"duration_seconds\": "<<double(animation_frames)/animation_fps
            <<",\n  \"path_length\": "<<path_length<<",\n  \"average_scene_speed\": "
            <<path_length/(double(animation_frames)/animation_fps)
            <<",\n  \"average_render_ms\": "<<render_ms/animation_frames
            <<",\n  \"average_adaptive_ms\": "<<adaptive_ms/animation_frames
            <<",\n  \"average_edge_discovery_ms\": "<<edge_ms/animation_frames
            <<",\n  \"average_boundary_reconstruction_ms\": "<<boundary_ms/animation_frames
            <<",\n  \"animation_total_ms\": "<<total_animation_ms<<"\n}\n";return 0;}
    const JourneyPose still_pose=journey_position>=0?journey_pose(journey_position,scene_mode):JourneyPose{};
    const Camera camera=journey_position>=0?
        make_look_camera(width,height,still_pose.position,still_pose.target,{0,1,0},still_pose.half_fov_degrees):
        make_camera_mode(width,height,camera_mode);RenderStats render_stats;
    std::vector<std::uint8_t> image=terminal_error>0?render_visible_edge_field(context,width,height,terminal_error,render_stats,&camera):
        render_exact(context,width,height,render_stats,&camera);
    const auto write_start=Clock::now();write_ppm(out,image,width,height);const double write_ms=
        std::chrono::duration<double,std::milli>(Clock::now()-write_start).count();const double total_ms=
        std::chrono::duration<double,std::milli>(Clock::now()-total_start).count();const int cavity=field.index_by_primitive[scene.cavity_target];
    int metal_primitive=-1;for(int i=0;i<static_cast<int>(scene.primitives.size());++i)
        if(scene.primitives[i].name=="rough_metal_sphere")metal_primitive=i;
    const std::uint64_t prism_bottom_crossings=beams.dielectric_by_primitive[scene.prism_bottom];
    const std::uint64_t prism_top_crossings=beams.dielectric_by_primitive[scene.prism_top];
    const double prism_floor_separation=scene.primitives[scene.prism_bottom].center.y-scene.primitives[scene.floor].origin.y;
    double maximum_beam_anisotropy=1,maximum_beam_cross_coupling=0;for(const OrientedBeamSheet& sheet:beams.oriented_sheets){
        maximum_beam_anisotropy=std::max(maximum_beam_anisotropy,sheet.anisotropy);
        maximum_beam_cross_coupling=std::max(maximum_beam_cross_coupling,sheet.cross_coupling);}
    std::cout<<std::fixed<<std::setprecision(3)<<"{\n  \"width\": "<<width<<",\n  \"height\": "<<height
        <<",\n  \"scene\": \""<<scene_mode<<"\""
        <<",\n  \"camera\": \""<<(journey_position>=0?"journey":camera_mode)<<"\""
        <<",\n  \"journey_position\": "<<(journey_position>=0?journey_position:-1)
        <<",\n  \"optical_mode\": \""<<optical_mode<<"\""
        <<",\n  \"acceleration\": \""<<selected_acceleration<<"\""
        <<",\n  \"acceleration_requested\": \""<<acceleration<<"\""
        <<",\n  \"optical_cutoff\": "<<std::scientific<<optical_cutoff<<std::fixed
        <<",\n  \"analytic_primitives\": "<<scene.primitives.size()<<",\n  \"materials\": "<<scene.materials.size()
        <<",\n  \"bvh_nodes\": "<<scene.bvh_nodes.size()
        <<",\n  \"diffuse_nodes\": "<<field.node_primitives.size()<<",\n  \"transport_couplings\": "<<field.nonzeros
        <<",\n  \"transport_propagated_edges\": "<<field.propagated_edges
        <<",\n  \"direct_atlas_samples\": "<<field.direct_atlas_samples
        <<",\n  \"transport_iterations\": "<<field.iterations<<",\n  \"transport_residual\": "<<field.final_residual
        <<",\n  \"beam_mode\": \""<<beam_mode<<"\""
        <<",\n  \"beam_bundles\": "<<scene.beams.size()<<",\n  \"spectral_beam_fibres\": "<<beams.launched
        <<",\n  \"beam_dielectric_crossings\": "<<beams.dielectric_crossings
        <<",\n  \"prism_bottom_crossings\": "<<prism_bottom_crossings
        <<",\n  \"prism_top_crossings\": "<<prism_top_crossings
        <<",\n  \"prism_floor_separation\": "<<prism_floor_separation
        <<",\n  \"beam_mirror_crossings\": "<<beams.mirror_crossings<<",\n  \"caustic_regions\": "<<(beams.oriented?beams.oriented_sheets.size():beams.deposits.size())
        <<",\n  \"oriented_beam_sheets\": "<<beams.oriented_sheets.size()
        <<",\n  \"maximum_beam_anisotropy\": "<<maximum_beam_anisotropy
        <<",\n  \"maximum_beam_cross_coupling\": "<<maximum_beam_cross_coupling
        <<",\n  \"glass_sheet_shadow_samples\": "<<count_sheet_shadow_samples(scene,metal_primitive)
        <<",\n  \"cavity_direct_energy\": "<<(cavity>=0?rgb_energy(field.direct[cavity]):0)
        <<",\n  \"cavity_indirect_energy\": "<<(cavity>=0?rgb_energy(field.bounce[cavity]):0)
        <<",\n  \"terminal_error\": "<<terminal_error<<",\n  \"terminal_exact_samples\": "<<render_stats.exact_samples
        <<",\n  \"terminal_topology_queries\": "<<render_stats.topology_queries
        <<",\n  \"terminal_topology_runs\": "<<render_stats.topology_runs
        <<",\n  \"terminal_certificate_samples\": "<<render_stats.certificate_samples
        <<",\n  \"terminal_subdivisions\": "<<render_stats.subdivisions
        <<",\n  \"terminal_exact_leaves\": "<<render_stats.exact_leaves
        <<",\n  \"terminal_interpolated_pixels\": "<<render_stats.interpolated
        <<",\n  \"terminal_exact_pixels\": "<<render_stats.exact_pixels
        <<",\n  \"terminal_boundary_pixels\": "<<render_stats.boundary_pixels
        <<",\n  \"terminal_boundary_samples\": "<<render_stats.boundary_samples
        <<",\n  \"terminal_boundary_topology_samples\": "<<render_stats.boundary_topology_samples
        <<",\n  \"terminal_boundary_radiance_samples\": "<<render_stats.boundary_radiance_samples
        <<",\n  \"primary_packet_rays\": "<<render_stats.primary_packet_rays
        <<",\n  \"terminal_edge_candidate_segments\": "<<render_stats.edge_candidate_segments
        <<",\n  \"terminal_visible_edge_segments\": "<<render_stats.visible_edge_segments
        <<",\n  \"terminal_edge_refinements\": "<<render_stats.edge_refinements
        <<",\n  \"terminal_prism_top_edge_pixels\": "<<render_stats.prism_top_edge_pixels
        <<",\n  \"terminal_prism_top_mixed_pixels\": "<<render_stats.prism_top_mixed_pixels
        <<",\n  \"terminal_analytic_edge_pixels\": "<<render_stats.analytic_edge_pixels
        <<",\n  \"terminal_filtered_edge_pixels\": "<<render_stats.filtered_edge_pixels
        <<",\n  \"primary_traces\": "<<render_stats.trace.primary<<",\n  \"secondary_traces\": "<<render_stats.trace.secondary
        <<",\n  \"mirror_events\": "<<render_stats.trace.mirror<<",\n  \"metal_events\": "<<render_stats.trace.metal
        <<",\n  \"dielectric_events\": "<<render_stats.trace.dielectric
        <<",\n  \"camera_prism_bottom_events\": "<<render_stats.trace.prism_bottom_events
        <<",\n  \"camera_prism_top_events\": "<<render_stats.trace.prism_top_events
        <<",\n  \"specular_caustic_camera_events\": "<<render_stats.trace.specular_caustic
        <<",\n  \"direct_atlas_gathers\": "<<render_stats.trace.direct_atlas_gathers
        <<",\n  \"direct_exact_calls\": "<<render_stats.trace.direct_exact_calls
        <<",\n  \"specular_area_calls\": "<<render_stats.trace.specular_area_calls
        <<",\n  \"primary_specular_area_calls\": "<<render_stats.trace.primary_specular_area_calls
        <<",\n  \"secondary_specular_area_calls\": "<<render_stats.trace.secondary_specular_area_calls
        <<",\n  \"secondary_specular_weight_lt_1e-4\": "<<render_stats.trace.secondary_specular_weight_lt_1e4
        <<",\n  \"secondary_specular_weight_lt_1e-3\": "<<render_stats.trace.secondary_specular_weight_lt_1e3
        <<",\n  \"secondary_specular_weight_lt_1e-2\": "<<render_stats.trace.secondary_specular_weight_lt_1e2
        <<",\n  \"camera_specular_gathers\": "<<render_stats.trace.camera_specular_gathers
        <<",\n  \"camera_specular_samples\": "<<render_stats.camera_specular_samples
        <<",\n  \"camera_specular_quadrature_samples\": "<<render_stats.camera_specular_quadrature_samples
        <<",\n  \"camera_specular_build_ms\": "<<render_stats.camera_specular_build_ms
        <<",\n  \"viewer_origin_field_reused\": "<<(render_stats.viewer_origin_field_reused?"true":"false")
        <<",\n  \"specular_memo_hits\": "<<render_stats.trace.specular_memo_hits
        <<",\n  \"surface_radiance_memo_hits\": "<<render_stats.trace.surface_radiance_memo_hits
        <<",\n  \"surface_radiance_memo_stores\": "<<render_stats.trace.surface_radiance_memo_stores
        <<",\n  \"emitter_source_rows\": "<<render_stats.trace.emitter_rows
        <<",\n  \"emitter_source_intervals\": "<<render_stats.trace.emitter_intervals
        <<",\n  \"emitter_quadrature_samples\": "<<render_stats.trace.emitter_quadrature_samples
        <<",\n  \"feedback_loops\": "<<render_stats.trace.feedback_loops
        <<",\n  \"feedback_returns\": "<<render_stats.trace.feedback_returns
        <<",\n  \"sealed_feedback_tails\": "<<render_stats.trace.sealed_feedback_tails
        <<",\n  \"sealed_residual_weight\": "<<render_stats.trace.sealed_residual_weight
        <<",\n  \"maximum_optical_packets\": "<<render_stats.trace.maximum_optical_packets
        <<",\n  \"scene_compile_ms\": "<<scene_ms
        <<",\n  \"beam_compile_ms\": "<<beam_ms
        <<",\n  \"field_compile_ms\": "<<field_ms<<",\n  \"raster_ms\": "<<render_stats.raster_ms
        <<",\n  \"adaptive_ms\": "<<render_stats.adaptive_ms
        <<",\n  \"edge_discovery_ms\": "<<render_stats.edge_discovery_ms
        <<",\n  \"boundary_reconstruction_ms\": "<<render_stats.boundary_reconstruction_ms
        <<",\n  \"write_ms\": "<<write_ms<<",\n  \"total_ms\": "<<total_ms<<",\n  \"output\": \""<<out<<"\"\n}\n";
    return 0;}catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}
