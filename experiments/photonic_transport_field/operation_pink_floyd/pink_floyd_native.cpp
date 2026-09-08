#include <algorithm>
#include <array>
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
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr double pi = std::numbers::pi_v<double>;
constexpr double medium_sigma_s=.115;
constexpr double medium_sigma_a=.010;
constexpr double medium_sigma_t=medium_sigma_s+medium_sigma_a;

struct Vec2 { double x=0, y=0; };
struct Vec3 { double x=0, y=0, z=0; };
struct RGB { double r=0, g=0, b=0; };

Vec2 operator+(Vec2 a, Vec2 b) { return {a.x+b.x,a.y+b.y}; }
Vec2 operator-(Vec2 a, Vec2 b) { return {a.x-b.x,a.y-b.y}; }
Vec2 operator*(Vec2 a, double s) { return {a.x*s,a.y*s}; }
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 operator*(Vec3 a, double s) { return {a.x*s,a.y*s,a.z*s}; }
Vec3 operator/(Vec3 a, double s) { return a*(1.0/s); }
RGB operator+(RGB a, RGB b) { return {a.r+b.r,a.g+b.g,a.b+b.b}; }
RGB operator*(RGB a, double s) { return {a.r*s,a.g*s,a.b*s}; }
RGB& operator+=(RGB& a, RGB b) { a=a+b; return a; }
double channel(RGB a,int index) { return index==0?a.r:(index==1?a.g:a.b); }
void set_channel(RGB& a,int index,double value) { if(index==0)a.r=value; else if(index==1)a.g=value; else a.b=value; }
double dot(Vec2 a, Vec2 b) { return a.x*b.x+a.y*b.y; }
double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
double cross(Vec2 a, Vec2 b) { return a.x*b.y-a.y*b.x; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
double norm(Vec2 a) { return std::hypot(a.x,a.y); }
double norm(Vec3 a) { return std::sqrt(dot(a,a)); }
Vec2 unit(Vec2 a) { const double n=norm(a); return n>0?a*(1.0/n):Vec2{}; }
Vec3 unit(Vec3 a) { const double n=norm(a); return n>0?a/n:Vec3{}; }

struct Ray { Vec3 origin{}, direction{}; };

RGB wavelength_rgb(double wavelength) {
    double r=0,g=0,b=0;
    if(wavelength<440) { r=(440-wavelength)/60; b=1; }
    else if(wavelength<490) { g=(wavelength-440)/50; b=1; }
    else if(wavelength<510) { g=1; b=(510-wavelength)/20; }
    else if(wavelength<580) { r=(wavelength-510)/70; g=1; }
    else if(wavelength<645) { r=1; g=(645-wavelength)/65; }
    else r=1;
    return {r,g,b};
}

// Schott N-F2 Sellmeier coefficients, wavelength in micrometres.
double refractive_index(double wavelength_nm) {
    const double l=wavelength_nm*1e-3,l2=l*l;
    constexpr double b1=1.34533359,b2=.209073176,b3=.937357162;
    constexpr double c1=.00997743871,c2=.0470450767,c3=111.886764;
    return std::sqrt(1+b1*l2/(l2-c1)+b2*l2/(l2-c2)+b3*l2/(l2-c3));
}

double glass_absorption(double wavelength_nm) {
    const double blue=std::max(0.0,(520.0-wavelength_nm)/120.0);
    return .008+.012*blue*blue;
}

struct Refraction2 { bool valid=false; Vec2 direction{}; double transmission=0, residual=0; };
struct Refraction3 { bool valid=false; Vec3 direction{}; double transmission=0, residual=0; };

Refraction2 refract(Vec2 incident,Vec2 normal,double n1,double n2) {
    incident=unit(incident); normal=unit(normal);
    const double cos_i=-dot(incident,normal),eta=n1/n2;
    const double sin2_t=eta*eta*std::max(0.0,1-cos_i*cos_i);
    if(cos_i<=0 || sin2_t>=1) return {};
    const double cos_t=std::sqrt(std::max(0.0,1-sin2_t));
    const Vec2 transmitted=unit(incident*eta+normal*(eta*cos_i-cos_t));
    const double rs=(n1*cos_i-n2*cos_t)/(n1*cos_i+n2*cos_t);
    const double rp=(n1*cos_t-n2*cos_i)/(n1*cos_t+n2*cos_i);
    return {true,transmitted,std::clamp(1-.5*(rs*rs+rp*rp),0.0,1.0),
        std::abs(n1*std::sqrt(std::max(0.0,1-cos_i*cos_i))-n2*std::sqrt(sin2_t))};
}

Refraction3 refract(Vec3 incident,Vec3 normal,double n1,double n2) {
    incident=unit(incident); normal=unit(normal);
    const double cos_i=-dot(incident,normal),eta=n1/n2;
    const double sin2_t=eta*eta*std::max(0.0,1-cos_i*cos_i);
    if(cos_i<=0 || sin2_t>=1) return {};
    const double cos_t=std::sqrt(std::max(0.0,1-sin2_t));
    const Vec3 transmitted=unit(incident*eta+normal*(eta*cos_i-cos_t));
    const double rs=(n1*cos_i-n2*cos_t)/(n1*cos_i+n2*cos_t);
    const double rp=(n1*cos_t-n2*cos_i)/(n1*cos_t+n2*cos_i);
    return {true,transmitted,std::clamp(1-.5*(rs*rs+rp*rp),0.0,1.0),
        std::abs(n1*std::sqrt(std::max(0.0,1-cos_i*cos_i))-n2*std::sqrt(sin2_t))};
}

struct Edge2 { Vec2 a{},b{},outward{}; };
struct Prism {
    // Triangular optical footprint in the sheet's x-z plane.
    std::array<Vec2,3> vertex{{{-0.72,-0.58},{0.72,-0.58},{0.0,0.62}}};
    std::array<Edge2,3> edge{};
    double bottom_y=-.66,top_y=.16;
    Prism() {
        for(int i=0;i<3;++i) {
            const Vec2 a=vertex[i],b=vertex[(i+1)%3],e=b-a;
            edge[i]={a,b,unit(Vec2{e.y,-e.x})};
        }
    }
};

struct BoundaryHit2 { bool valid=false; int face=-1; double distance=0; Vec2 point{}; };

BoundaryHit2 first_boundary(const Prism& prism,Vec2 origin,Vec2 direction,int ignore=-1) {
    BoundaryHit2 best; best.distance=std::numeric_limits<double>::infinity();
    for(int face=0;face<3;++face) {
        if(face==ignore) continue;
        const Vec2 a=prism.edge[face].a,e=prism.edge[face].b-a;
        const double denominator=cross(direction,e);
        if(std::abs(denominator)<1e-14) continue;
        const double distance=cross(a-origin,e)/denominator;
        const double coordinate=cross(a-origin,direction)/denominator;
        if(distance<=1e-8 || coordinate<0 || coordinate>1) continue;
        if(distance<best.distance) best={true,face,distance,origin+direction*distance};
    }
    return best;
}

struct OpticalPath {
    bool valid=false;
    BoundaryHit2 entry{},exit{};
    Vec2 internal{},outgoing{};
    double entry_t=0,exit_t=0,length=0,residual=0;
};

OpticalPath trace_prism(const Prism& prism,Vec2 source,Vec2 incoming,double wavelength) {
    OpticalPath path;
    path.entry=first_boundary(prism,source,incoming);
    if(!path.entry.valid) return path;
    const double n=refractive_index(wavelength);
    const Refraction2 enter=refract(incoming,prism.edge[path.entry.face].outward,1,n);
    if(!enter.valid) return path;
    path.internal=enter.direction;
    path.exit=first_boundary(prism,path.entry.point+path.internal*1e-7,path.internal,path.entry.face);
    if(!path.exit.valid) return path;
    const Refraction2 leave=refract(path.internal,prism.edge[path.exit.face].outward*-1,n,1);
    if(!leave.valid) return path;
    path.outgoing=leave.direction;
    path.entry_t=enter.transmission; path.exit_t=leave.transmission;
    path.length=norm(path.exit.point-path.entry.point);
    path.residual=std::max(enter.residual,leave.residual);
    path.valid=true;
    return path;
}

int retained_modes(double sigma,double half_width) {
    int modes=0;
    while(modes<96) {
        const double k=pi*(modes+1)/half_width;
        if(std::exp(-.5*k*k*sigma*sigma)<1e-9) break;
        ++modes;
    }
    return modes;
}

struct FourierBeam {
    Vec3 origin{},direction{};
    RGB color{};
    double x_begin=0,x_end=0,wavelength=550,power=0;
    double sigma_y=.025,sigma_z=.060,half_y=.16,half_z=.24;
    double extinction=0,diffusion_y=0,diffusion_z=0;
    int modes_y=0,modes_z=0;

    double distance_at_x(double x) const { return (x-origin.x)/direction.x; }
    Vec3 centre_at_x(double x) const { return origin+direction*distance_at_x(x); }

    double fourier_axis(double delta,double sigma,double half,int modes,double diffusion,double distance) const {
        double sum=1;
        for(int m=1;m<=modes;++m) {
            const double k=pi*m/half;
            const double coefficient=std::exp(-.5*k*k*sigma*sigma-diffusion*k*k*distance);
            sum+=2*coefficient*std::cos(k*delta);
        }
        return std::max(0.0,sum/(2*half));
    }

    double exact_axis(double delta,double sigma,double diffusion,double distance) const {
        const double variance=sigma*sigma+2*diffusion*distance;
        return std::exp(-.5*delta*delta/variance)/std::sqrt(2*pi*variance);
    }

    double density(Vec3 p,bool fourier) const {
        if(p.x<x_begin || p.x>x_end) return 0;
        const double distance=distance_at_x(p.x);
        if(distance<0) return 0;
        const Vec3 centre=origin+direction*distance;
        const double dy=p.y-centre.y,dz=p.z-centre.z;
        // A local retained chart is not a periodic world.  Its support bound
        // is part of the packet registration and seals Fourier replicas.
        if(std::abs(dy)>=half_y||std::abs(dz)>=half_z) return 0;
        const double fy=fourier?fourier_axis(dy,sigma_y,half_y,modes_y,diffusion_y,distance)
                               :exact_axis(dy,sigma_y,diffusion_y,distance);
        const double fz=fourier?fourier_axis(dz,sigma_z,half_z,modes_z,diffusion_z,distance)
                               :exact_axis(dz,sigma_z,diffusion_z,distance);
        return std::exp(-extinction*distance)*fy*fz;
    }
};

struct AABB { Vec3 lo{},hi{}; };

bool intersect_box(const Ray& ray,const AABB& box,double& near_t,double& far_t) {
    near_t=0; far_t=std::numeric_limits<double>::infinity();
    const std::array<double,3> o{{ray.origin.x,ray.origin.y,ray.origin.z}};
    const std::array<double,3> d{{ray.direction.x,ray.direction.y,ray.direction.z}};
    const std::array<double,3> lo{{box.lo.x,box.lo.y,box.lo.z}},hi{{box.hi.x,box.hi.y,box.hi.z}};
    for(int axis=0;axis<3;++axis) {
        if(std::abs(d[axis])<1e-14) { if(o[axis]<lo[axis]||o[axis]>hi[axis]) return false; continue; }
        double a=(lo[axis]-o[axis])/d[axis],b=(hi[axis]-o[axis])/d[axis];
        if(a>b) std::swap(a,b);
        near_t=std::max(near_t,a); far_t=std::min(far_t,b);
        if(far_t<=near_t) return false;
    }
    return far_t>0;
}

struct Triangle { Vec3 a{},b{},c{},normal{}; int face=-1; };

std::vector<Triangle> prism_triangles(const Prism& prism) {
    std::vector<Triangle> t;
    const auto v=[&](int i,double height){return Vec3{prism.vertex[i].x,height,prism.vertex[i].y};};
    t.push_back({v(0,prism.top_y),v(2,prism.top_y),v(1,prism.top_y),{0,1,0},3});
    t.push_back({v(0,prism.bottom_y),v(1,prism.bottom_y),v(2,prism.bottom_y),{0,-1,0},4});
    for(int i=0;i<3;++i) {
        const int j=(i+1)%3;
        const Vec3 n{prism.edge[i].outward.x,0,prism.edge[i].outward.y};
        t.push_back({v(i,prism.top_y),v(i,prism.bottom_y),v(j,prism.bottom_y),n,i});
        t.push_back({v(i,prism.top_y),v(j,prism.bottom_y),v(j,prism.top_y),n,i});
    }
    return t;
}

struct Hit { bool valid=false; double t=0; Vec3 point{},normal{}; int face=-1; };

Hit intersect_triangles(const Ray& ray,const std::vector<Triangle>& triangles,double minimum_t=1e-6) {
    Hit best; best.t=std::numeric_limits<double>::infinity();
    for(const Triangle& tri:triangles) {
        const Vec3 e1=tri.b-tri.a,e2=tri.c-tri.a,p=cross(ray.direction,e2);
        const double det=dot(e1,p);
        if(std::abs(det)<1e-12) continue;
        const double inv=1/det;
        const Vec3 s=ray.origin-tri.a;
        const double u=dot(s,p)*inv;
        if(u<0||u>1) continue;
        const Vec3 q=cross(s,e1);
        const double v=dot(ray.direction,q)*inv;
        if(v<0||u+v>1) continue;
        const double distance=dot(e2,q)*inv;
        if(distance>minimum_t&&distance<best.t) best={true,distance,ray.origin+ray.direction*distance,tri.normal,tri.face};
    }
    return best;
}

struct Camera {
    // A near-overhead pinhole view: the prism is directly below the camera,
    // while a small +z displacement preserves the vertical extrusion in the
    // image instead of degenerating it to a flat triangle.
    Vec3 origin{.06,4.55,.82},forward{},right{},up{};
    double tangent=std::tan(27.0*pi/180);
    int width=256,height=256;
    Camera(int w,int h):width(w),height(h) {
        forward=unit(Vec3{.02,-.48,-.02}-origin);
        right=unit(cross(forward,{0,1,0}));
        up=cross(right,forward);
    }
    Ray ray(int x,int y) const {
        const double nx=(2*(x+.5)/width-1)*tangent;
        const double ny=(1-2*(y+.5)/height)*tangent;
        return {origin,unit(forward+right*nx+up*ny)};
    }
};

struct Image {
    int width=0,height=0;
    std::vector<RGB> pixel;
    Image(int w,int h):width(w),height(h),pixel(static_cast<std::size_t>(w)*h) {}
    RGB& at(int x,int y) { return pixel[static_cast<std::size_t>(y)*width+x]; }
};

struct Scene {
    Prism prism{};
    std::vector<Triangle> triangles=prism_triangles(prism);
    std::vector<FourierBeam> sheet_fields,prism_feed,prism_internal,prism_emergent;
    AABB sheet{{-2.25,-.72,-1.35},{2.25,-.66,1.35}};
    AABB source_body{{-1.32,-.66,-.59},{-1.10,-.48,-.37}};
    Vec2 source_position{-1.10,-.48};
    Vec2 source_direction{std::cos(26*pi/180),std::sin(26*pi/180)};
    double sheet_top=-.66,membrane_y=-.692;
};

struct SimulationStats {
    std::uint64_t spectral_paths=0,volume_density_evaluations=0,fourier_terms=0;
    int maximum_y_modes=0,maximum_z_modes=0;
    double incident_energy=0,sheet_bypass_energy=0,transmitted_energy=0;
    double entry_reflection=0,glass_absorption=0,exit_reflection=0;
    double maximum_snell_residual=0;
    double blue_angle=0,green_angle=0,red_angle=0,render_ms=0;
};

Scene build_scene(SimulationStats& stats) {
    Scene scene;
    constexpr int wavelengths=65;
    // The collimator is thin across the sheet.  The y split is independent:
    // its exact half-Gaussian moments still put half of the packet into the
    // parent sheet and half through the prism boundary.
    constexpr double source_sigma_y=.045,source_sigma_z=.028;
    constexpr double half_fraction=.5;
    const double split_shift=source_sigma_y*std::sqrt(2/pi);
    const double split_sigma=source_sigma_y*std::sqrt(1-2/pi);
    for(int band=0;band<wavelengths;++band) {
        const double wavelength=420.0+260.0*band/(wavelengths-1);
        const double spectral_power=1.0/wavelengths;
        const Vec2 optical_source=scene.source_position;
        const OpticalPath path=trace_prism(scene.prism,optical_source,scene.source_direction,wavelength);
        ++stats.spectral_paths; stats.incident_energy+=spectral_power;
        if(!path.valid) throw std::runtime_error("configured spectral lane failed prism traversal");
        stats.maximum_snell_residual=std::max(stats.maximum_snell_residual,path.residual);
        const RGB color=wavelength_rgb(wavelength);

        FourierBeam sheet_packet;
        sheet_packet.origin={optical_source.x,scene.sheet_top-split_shift,optical_source.y};
        sheet_packet.direction={scene.source_direction.x,0,scene.source_direction.y};
        sheet_packet.color=color; sheet_packet.wavelength=wavelength;
        sheet_packet.power=spectral_power*half_fraction;
        sheet_packet.x_begin=scene.source_position.x; sheet_packet.x_end=scene.sheet.hi.x;
        sheet_packet.sigma_y=split_sigma; sheet_packet.sigma_z=source_sigma_z;
        sheet_packet.modes_y=retained_modes(sheet_packet.sigma_y,sheet_packet.half_y);
        sheet_packet.modes_z=retained_modes(sheet_packet.sigma_z,sheet_packet.half_z);
        sheet_packet.extinction=1.40;
        sheet_packet.diffusion_y=3.0e-5; sheet_packet.diffusion_z=3.0e-5;
        scene.sheet_fields.push_back(sheet_packet);
        stats.sheet_bypass_energy+=sheet_packet.power;

        FourierBeam prism_packet;
        prism_packet.origin={optical_source.x,scene.sheet_top+split_shift,optical_source.y};
        prism_packet.direction={scene.source_direction.x,0,scene.source_direction.y};
        prism_packet.color=color; prism_packet.wavelength=wavelength;
        prism_packet.power=spectral_power*half_fraction;
        prism_packet.x_begin=optical_source.x; prism_packet.x_end=path.entry.point.x;
        prism_packet.sigma_y=split_sigma; prism_packet.sigma_z=source_sigma_z;
        prism_packet.modes_y=retained_modes(prism_packet.sigma_y,prism_packet.half_y);
        prism_packet.modes_z=retained_modes(prism_packet.sigma_z,prism_packet.half_z);
        scene.prism_feed.push_back(prism_packet);

        const double after_entry=prism_packet.power*path.entry_t;
        const double after_glass=after_entry*std::exp(-glass_absorption(wavelength)*path.length);
        const double after_exit=after_glass*path.exit_t;
        stats.entry_reflection+=prism_packet.power-after_entry;
        stats.glass_absorption+=after_entry-after_glass;
        stats.exit_reflection+=after_glass-after_exit;
        stats.transmitted_energy+=after_exit;

        // The boundary hands a retained packet to the prism child volume.
        // It is normally invisible; a small physical bulk-scattering term is
        // evaluated later only where a camera characteristic crosses it.
        FourierBeam internal;
        internal.origin={path.entry.point.x,scene.sheet_top+split_shift,path.entry.point.y};
        internal.direction={path.internal.x,0,path.internal.y};
        internal.color=color; internal.wavelength=wavelength; internal.power=after_entry;
        internal.x_begin=path.entry.point.x; internal.x_end=path.exit.point.x;
        internal.sigma_y=split_sigma; internal.sigma_z=source_sigma_z;
        internal.half_y=std::max(internal.half_y,6*internal.sigma_y);
        internal.half_z=std::max(internal.half_z,6*internal.sigma_z);
        internal.modes_y=retained_modes(internal.sigma_y,internal.half_y);
        internal.modes_z=retained_modes(internal.sigma_z,internal.half_z);
        internal.extinction=glass_absorption(wavelength);
        scene.prism_internal.push_back(internal);

        constexpr double epsilon=1e-4;
        const OpticalPath minus=trace_prism(scene.prism,{optical_source.x,optical_source.y-epsilon},
            scene.source_direction,wavelength);
        const OpticalPath plus=trace_prism(scene.prism,{optical_source.x,optical_source.y+epsilon},
            scene.source_direction,wavelength);
        if(!minus.valid||!plus.valid) throw std::runtime_error("boundary Jacobian left configured face");
        const double spatial_jacobian=std::abs((plus.exit.point.y-minus.exit.point.y)/(2*epsilon));

        FourierBeam outgoing;
        outgoing.origin={path.exit.point.x,scene.sheet_top+split_shift,path.exit.point.y};
        outgoing.direction={path.outgoing.x,0,path.outgoing.y};
        outgoing.color=color; outgoing.wavelength=wavelength; outgoing.power=after_exit;
        outgoing.x_begin=path.exit.point.x; outgoing.x_end=scene.sheet.hi.x;
        outgoing.sigma_y=split_sigma; outgoing.sigma_z=source_sigma_z*spatial_jacobian;
        outgoing.half_y=std::max(outgoing.half_y,6*outgoing.sigma_y);
        outgoing.half_z=std::max(outgoing.half_z,6*outgoing.sigma_z);
        outgoing.modes_y=retained_modes(outgoing.sigma_y,outgoing.half_y);
        outgoing.modes_z=retained_modes(outgoing.sigma_z,outgoing.half_z);
        // The emergent packet is still centred above the membrane.  This y
        // diffusion is the medium's physical downward coupling into the thin
        // sheet; z remains tight enough for the wavelength fan to separate.
        outgoing.extinction=medium_sigma_t; outgoing.diffusion_y=1.2e-4; outgoing.diffusion_z=1.5e-5;
        scene.prism_emergent.push_back(outgoing);

        stats.maximum_y_modes=std::max({stats.maximum_y_modes,sheet_packet.modes_y,prism_packet.modes_y,
            internal.modes_y,outgoing.modes_y});
        stats.maximum_z_modes=std::max({stats.maximum_z_modes,sheet_packet.modes_z,prism_packet.modes_z,
            internal.modes_z,outgoing.modes_z});
        if(band==0) stats.blue_angle=std::atan2(path.outgoing.y,path.outgoing.x)*180/pi;
        if(band==wavelengths/2) stats.green_angle=std::atan2(path.outgoing.y,path.outgoing.x)*180/pi;
        if(band==wavelengths-1) stats.red_angle=std::atan2(path.outgoing.y,path.outgoing.x)*180/pi;
    }
    return scene;
}

double henyey_greenstein(double cosine,double g) {
    const double denominator=1+g*g-2*g*cosine;
    return (1-g*g)/(4*pi*denominator*std::sqrt(denominator));
}

struct MembranePoint { Vec3 position{},normal{}; };

MembranePoint membrane_point(const Scene& scene,double x,double z) {
    constexpr std::array<double,6> amplitude{{.0019,.0014,.0011,.0008,.00065,.0005}};
    constexpr std::array<double,6> kx{{17,31,47,71,103,149}};
    constexpr std::array<double,6> kz{{29,-43,61,-89,127,-173}};
    constexpr std::array<double,6> phase{{.2,1.1,2.4,.7,2.9,1.8}};
    double h=scene.membrane_y,dx=0,dz=0;
    for(std::size_t i=0;i<amplitude.size();++i) {
        const double argument=kx[i]*x+kz[i]*z+phase[i];
        h+=amplitude[i]*std::sin(argument);
        dx+=amplitude[i]*kx[i]*std::cos(argument);
        dz+=amplitude[i]*kz[i]*std::cos(argument);
    }
    return {{x,h,z},unit(Vec3{-dx,1,-dz})};
}

RGB sheet_face_radiance(Vec3 face_point,Vec3 to_camera,const Scene& scene,bool fourier,SimulationStats& stats) {
    const MembranePoint membrane=membrane_point(scene,face_point.x,face_point.z);
    RGB outgoing{.00022,.00025,.00034};
    const double face_distance=scene.sheet_top-membrane.position.y;
    const double face_transmission=std::exp(-medium_sigma_t*face_distance/std::max(.08,to_camera.y));
    const auto gather=[&](const FourierBeam& beam) {
        if(face_point.x<beam.x_begin||face_point.x>beam.x_end) return;
        ++stats.volume_density_evaluations;
        stats.fourier_terms+=static_cast<std::uint64_t>(beam.modes_y+beam.modes_z+2);
        const double density=beam.density(membrane.position,fourier);
        if(density<=0) return;
        const double phase=henyey_greenstein(dot(beam.direction,to_camera),.12);
        const Vec3 half_vector=unit(to_camera-beam.direction);
        const double aligned=std::max(0.0,dot(membrane.normal,half_vector));
        const double glitter=std::pow(aligned,96);
        const double deformation=.22+.78*std::abs(dot(beam.direction,membrane.normal));
        const double response=.43*medium_sigma_s*phase*deformation+.55*glitter;
        outgoing+=beam.color*(beam.power*density*response*face_transmission);
    };
    for(const FourierBeam& beam:scene.sheet_fields) gather(beam);
    for(const FourierBeam& beam:scene.prism_emergent) gather(beam);
    return outgoing;
}

RGB background_radiance(const Ray& ray,const Scene& scene,bool fourier,SimulationStats& stats,double* distance_out=nullptr) {
    double sheet_t=std::numeric_limits<double>::infinity();
    if(ray.direction.y< -1e-10) {
        const double t=(scene.sheet_top-ray.origin.y)/ray.direction.y;
        if(t>1e-6) {
            const Vec3 point=ray.origin+ray.direction*t;
            if(point.x>=scene.sheet.lo.x&&point.x<=scene.sheet.hi.x&&
                    point.z>=scene.sheet.lo.z&&point.z<=scene.sheet.hi.z) sheet_t=t;
        }
    }
    if(std::isfinite(sheet_t)) {
        if(distance_out) *distance_out=sheet_t;
        const Vec3 point=ray.origin+ray.direction*sheet_t;
        return sheet_face_radiance(point,ray.direction*-1,scene,fourier,stats);
    }
    if(distance_out) *distance_out=std::numeric_limits<double>::infinity();
    return {};
}

RGB glass_radiance(const Ray& camera_ray,const Hit& entry,const Scene& scene,bool fourier,SimulationStats& stats) {
    constexpr std::array<double,3> camera_wavelength{{620,550,460}};
    RGB result{};
    for(int c=0;c<3;++c) {
        const double wavelength=camera_wavelength[c],n=refractive_index(wavelength);
        const Refraction3 into=refract(camera_ray.direction,entry.normal,1,n);
        if(!into.valid) continue;
        const Ray internal{entry.point+into.direction*2e-6,into.direction};
        const Hit exit=intersect_triangles(internal,scene.triangles);
        if(!exit.valid) continue;
        RGB behind{};
        double exit_transmission=0;
        if(exit.face==4) {
            constexpr double sheet_index=1.08;
            const Refraction3 into_sheet=refract(internal.direction,exit.normal*-1,n,sheet_index);
            if(!into_sheet.valid) continue;
            behind=sheet_face_radiance(exit.point,into_sheet.direction*-1,scene,fourier,stats);
            exit_transmission=into_sheet.transmission;
        } else {
            const Refraction3 out=refract(internal.direction,exit.normal*-1,n,1);
            if(!out.valid) continue;
            const Ray external{exit.point+out.direction*2e-6,out.direction};
            behind=background_radiance(external,scene,fourier,stats);
            exit_transmission=out.transmission;
        }
        const double length=norm(exit.point-entry.point);
        const double transmitted=channel(behind,c)*into.transmission*exit_transmission*
            std::exp(-glass_absorption(wavelength)*length);
        set_channel(result,c,transmitted);
    }

    // Exact thin-beam line crossing in the prism child volume.  The camera
    // ray is steep and each retained packet is narrow in y, so integrating
    // the normalized y Gaussian reduces to one evaluation at its centre
    // plane times 1/|dy/dt|.  This exposes only genuine bulk scatter from the
    // beam near the prism's lower edge; the packet remains invisible elsewhere.
    constexpr double glass_sigma_s=.0075;
    for(const FourierBeam& beam:scene.prism_internal) {
        const Refraction3 into=refract(camera_ray.direction,entry.normal,1,refractive_index(beam.wavelength));
        if(!into.valid||std::abs(into.direction.y)<1e-8) continue;
        const Ray internal_ray{entry.point+into.direction*2e-6,into.direction};
        const Hit exit=intersect_triangles(internal_ray,scene.triangles);
        if(!exit.valid) continue;
        const double segment_length=norm(exit.point-entry.point);
        const double t=(beam.origin.y-entry.point.y)/into.direction.y;
            if(t<=0||t>=segment_length) continue;
            const Vec3 crossing=entry.point+into.direction*t;
            if(crossing.x<beam.x_begin||crossing.x>beam.x_end) continue;
            ++stats.volume_density_evaluations;
            stats.fourier_terms+=static_cast<std::uint64_t>(beam.modes_y+beam.modes_z+2);
            const double density=beam.density(crossing,fourier);
            if(density<=0) continue;
            const double integrated_y=std::sqrt(2*pi)*beam.sigma_y/std::abs(into.direction.y);
            const double phase=henyey_greenstein(dot(beam.direction,into.direction*-1),.08);
            const double camera_attenuation=std::exp(-glass_absorption(beam.wavelength)*t);
            result+=beam.color*(beam.power*density*integrated_y*glass_sigma_s*phase*camera_attenuation);
    }

    // The polished face is nearly specular. A small measured rough component
    // allows the collimated source to illuminate its entry edge without
    // turning the complete prism into a diffuse emitter.
    if(entry.face==2) {
        const Vec3 to_camera=camera_ray.direction*-1;
        for(const FourierBeam& beam:scene.prism_feed) {
            if(entry.point.x<beam.x_begin-.02||entry.point.x>beam.x_end+.02) continue;
            const double density=beam.density(entry.point,fourier);
            if(density<=0) continue;
            const Vec3 half_vector=unit(to_camera-beam.direction);
            const double alignment=std::max(0.0,dot(entry.normal,half_vector));
            result+=beam.color*(beam.power*density*.018*std::pow(alignment,160));
        }
    }
    return result;
}

RGB surface_radiance(const Ray& ray,const Scene& scene,bool fourier,SimulationStats& stats,double& surface_t) {
    const Hit prism_hit=intersect_triangles(ray,scene.triangles);
    double background_t=0;
    const RGB background=background_radiance(ray,scene,fourier,stats,&background_t);
    double source_near=0,source_far=0;
    const bool source_hit=intersect_box(ray,scene.source_body,source_near,source_far)&&source_near<background_t;
    if(source_hit&&(!prism_hit.valid||source_near<prism_hit.t)) {
        surface_t=source_near;
        return RGB{.006,.006,.007};
    }
    if(prism_hit.valid&&prism_hit.t<background_t) {
        surface_t=prism_hit.t;
        return glass_radiance(ray,prism_hit,scene,fourier,stats);
    }
    surface_t=background_t;
    return background;
}

std::vector<std::uint8_t> render(int width,int height,bool fourier,SimulationStats& stats) {
    const auto start=Clock::now();
    const Scene scene=build_scene(stats);
    const Camera camera(width,height);
    Image image(width,height);
    for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
        const Ray ray=camera.ray(x,y);
        double surface_t=std::numeric_limits<double>::infinity();
        const RGB color=surface_radiance(ray,scene,fourier,stats,surface_t);
        image.at(x,y)=color;
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(width)*height*3);
    for(std::size_t i=0;i<image.pixel.size();++i) {
        const std::array<double,3> channel{{image.pixel[i].r,image.pixel[i].g,image.pixel[i].b}};
        for(int c=0;c<3;++c) {
            const double mapped=1-std::exp(-18*std::max(0.0,channel[c]));
            bytes[i*3+c]=static_cast<std::uint8_t>(std::lround(255*std::pow(std::clamp(mapped,0.0,1.0),1/2.2)));
        }
    }
    stats.render_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    return bytes;
}

void write_ppm(const std::string& path,const std::vector<std::uint8_t>& image,int width,int height) {
    std::ofstream file(path,std::ios::binary);
    if(!file) throw std::runtime_error("cannot open output");
    file<<"P6\n"<<width<<" "<<height<<"\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()),static_cast<std::streamsize>(image.size()));
    if(!file) throw std::runtime_error("cannot write output");
}

bool self_test() {
    SimulationStats construction;
    const Scene scene=build_scene(construction);
    if(scene.sheet_fields.size()!=65||scene.prism_feed.size()!=65||scene.prism_internal.size()!=65||
            scene.prism_emergent.size()!=65) {
        std::cerr<<"packet-count gate failed\n"; return false;
    }
    if(construction.maximum_snell_residual>2e-14) { std::cerr<<"Snell gate failed\n"; return false; }
    if(!(construction.blue_angle<construction.green_angle&&construction.green_angle<construction.red_angle)) {
        std::cerr<<"dispersion-order gate failed\n"; return false;
    }
    const double ledger=construction.sheet_bypass_energy+construction.transmitted_energy+construction.entry_reflection+
        construction.glass_absorption+construction.exit_reflection;
    if(std::abs(ledger-construction.incident_energy)>2e-13) { std::cerr<<"energy gate failed: "<<ledger<<"\n"; return false; }
    double maximum_relative=0;
    for(const FourierBeam& beam:scene.prism_emergent) {
        for(double fraction:{.1,.5,.9}) {
            const double x=beam.x_begin+fraction*(beam.x_end-beam.x_begin);
            const Vec3 centre=beam.centre_at_x(x);
            for(double offset:{-2.0,-1.0,0.0,1.0,2.0}) {
                const Vec3 p{centre.x,centre.y+offset*beam.sigma_y,centre.z+.7*offset*beam.sigma_z};
                const double exact=beam.density(p,false),approximate=beam.density(p,true);
                maximum_relative=std::max(maximum_relative,std::abs(approximate-exact)/std::max(exact,1e-14));
            }
        }
    }
    if(maximum_relative>2e-5) { std::cerr<<"profile gate failed: "<<maximum_relative<<"\n"; return false; }
    SimulationStats fs,rs;
    const auto f=render(64,64,true,fs),r=render(64,64,false,rs);
    int maximum=0; double mean=0;
    for(std::size_t i=0;i<f.size();++i) { const int d=std::abs(int(f[i])-int(r[i])); maximum=std::max(maximum,d); mean+=d; }
    mean/=f.size();
    if(maximum>2||mean>=.02) std::cerr<<"image gate failed: max="<<maximum<<" mean="<<mean<<"\n";
    return maximum<=2&&mean<.02;
}

} // namespace

int main(int argc,char** argv) try {
    int width=256,height=256;
    bool test=false;
    std::string backend="fourier",out="/tmp/pink_floyd.ppm";
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--self-test") { test=true; continue; }
        if(i+1>=argc) throw std::runtime_error("missing argument value");
        if(arg=="--width") width=std::stoi(argv[++i]);
        else if(arg=="--height") height=std::stoi(argv[++i]);
        else if(arg=="--backend") backend=argv[++i];
        else if(arg=="--out") out=argv[++i];
        else throw std::runtime_error("unknown argument: "+arg);
    }
    if(backend!="fourier"&&backend!="reference") throw std::runtime_error("backend must be fourier or reference");
    if(width<32||height<32) throw std::runtime_error("invalid dimensions");
    if(test) {
        if(!self_test()) throw std::runtime_error("pink-floyd invariants failed");
        std::cout<<"operation-pink-floyd invariants: ok\n";
        return 0;
    }
    SimulationStats stats;
    const auto image=render(width,height,backend=="fourier",stats);
    write_ppm(out,image,width,height);
    std::cout<<std::fixed<<std::setprecision(8)<<"{\n  \"operation\": \"pink-floyd-volume\",\n  \"backend\": \""<<backend
        <<"\",\n  \"width\": "<<width<<",\n  \"height\": "<<height
        <<",\n  \"spectral_boundary_packets\": "<<stats.spectral_paths
        <<",\n  \"maximum_y_modes\": "<<stats.maximum_y_modes<<",\n  \"maximum_z_modes\": "<<stats.maximum_z_modes
        <<",\n  \"volume_density_evaluations\": "<<stats.volume_density_evaluations
        <<",\n  \"fourier_terms_evaluated\": "<<stats.fourier_terms
        <<",\n  \"maximum_snell_residual\": "<<stats.maximum_snell_residual
        <<",\n  \"incident_energy\": "<<stats.incident_energy<<",\n  \"transmitted_energy\": "<<stats.transmitted_energy
        <<",\n  \"sheet_bypass_energy\": "<<stats.sheet_bypass_energy
        <<",\n  \"entry_reflection\": "<<stats.entry_reflection<<",\n  \"glass_absorption\": "<<stats.glass_absorption
        <<",\n  \"exit_reflection\": "<<stats.exit_reflection
        <<",\n  \"blue_exit_angle_degrees\": "<<stats.blue_angle
        <<",\n  \"green_exit_angle_degrees\": "<<stats.green_angle
        <<",\n  \"red_exit_angle_degrees\": "<<stats.red_angle
        <<",\n  \"render_ms\": "<<stats.render_ms<<",\n  \"output\": \""<<out<<"\"\n}\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<"error: "<<error.what()<<"\n";
    return 2;
}
