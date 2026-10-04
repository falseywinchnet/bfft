#include "engine.hpp"
#include "broadphase.hpp"
#include "collide.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace obligation {
using namespace zc::phys;
namespace {
constexpr double infinity=std::numeric_limits<double>::infinity();
constexpr double time_epsilon=1e-11;
Quat turned(Quat q,Vec3 rotation) {
    double angle=length(rotation);if(angle<1e-14)return q;
    double scale=std::sin(.5*angle)/angle;
    Quat turn{std::cos(.5*angle),rotation.x*scale,rotation.y*scale,rotation.z*scale};
    Quat out=multiply(turn,q);double n=std::sqrt(out.w*out.w+out.x*out.x+out.y*out.y+out.z*out.z);
    return {out.w/n,out.x/n,out.y/n,out.z/n};
}
double orientation_distance(Quat a,Quat b) {
    double d=std::abs(a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z);return 2*std::sqrt(std::max(0.0,1-d*d));
}
double component(Vec3 v,int k){return k==0?v.x:k==1?v.y:v.z;}
void component(Vec3& v,int k,double value){if(k==0)v.x=value;else if(k==1)v.y=value;else v.z=value;}
bool finite(Vec3 v){return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);}
double exhaustive_separation(const WorldHull& a,const WorldHull& b,Vec3& normal) {
    double best=-infinity;
    auto axis=[&](Vec3 n){double norm=length(n);if(norm<1e-15)return;n=n*(1/norm);
        double al=infinity,ah=-infinity,bl=infinity,bh=-infinity;
        for(Vec3 p:a.vertices){double x=dot(n,p);al=std::min(al,x);ah=std::max(ah,x);}
        for(Vec3 p:b.vertices){double x=dot(n,p);bl=std::min(bl,x);bh=std::max(bh,x);}
        double gap=al-bh;if(bl-ah>gap){gap=bl-ah;n=n*(-1);}
        if(gap>best){best=gap;normal=n;}
    };
    for(Vec3 n:a.normals)axis(n);for(Vec3 n:b.normals)axis(n);
    for(Vec3 ea:a.edge_directions)for(Vec3 eb:b.edge_directions){Vec3 n=cross(ea,eb);
        if(dot(n,n)>1e-24*dot(ea,ea)*dot(eb,eb))axis(n);}
    return best;
}
struct Body {
    std::shared_ptr<const Shape> shape;
    Pose pose;Vec3 v,w,angular_momentum,acceleration,torque,alpha;
    Mat3 inertia_local,inverse_local,inertia,inverse;
    double inv_mass=0,stamp=0,envelope_until=0;
    bool fixed=false,alive=true,equilibrium=false;
    int leaf=-1;
    Bounds bounds{};
    std::vector<WorldHull> cached_hulls;
    std::vector<Pose> cached_poses;
    std::vector<char> cache_valid;
    std::uint64_t revision=1,clock=0;
    std::uint64_t visit=0;
    std::vector<int> pairs;
};
struct Atomic {
    Vec3 local_a,local_b,normal_a;
    Vec3 tangent_force;
    double normal_force=0;
    int hull_a=-1,hull_b=-1;
};
struct Pair {
    int a=-1,b=-1;
    bool standing=false;
    std::vector<Atomic> atoms;
    std::vector<char> supported_parts;
    std::vector<Vec3> separating_axes;
    Pose pose_a,pose_b;
    std::uint64_t clock=0,revision_a=0,revision_b=0;
    double deadline=infinity,last_dt=0;
};
struct Event { double t;std::uint64_t order,clock;int id,kind; };
struct Later { bool operator()(const Event& a,const Event& b)const{return a.t!=b.t?a.t>b.t:a.order>b.order;} };
struct Row {
    int pair=0,atom=0,a=-1,b=-1;
    Vec3 n,ra,rb,pa,pb,t1,t2;
    double gap=0,mu=0,bounce=0,kn=0,k11=0,k12=0,k22=0;
    double impulse_n=0;Vec3 impulse_t;
};
}

struct World::Impl {
    Parameters p;double now=0;Statistics stats;Tree tree;
    std::vector<Body> bodies;std::deque<Pair> pairs;
    std::unordered_map<std::uint64_t,int> pair_index;
    std::priority_queue<Event,std::vector<Event>,Later> events;
    std::uint64_t serial=0;
    std::uint64_t visit_serial=0;
    CollideScratch scratch;std::vector<ManifoldPoint> manifold;
    const std::vector<WorldHull> empty_hulls;

    BodyState predicted(int id,double t) const {
        const Body& b=bodies[id];BodyState s;s.pose=b.pose;s.v=b.v;s.w=b.w;s.asleep=b.equilibrium;
        if(b.fixed||b.equilibrium)return s;
        double dt=std::max(0.0,t-b.stamp);
        s.pose.p=b.pose.p+b.v*dt+b.acceleration*(.5*dt*dt);s.v=b.v+b.acceleration*dt;
        s.pose.q=turned(b.pose.q,b.w*dt+b.alpha*(.5*dt*dt));
        s.w=mul(rotated(to_matrix(s.pose.q),b.inverse_local),b.angular_momentum+b.torque*dt);
        return s;
    }
    void tensors(Body& b) {
        b.inertia=rotated(to_matrix(b.pose.q),b.inertia_local);b.inverse=rotated(to_matrix(b.pose.q),b.inverse_local);
        b.w=mul(b.inverse,b.angular_momentum);
        b.alpha=mul(b.inverse,b.torque-cross(b.w,b.angular_momentum));
    }
    void materialize(int id) {
        Body& b=bodies[id];if(b.stamp>=now-time_epsilon)return;
        BodyState s=predicted(id,now);b.angular_momentum=b.angular_momentum+b.torque*(now-b.stamp);
        b.pose=s.pose;b.v=s.v;b.w=s.w;b.stamp=now;tensors(b);stats.integrated_bodies++;
        if(!finite(b.pose.p)||!finite(b.v)||!finite(b.w))throw std::runtime_error("nonfinite obligation trajectory");
    }
    double speed_bound(int id,double horizon) const {
        if(id<0||bodies[id].fixed||bodies[id].equilibrium)return 0;
        const Body& b=bodies[id];double dt=std::max(0.0,now-b.stamp)+horizon;
        return length(b.v)+length(b.acceleration)*dt+b.shape->radius*(length(b.w)+length(b.alpha)*dt);
    }
    Bounds envelope(int id,double horizon) {
        Body& b=bodies[id];BodyState s=predicted(id,now);
        Bounds box{{infinity,infinity,infinity},{-infinity,-infinity,-infinity}};
        for(const HullData& hull:b.shape->hulls)for(Vec3 v:hull.vertices){Vec3 w=s.pose.p+rotate(s.pose.q,v);
            for(int k=0;k<3;k++){component(box.low,k,std::min(component(box.low,k),component(w,k)));component(box.high,k,std::max(component(box.high,k),component(w,k)));}}
        if(b.fixed||b.equilibrium)horizon=0;
        for(int k=0;k<3;k++){
            double v=component(s.v,k),a=component(b.acceleration,k),end=v*horizon+.5*a*horizon*horizon;
            double low=std::min(0.0,end),high=std::max(0.0,end);
            if(a!=0){double t=-v/a;if(t>0&&t<horizon){double x=v*t+.5*a*t*t;low=std::min(low,x);high=std::max(high,x);}}
            component(box.low,k,component(box.low,k)+low);component(box.high,k,component(box.high,k)+high);
        }
        double rotation=b.shape->radius*std::min(2.0,(length(s.w)+.5*length(b.alpha)*horizon)*horizon);
        double pad=rotation+2*p.release_gap;Vec3 padding{pad,pad,pad};box.low=box.low-padding;box.high=box.high+padding;return box;
    }
    const std::vector<WorldHull>& placed(int id,double t,const std::vector<char>* selection=nullptr) {
        Body& b=bodies[id];Pose pose=predicted(id,t).pose;
        b.cached_hulls.resize(b.shape->hulls.size());b.cached_poses.resize(b.cached_hulls.size());b.cache_valid.resize(b.cached_hulls.size());
        for(std::size_t h=0;h<b.cached_hulls.size();h++){
            if(selection&&!(*selection)[h])continue;const Pose& old=b.cached_poses[h];
            if(!b.cache_valid[h]||pose.p.x!=old.p.x||pose.p.y!=old.p.y||pose.p.z!=old.p.z||pose.q.w!=old.q.w||pose.q.x!=old.q.x||pose.q.y!=old.q.y||pose.q.z!=old.q.z){
                transform_hull(b.shape->hulls[h],pose.p,pose.q,b.cached_hulls[h]);
                b.cached_poses[h]=pose;b.cache_valid[h]=true;
            }
        }
        return b.cached_hulls;
    }
    // A SAT separating gap is a conservative distance lower bound. Advance
    // only by gap / a bound on surface closure, including angular sweep and
    // acceleration. Exhausted cast work returns an early recheck, never a miss.
    double cast(Pair& pair,double until) {
        stats.casts++;double t=now;
        const std::size_t columns=pair.b<0?1:bodies[pair.b].shape->hulls.size();
        std::vector<char> supported(bodies[pair.a].shape->hulls.size()*columns,0);
        if(pair.standing&&pair.supported_parts.size()==supported.size())supported=pair.supported_parts;
        pair.separating_axes.resize(supported.size());
        // A retained separating plane is a certificate for the whole interval
        // when projected translation plus angular sweep cannot close its gap.
        // This avoids repeatedly transforming every compound part into future
        // poses merely to rediscover that the same plane still separates it.
        double h=std::isfinite(until)?std::max(0.0,until-now):0;
        BodyState sa=predicted(pair.a,now),sb=pair.b<0?BodyState{}:predicted(pair.b,now);
        Vec3 relative_v=sa.v-sb.v,relative_a=bodies[pair.a].acceleration-(pair.b<0?Vec3{}:bodies[pair.b].acceleration);
        struct RotationBound {Vec3 omega;double acceleration=0;};
        auto rotation_bound=[&](int id){RotationBound bound;if(id<0||bodies[id].fixed||bodies[id].equilibrium)return bound;
            const Body& b=bodies[id];double dt=std::max(0.0,now-b.stamp);
            Vec3 phi=b.w*dt+b.alpha*(.5*dt*dt),rate=b.w+b.alpha*dt;double angle=length(phi);
            double a=.5,c=1.0/6;
            if(angle>1e-5){a=(1-std::cos(angle))/(angle*angle);c=(angle-std::sin(angle))/(angle*angle*angle);}
            bound.omega=rate+cross(phi,rate)*a+cross(phi,cross(phi,rate))*c;
            double fastest=length(rate)+length(b.alpha)*h;
            // ||d J_l(phi)/dt|| <= ||phi_dot||/2, from its integral
            // representation. Point acceleration is bounded by R (1.5 w²+a).
            bound.acceleration=b.shape->radius*(1.5*fastest*fastest+length(b.alpha));return bound;
        };
        RotationBound rotation_a=rotation_bound(pair.a),rotation_b=rotation_bound(pair.b);
        auto clear_interval=[&](Vec3 axis,double gap,const WorldHull& hull_a,const WorldHull* hull_b){
            double low=0,high=0;Vec3 ua=cross(axis,rotation_a.omega),ub=cross(axis,rotation_b.omega);
            if(dot(ua,ua)>0){low=infinity;for(Vec3 point:hull_a.vertices)low=std::min(low,dot(ua,point-sa.pose.p));}
            if(hull_b&&dot(ub,ub)>0){high=-infinity;for(Vec3 point:hull_b->vertices)high=std::max(high,dot(ub,point-sb.pose.p));}
            double v=dot(relative_v,axis)+low-high,a=dot(relative_a,axis)-rotation_a.acceleration-rotation_b.acceleration;
            double least=std::min(0.0,v*h+.5*a*h*h);
            if(a>0){double extremum=-v/a;if(extremum>0&&extremum<h)least=std::min(least,v*extremum+.5*a*extremum*extremum);}
            return gap+least>p.skin+1e-8;
        };
        const auto& start_a=placed(pair.a,now);const auto& start_b=pair.b<0?empty_hulls:placed(pair.b,now);
        for(std::size_t ha=0;ha<start_a.size();ha++)for(std::size_t hb=0;hb<columns;hb++){
            std::size_t k=ha*columns+hb;if(supported[k])continue;const WorldHull& a=start_a[ha];
            if(pair.b<0){if(clear_interval({0,0,1},a.low.z-p.ground_z,a,nullptr)){supported[k]=1;stats.separation_certificates++;}continue;}
            const WorldHull& b=start_b[hb];
            auto witness=[&](Vec3 axis){
                double norm=length(axis);if(norm<1e-12)return false;axis=axis*(1/norm);
                double low_a=infinity,high_a=-infinity,low_b=infinity,high_b=-infinity;
                for(Vec3 v:a.vertices){double x=dot(v,axis);low_a=std::min(low_a,x);high_a=std::max(high_a,x);}
                for(Vec3 v:b.vertices){double x=dot(v,axis);low_b=std::min(low_b,x);high_b=std::max(high_b,x);}
                double gap=low_a-high_b;if(low_b-high_a>gap){gap=low_b-high_a;axis=axis*(-1);}
                if(gap>p.skin+1e-8)pair.separating_axes[k]=axis;
                return clear_interval(axis,gap,a,&b);
            };
            bool clear=witness(pair.separating_axes[k]);
            if(!clear)for(int dim=0;dim<3;dim++){Vec3 axis{};component(axis,dim,1);
                double gap=component(a.low,dim)-component(b.high,dim),other=component(b.low,dim)-component(a.high,dim);
                if(other>gap){gap=other;axis=axis*(-1);}if(clear_interval(axis,gap,a,&b)){pair.separating_axes[k]=axis;clear=true;break;}}
            if(!clear)for(Vec3 axis:a.normals)if(witness(axis)){clear=true;break;}
            if(!clear)for(Vec3 axis:b.normals)if(witness(axis)){clear=true;break;}
            if(clear){supported[k]=1;stats.separation_certificates++;}
        }
        if(std::all_of(supported.begin(),supported.end(),[](char value){return value!=0;}))return infinity;
        std::vector<char> needed_a(start_a.size(),0),needed_b(columns,0);
        for(std::size_t ha=0;ha<start_a.size();ha++)for(std::size_t hb=0;hb<columns;hb++)if(!supported[ha*columns+hb]){needed_a[ha]=1;needed_b[hb]=1;}
        double relative=0;
        if(pair.b>=0){BodyState a=predicted(pair.a,now),b=predicted(pair.b,now);
            relative=length(a.v-b.v)+length(bodies[pair.a].acceleration-bodies[pair.b].acceleration)*(until-now);
            for(int id:{pair.a,pair.b})if(!bodies[id].fixed&&!bodies[id].equilibrium){const Body& x=bodies[id];relative+=x.shape->radius*(length(x.w)+length(x.alpha)*(until-x.stamp));}}
        else relative=speed_bound(pair.a,until-now);
        for(int iteration=0;iteration<48;iteration++){
            stats.cast_iterations++;const auto& hull_a=placed(pair.a,t,&needed_a);const auto& hull_b=pair.b<0?empty_hulls:placed(pair.b,t,&needed_b);
            double gap=infinity;
            if(pair.b<0){for(std::size_t ha=0;ha<hull_a.size();ha++)if(!supported[ha])gap=std::min(gap,hull_a[ha].low.z-p.ground_z);}
            else for(std::size_t ha=0;ha<hull_a.size();ha++)for(std::size_t hb=0;hb<hull_b.size();hb++){
                if(supported[ha*columns+hb])continue;
                const WorldHull& a=hull_a[ha];const WorldHull& b=hull_b[hb];
                double boxgap=std::max({a.low.x-b.high.x,b.low.x-a.high.x,a.low.y-b.high.y,b.low.y-a.high.y,a.low.z-b.high.z,b.low.z-a.high.z});
                if(boxgap>p.skin+1e-8){gap=std::min(gap,boxgap);continue;}
                manifold.clear();double separation;stats.narrowphase++;collide_hulls(a,b,p.skin+1e-8,scratch,manifold,separation);
                if(manifold.empty()&&!std::isfinite(separation)){
                    separation=exhaustive_separation(a,b,pair.separating_axes[ha*columns+hb]);
                    if(separation<=p.skin+1e-8){std::ostringstream error;error<<"convex contact lacks manifold; exhaustive gap="<<separation<<" bodies="<<pair.a<<","<<pair.b<<" parts="<<ha<<","<<hb;throw std::runtime_error(error.str());}
                }
                if(!manifold.empty())return t;
                gap=std::min(gap,separation);
            }
            if(gap<=p.skin+1e-8)return t;
            if(relative<1e-14)return infinity;
            double dt=.9*(gap-p.skin)/relative;
            if(!std::isfinite(dt)||t+dt>=until)return infinity;
            if(t+dt==t)return t;
            t+=dt;
        }
        return t;
    }
    int ensure_pair(int a,int b) {
        if(a==b)return -1;
        if(b>=0&&(bodies[a].fixed||(!bodies[b].fixed&&b<a)))std::swap(a,b);
        if(bodies[a].fixed)return -1;
        std::uint64_t key=(std::uint64_t(std::uint32_t(a))<<32)|std::uint32_t(b+1);
        auto found=pair_index.find(key);if(found!=pair_index.end())return found->second;
        int index=int(pairs.size());Pair pair;pair.a=a;pair.b=b;pairs.push_back(pair);pair_index[key]=index;
        bodies[a].pairs.push_back(index);if(b>=0)bodies[b].pairs.push_back(index);return index;
    }
    void plan_pair(int index) {
        if(index<0)return;Pair& pair=pairs[index];Body& a=bodies[pair.a];
        if(!a.alive||(pair.b>=0&&!bodies[pair.b].alive))return;
        std::uint64_t rb=pair.b<0?0:bodies[pair.b].revision;
        if(pair.revision_a==a.revision&&pair.revision_b==rb&&pair.deadline>=now){stats.pending_reuses++;return;}
        pair.revision_a=a.revision;pair.revision_b=rb;pair.clock++;
        if((pair.b>=0&&!intersects(a.bounds,bodies[pair.b].bounds))||(pair.b<0&&a.bounds.low.z>p.ground_z+p.skin)){
            pair.deadline=infinity;return;
        }
        double until=a.envelope_until;if(pair.b>=0)until=std::min(until,bodies[pair.b].envelope_until);
        pair.deadline=cast(pair,until);
        if(std::isfinite(pair.deadline))events.push({std::max(now,pair.deadline),serial++,pair.clock,index,1});
    }
    void schedule_body(int id,double delay) {
        Body& b=bodies[id];b.clock++;
        if(!b.fixed&&!b.equilibrium&&b.alive)events.push({now+std::max(delay,1e-9),serial++,b.clock,id,0});
    }
    double support_horizon() const {
        return (std::floor((now+time_epsilon)/p.contact_step)+1)*p.contact_step-now;
    }
    void update_envelopes(const std::vector<int>& group,bool has_contacts) {
        for(int id:group){Body& b=bodies[id];if(!b.alive||b.fixed)continue;
            double horizon=p.lookahead;
            // Rotation is integrated with a midpoint exponential and exact
            // angular-momentum transport. Bound its approximation interval.
            if(length(b.w)>1e-8)horizon=std::min(horizon,p.angular_step/length(b.w));
            if(length(b.alpha)>1e-8)horizon=std::min(horizon,std::sqrt(2*p.angular_step/length(b.alpha)));
            if(has_contacts){
                // Changing contacts share a support horizon. An arrival may
                // change momenta immediately without postponing this clock.
                horizon=std::min(horizon,support_horizon());
            }
            b.envelope_until=b.equilibrium?infinity:now+horizon;
            Bounds box=envelope(id,horizon);b.bounds=box;if(b.leaf>=0)tree.erase(b.leaf);b.leaf=tree.insert(id,box);
            b.revision++;schedule_body(id,horizon);
        }
        std::vector<int> changed;
        for(int id:group){Body& b=bodies[id];if(!b.alive||b.fixed)continue;
            changed.insert(changed.end(),b.pairs.begin(),b.pairs.end());
            double horizon=b.equilibrium?0:b.envelope_until-now;Bounds box=envelope(id,horizon);stats.broadphase_queries++;
            tree.query(box,[&](int other){if(other!=id&&bodies[other].alive){int pair=ensure_pair(id,other);if(pair>=0)changed.push_back(pair);}});
            changed.push_back(ensure_pair(id,-1));
        }
        std::sort(changed.begin(),changed.end());changed.erase(std::unique(changed.begin(),changed.end()),changed.end());
        for(int index:changed)plan_pair(index);
    }
    Vec3 point_velocity(int id,Vec3 r) const {return id<0?Vec3{}:bodies[id].v+cross(bodies[id].w,r);}
    Vec3 response(int id,Vec3 r,Vec3 f) const {return id<0?Vec3{}:f*bodies[id].inv_mass+cross(mul(bodies[id].inverse,cross(r,f)),r);}
    Vec3 point_acceleration(int id,Vec3 r) const {if(id<0)return {};const Body& b=bodies[id];return b.acceleration+cross(b.alpha,r)+cross(b.w,cross(b.w,r));}
    void impulse(int id,Vec3 r,Vec3 j) {
        if(id<0||bodies[id].fixed)return;Body& b=bodies[id];b.v=b.v+j*b.inv_mass;b.angular_momentum=b.angular_momentum+cross(r,j);b.w=b.w+mul(b.inverse,cross(r,j));
    }
    void force(int id,Vec3 r,Vec3 f) {
        if(id<0||bodies[id].fixed)return;Body& b=bodies[id];b.acceleration=b.acceleration+f*b.inv_mass;b.torque=b.torque+cross(r,f);b.alpha=mul(b.inverse,b.torque-cross(b.w,b.angular_momentum));
    }
    bool refresh_pair(int index,bool force_refresh=false) {
        Pair& pair=pairs[index];const Body& a=bodies[pair.a];
        if(!a.alive||(pair.b>=0&&!bodies[pair.b].alive)){pair.standing=false;pair.atoms.clear();return false;}
        const Pose pose_a=predicted(pair.a,now).pose,pose_b=pair.b<0?Pose{}:predicted(pair.b,now).pose;
        double movement=length(pose_a.p-pair.pose_a.p)+a.shape->radius*orientation_distance(pose_a.q,pair.pose_a.q);
        if(pair.b>=0)movement+=length(pose_b.p-pair.pose_b.p)+bodies[pair.b].shape->radius*orientation_distance(pose_b.q,pair.pose_b.q);
        if(pair.standing&&!force_refresh&&movement<1e-9){stats.manifold_reuses++;return true;}
        const auto& hull_a=placed(pair.a,now);const auto& hull_b=pair.b<0?empty_hulls:placed(pair.b,now);
        std::vector<Atomic> next;double margin=pair.standing?p.release_gap:p.skin+2e-8;
        const std::size_t columns=pair.b<0?1:hull_b.size();
        pair.supported_parts.assign(hull_a.size()*columns,0);
        for(std::size_t ha=0;ha<hull_a.size();ha++){
            std::size_t count=pair.b<0?1:hull_b.size();
            for(std::size_t hb=0;hb<count;hb++){
                manifold.clear();stats.narrowphase++;
                if(pair.b<0)collide_ground(hull_a[ha],p.ground_z,margin,manifold);
                else {if(!bounds_overlap(hull_a[ha],hull_b[hb],margin))continue;double separation;collide_hulls(hull_a[ha],hull_b[hb],margin,scratch,manifold,separation);}
                // Deduplicating a witness shared by adjacent convex sectors
                // must not turn the second sector back into a pending impact.
                if(!manifold.empty())pair.supported_parts[ha*columns+hb]=1;
                for(const ManifoldPoint& m:manifold){
                    Atomic atom;atom.local_a=rotate(conjugate(pose_a.q),m.point_a-pose_a.p);
                    atom.local_b=pair.b<0?m.point_b:rotate(conjugate(pose_b.q),m.point_b-pose_b.p);
                    atom.normal_a=rotate(conjugate(pose_a.q),m.normal);atom.hull_a=int(ha);atom.hull_b=int(hb);
                    double best=1e-6;const Atomic* match=nullptr;
                    for(const Atomic& old:pair.atoms){double d=dot(atom.local_a-old.local_a,atom.local_a-old.local_a);
                        if(old.hull_a==atom.hull_a&&old.hull_b==atom.hull_b&&dot(old.normal_a,atom.normal_a)>.98&&d<best){best=d;match=&old;}}
                    if(match){atom.normal_force=match->normal_force;atom.tangent_force=match->tangent_force;}
                    // Shared sector boundaries can produce coincident witnesses.
                    bool duplicate=false;for(const Atomic& other:next)if(length(atom.local_a-other.local_a)<1e-7&&dot(atom.normal_a,other.normal_a)>.9999){duplicate=true;break;}
                    if(!duplicate)next.push_back(atom);
                }
            }
        }
        pair.pose_a=pose_a;pair.pose_b=pose_b;pair.atoms=std::move(next);pair.standing=!pair.atoms.empty();pair.clock++;pair.deadline=infinity;
        pair.revision_a=pair.revision_b=0;return pair.standing;
    }
    std::vector<int> connected(int start) {
        std::vector<int> group{start};std::uint64_t visit=++visit_serial;bodies[start].visit=visit;
        for(std::size_t k=0;k<group.size();k++)for(int index:bodies[group[k]].pairs){Pair& pair=pairs[index];if(!pair.standing)continue;
            int other=pair.a==group[k]?pair.b:pair.a;
            if(other>=0&&bodies[other].alive&&!bodies[other].fixed&&bodies[other].visit!=visit){bodies[other].visit=visit;group.push_back(other);}}
        std::sort(group.begin(),group.end());return group;
    }
    std::vector<Row> rows_for_pairs(const std::vector<int>& indices) {
        std::vector<Row> rows;
        for(int index:indices){Pair& pair=pairs[index];if(!refresh_pair(index))continue;
            Body& a=bodies[pair.a];for(std::size_t k=0;k<pair.atoms.size();k++){
                const Atomic& atom=pair.atoms[k];Row row;row.pair=index;row.atom=int(k);row.a=pair.a;row.b=pair.b;
                row.ra=rotate(a.pose.q,atom.local_a);row.pa=a.pose.p+row.ra;row.n=rotate(a.pose.q,atom.normal_a);
                if(pair.b<0)row.pb=atom.local_b;else {row.rb=rotate(bodies[pair.b].pose.q,atom.local_b);row.pb=bodies[pair.b].pose.p+row.rb;}
                row.gap=dot(row.pa-row.pb,row.n);row.mu=std::sqrt(a.shape->friction*(pair.b<0?p.ground_friction:bodies[pair.b].shape->friction));
                Vec3 shared_point=(row.pa+row.pb)*.5;
                row.ra=shared_point-a.pose.p;if(pair.b>=0)row.rb=shared_point-bodies[pair.b].pose.p;
                row.t1=normalized(cross(row.n,std::abs(row.n.x)<.7?Vec3{1,0,0}:Vec3{0,1,0}));row.t2=cross(row.n,row.t1);
                Vec3 rn=response(row.a,row.ra,row.n)+response(row.b,row.rb,row.n);
                Vec3 rt1=response(row.a,row.ra,row.t1)+response(row.b,row.rb,row.t1),rt2=response(row.a,row.ra,row.t2)+response(row.b,row.rb,row.t2);
                row.kn=dot(row.n,rn);row.k11=dot(row.t1,rt1);row.k12=dot(row.t1,rt2);row.k22=dot(row.t2,rt2);
                Vec3 relative=point_velocity(row.a,row.ra)-point_velocity(row.b,row.rb);double vn=dot(relative,row.n);
                double restitution=std::max(a.shape->restitution,pair.b<0?0:bodies[pair.b].shape->restitution);
                row.bounce=vn< -p.restitution_threshold?-restitution*vn:0;
                if(row.kn>0)rows.push_back(row);
            }
        }
        return rows;
    }
    std::vector<Row> rows_for(const std::vector<int>& group) {
        std::vector<int> indices;for(int id:group)for(int index:bodies[id].pairs)if(pairs[index].standing)indices.push_back(index);
        std::sort(indices.begin(),indices.end());indices.erase(std::unique(indices.begin(),indices.end()),indices.end());
        return rows_for_pairs(indices);
    }
    Vec3 tangential_step(const Row& row,Vec3 value) {
        double u=dot(value,row.t1),v=dot(value,row.t2),det=row.k11*row.k22-row.k12*row.k12;
        if(det<=1e-30)return {};
        return row.t1*((row.k22*u-row.k12*v)/det)+row.t2*((row.k11*v-row.k12*u)/det);
    }
    double impact_atomic(Row& row,bool support=false) {
        Vec3 velocity=point_velocity(row.a,row.ra)-point_velocity(row.b,row.rb);
        double next_n=std::max(0.0,row.impulse_n+(row.bounce-dot(velocity,row.n))/row.kn);
        Vec3 normal_change=row.n*(next_n-row.impulse_n);
        Vec3 after_normal=velocity+response(row.a,row.ra,normal_change)+response(row.b,row.rb,normal_change);
        Vec3 next_t=row.impulse_t-tangential_step(row,after_normal);
        double limit=row.mu*next_n,norm=length(next_t);if(norm>limit)next_t=next_t*(limit/norm);
        Vec3 change=normal_change+next_t-row.impulse_t;
        double fraction=1;
        if(row.bounce==0&&!support){
            bool admissible=false;
            for(int back=0;back<24;back++){
                Vec3 delta=change*fraction;
                Vec3 response_v=response(row.a,row.ra,delta)+response(row.b,row.rb,delta);
                if(dot(delta,velocity)+.5*dot(delta,response_v)<=1e-18){admissible=true;break;}
                fraction*=.5;
            }
            if(!admissible)fraction=0;
        }
        // Interpolate the entire impulse, not only its tangent. The Coulomb
        // cone is convex, so this preserves feasibility while guarding energy.
        row.impulse_n+=(next_n-row.impulse_n)*fraction;
        row.impulse_t=row.impulse_t+(next_t-row.impulse_t)*fraction;
        change=change*fraction;
        impulse(row.a,row.ra,change);impulse(row.b,row.rb,change*(-1));
        if(support)stats.force_atomics++;else stats.impact_atomics++;
        return length(response(row.a,row.ra,change))+length(response(row.b,row.rb,change));
    }
    template<class AtomicUpdate> bool propagate(std::vector<Row>& rows,double tolerance,AtomicUpdate update) {
        std::unordered_map<int,std::vector<int>> adjacency;
        for(std::size_t i=0;i<rows.size();i++){adjacency[rows[i].a].push_back(int(i));if(rows[i].b>=0&&!bodies[rows[i].b].fixed)adjacency[rows[i].b].push_back(int(i));}
        std::deque<int> queue;std::vector<char> queued(rows.size(),1);for(std::size_t i=0;i<rows.size();i++)queue.push_back(int(i));
        int updates=0;
        while(!queue.empty()&&updates<p.max_atomic_updates){int index=queue.front();queue.pop_front();queued[index]=0;Row& row=rows[index];updates++;
            if(update(row)>tolerance){for(int id:{row.a,row.b})if(id>=0&&!bodies[id].fixed)for(int other:adjacency[id])if(!queued[other]){queued[other]=1;queue.push_back(other);}}
        }
        return queue.empty();
    }
    void correct_positions(std::vector<Row>& rows) {
        for(int pass=0;pass<6;pass++){
            bool changed=false;
            for(Row& row:rows){const Atomic& atom=pairs[row.pair].atoms[row.atom];Body& a=bodies[row.a];
                Vec3 ra=rotate(a.pose.q,atom.local_a),pa=a.pose.p+ra,n=rotate(a.pose.q,atom.normal_a),rb{},pb=atom.local_b;
                if(row.b>=0){rb=rotate(bodies[row.b].pose.q,atom.local_b);pb=bodies[row.b].pose.p+rb;}
                double gap=dot(pa-pb,n);if(gap>=-p.skin*.25)continue;
                double k=dot(n,response(row.a,ra,n)+response(row.b,rb,n));double amount=.8*(-gap-p.skin*.25)/k;
                Vec3 j=n*amount;
                for(int side=0;side<2;side++){int id=side?row.b:row.a;if(id<0||bodies[id].fixed)continue;Body& b=bodies[id];Vec3 r=side?rb:ra,delta=side?j*(-1):j;
                    b.pose.p=b.pose.p+delta*b.inv_mass;b.pose.q=turned(b.pose.q,mul(b.inverse,cross(r,delta)));tensors(b);}
                changed=true;stats.position_atomics++;
            }
            if(!changed)break;
        }
    }
    void solve_group(int seed) {
        std::vector<int> group=connected(seed);stats.group_solves++;stats.largest_group=std::max(stats.largest_group,int(group.size()));
        for(int id:group){materialize(id);if(bodies[id].equilibrium)stats.wakes++;bodies[id].equilibrium=false;}
        std::vector<Row> rows=rows_for(group);
        bool impact_converged=true;
        double approach_speed=0;for(int id:group)approach_speed=std::max(approach_speed,length(bodies[id].v)+bodies[id].shape->radius*length(bodies[id].w));
        double impact_tolerance=std::max(p.velocity_tolerance,std::min(p.moving_velocity_tolerance,approach_speed*1e-4));
        if(!rows.empty()){
            correct_positions(rows);
            // Position projection changes lever arms and effective masses.
            rows=rows_for(group);
            impact_converged=propagate(rows,impact_tolerance,[&](Row& r){return impact_atomic(r);});
            for(const Row& row:rows){
                double vn=dot(point_velocity(row.a,row.ra)-point_velocity(row.b,row.rb),row.n);
                if(vn<row.bounce-4*impact_tolerance)impact_converged=false;
            }
            if(!impact_converged)stats.unconverged_impacts++;
        }
        double horizon=support_horizon(),max_speed=0;
        for(int id:group){const Body& b=bodies[id];max_speed=std::max(max_speed,length(b.v)+b.shape->radius*length(b.w));}
        struct SavedMotion { Vec3 v,w,momentum; };
        std::vector<SavedMotion> saved;saved.reserve(group.size());
        for(int id:group){const Body& b=bodies[id];saved.push_back({b.v,b.w,b.angular_momentum});}
        // Support is a finite-horizon momentum exchange. A near-stationary
        // group first tests a genuinely static field, independently of its
        // tiny remaining velocity. If that certificate fails, use its actual
        // motion when constructing the next interval's force field.
        auto support=[&](bool static_trial){
            for(std::size_t k=0;k<group.size();k++){Body& b=bodies[group[k]];
                b.v=static_trial?Vec3{}:saved[k].v;b.angular_momentum=static_trial?Vec3{}:saved[k].momentum;
                b.acceleration=p.gravity;b.torque={};tensors(b);
                b.v=b.v+p.gravity*horizon;b.w=b.w+b.alpha*horizon;
            }
            for(Row& row:rows){Atomic& atom=pairs[row.pair].atoms[row.atom];
                row.impulse_n=atom.normal_force*horizon;
                row.impulse_t=(atom.tangent_force-row.n*dot(atom.tangent_force,row.n))*horizon;
                double limit=row.mu*row.impulse_n,norm=length(row.impulse_t);
                if(norm>limit)row.impulse_t=row.impulse_t*(limit/norm);
                row.bounce=-std::max(0.0,row.gap-p.skin)/horizon;
                Vec3 j=row.n*row.impulse_n+row.impulse_t;impulse(row.a,row.ra,j);impulse(row.b,row.rb,j*(-1));
            }
            double tolerance=static_trial?p.force_tolerance*horizon:std::max(p.force_tolerance*horizon,std::min({p.moving_velocity_tolerance,p.skin*.05/horizon,max_speed*1e-3}));
            bool converged=propagate(rows,tolerance,[&](Row& r){return impact_atomic(r,true);});
            if(!converged)stats.unconverged_forces++;
            for(std::size_t k=0;k<group.size();k++){Body& b=bodies[group[k]];b.v=saved[k].v;b.w=saved[k].w;b.angular_momentum=saved[k].momentum;b.acceleration=p.gravity;b.torque={};tensors(b);}
            for(Row& row:rows){Atomic& atom=pairs[row.pair].atoms[row.atom];atom.normal_force=row.impulse_n/horizon;atom.tangent_force=row.impulse_t*(1/horizon);
                Vec3 f=row.n*atom.normal_force+atom.tangent_force;force(row.a,row.ra,f);force(row.b,row.rb,f*(-1));}
            return converged;
        };
        bool static_trial=p.certify_equilibrium&&max_speed<=p.equilibrium_speed&&!rows.empty();
        bool force_converged=support(static_trial);
        bool quiet=!rows.empty()&&impact_converged&&force_converged&&p.certify_equilibrium;
        for(int id:group){Body& b=bodies[id];double speed=length(b.v)+b.shape->radius*length(b.w);
            double residual=length(b.acceleration)+b.shape->radius*length(b.alpha);stats.max_force_residual=std::max(stats.max_force_residual,residual);
            if(speed>p.equilibrium_speed||residual>p.equilibrium_acceleration)quiet=false;}
        for(const Row& row:rows){double load=pairs[row.pair].atoms[row.atom].normal_force;
            if(row.gap< -p.skin||(load>1e-8&&row.gap>p.skin*2))quiet=false;}
        if(static_trial&&!quiet)support(false);
        if(quiet)for(int id:group){Body& b=bodies[id];
            stats.discarded_equilibrium_energy+=.5*b.shape->mass*dot(b.v,b.v)+.5*dot(b.w,b.angular_momentum);
            b.v={};b.w={};b.angular_momentum={};b.acceleration={};b.alpha={};b.torque={};b.equilibrium=true;stats.equilibrium_certificates++;}
        update_envelopes(group,!rows.empty());
    }
    void handle_pair(int index) {
        Pair& pair=pairs[index];materialize(pair.a);if(pair.b>=0&&!bodies[pair.b].fixed)materialize(pair.b);
        if(refresh_pair(index,true))transmit_arrivals({index});
        else {pair.revision_a=pair.revision_b=0;pair.deadline=-infinity;
            // A conservative cast can end before actual contact. Preserve the
            // certified advance and compute the next deadline from here.
            plan_pair(index);
        }
    }
    void transmit_arrivals(const std::vector<int>& initial) {
        // Traverse the contact graph only while a changed impulse requires it.
        // Retained support fields remain in force until their common horizon;
        // thousands of arrivals therefore do not each rebuild a whole pile.
        std::map<int,std::vector<Row>> active;
        std::deque<int> queue(initial.begin(),initial.end());std::unordered_map<int,bool> queued;
        for(int index:initial)queued[index]=true;std::vector<int> touched;
        std::uint64_t visit=++visit_serial;
        auto touch=[&](int id){if(id<0||bodies[id].fixed||bodies[id].visit==visit)return;
            Body& b=bodies[id];materialize(id);b.visit=visit;
            if(b.equilibrium){stats.wakes++;b.equilibrium=false;}
            touched.push_back(id);
        };
        int updates=0;
        while(!queue.empty()&&updates<p.max_atomic_updates){
            int index=queue.front();queue.pop_front();queued[index]=false;Pair& pair=pairs[index];
            auto found=active.find(index);
            if(found==active.end()){
                touch(pair.a);touch(pair.b);
                found=active.emplace(index,rows_for_pairs({index})).first;
            }
            double change=0;
            for(Row& row:found->second){change=std::max(change,impact_atomic(row));updates++;}
            double speed=0;for(int id:{pair.a,pair.b})if(id>=0&&!bodies[id].fixed)
                speed=std::max(speed,length(bodies[id].v)+bodies[id].shape->radius*length(bodies[id].w));
            double tolerance=std::max(p.velocity_tolerance,std::min(p.moving_velocity_tolerance,speed*1e-4));
            if(change>tolerance)for(int id:{pair.a,pair.b})if(id>=0&&!bodies[id].fixed)
                for(int next:bodies[id].pairs)if(pairs[next].standing&&!queued[next]){queued[next]=true;queue.push_back(next);}
        }
        if(!queue.empty())stats.unconverged_impacts++;
        stats.group_solves++;stats.largest_group=std::max(stats.largest_group,int(touched.size()));
        for(int id:touched)tensors(bodies[id]);
        update_envelopes(touched,true);
    }
    void handle_body(int id) {
        double tick=std::round(now/p.contact_step)*p.contact_step;
        if(std::abs(now-tick)<=time_epsilon){solve_group(id);return;}
        materialize(id);
        std::vector<int> standing;
        for(int index:bodies[id].pairs)if(pairs[index].standing)standing.push_back(index);
        // An orientation/envelope deadline belongs to this moving body. It
        // refreshes local witnesses and transmits impacts, without turning the
        // fastest spinner into the support clock for its entire contact graph.
        if(standing.empty()){
            bodies[id].acceleration=p.gravity;bodies[id].torque={};tensors(bodies[id]);
            update_envelopes({id},false);
        }
        else transmit_arrivals(standing);
    }
    int add(const Shape& shape,Pose pose,Vec3 v,Vec3 w,bool fixed) {
        if(!(shape.mass>0)||shape.hulls.empty())throw std::invalid_argument("body needs cooked finite-volume geometry");
        Body b;b.shape=std::make_shared<Shape>(shape);b.pose=pose;b.v=v;b.w=w;b.fixed=fixed;b.stamp=now;
        for(int k=0;k<9;k++)b.inertia_local.m[k]=shape.inertia[k];
        if(!fixed){b.inv_mass=1/shape.mass;b.inverse_local=inverse(b.inertia_local);b.acceleration=p.gravity;}
        b.inertia=rotated(to_matrix(pose.q),b.inertia_local);b.angular_momentum=mul(b.inertia,w);tensors(b);
        int id=int(bodies.size());bodies.push_back(b);
        if(fixed){bodies[id].envelope_until=infinity;bodies[id].bounds=envelope(id,0);bodies[id].leaf=tree.insert(id,bodies[id].bounds);
            Bounds box=envelope(id,0);std::vector<int> near;tree.query(box,[&](int other){if(other!=id&&!bodies[other].fixed)near.push_back(other);});
            for(int other:near){int pair=ensure_pair(other,id);plan_pair(pair);}}
        else update_envelopes({id},false);
        return id;
    }
};

World::World(Parameters p):impl(new Impl){
    if(!(p.lookahead>0&&p.contact_step>0&&p.angular_step>0&&p.skin>0&&p.release_gap>=p.skin))throw std::invalid_argument("invalid obligation tolerances");impl->p=p;
}
World::~World()=default;
BodyId World::add_body(const Shape& s,Pose p,Vec3 v,Vec3 w){return impl->add(s,p,v,w,false);}
BodyId World::add_static_body(const Shape& s,Pose p){return impl->add(s,p,{},{},true);}
void World::advance(double seconds){
    if(!std::isfinite(seconds)||seconds<0)throw std::invalid_argument("advance duration must be finite and nonnegative");
    double end=impl->now+seconds;int count=0;
    while(!impl->events.empty()&&impl->events.top().t<=end+time_epsilon){Event event=impl->events.top();impl->events.pop();
        bool valid=event.kind?impl->pairs[event.id].clock==event.clock:impl->bodies[event.id].clock==event.clock;
        if(!valid){impl->stats.stale_events++;continue;}
        if(++count>impl->p.max_events_per_advance){std::ostringstream message;message.precision(14);message<<"obligation event budget exhausted at t="<<impl->now<<" kind="<<event.kind<<" id="<<event.id;
            if(!event.kind){const Body& b=impl->bodies[event.id];message<<" v="<<length(b.v)<<" w="<<length(b.w)<<" a="<<length(b.acceleration)<<" alpha="<<length(b.alpha)<<" horizon="<<b.envelope_until-impl->now;}
            else {const Pair& pair=impl->pairs[event.id];message<<" bodies="<<pair.a<<","<<pair.b<<" standing="<<pair.standing<<" atoms="<<pair.atoms.size();
                const auto& a=impl->placed(pair.a,impl->now);const auto& b=pair.b<0?impl->empty_hulls:impl->placed(pair.b,impl->now);
                for(std::size_t ha=0;ha<a.size();ha++)for(std::size_t hb=0;hb<(pair.b<0?1:b.size());hb++){
                    std::size_t k=ha*(pair.b<0?1:b.size())+hb;if(pair.standing&&k<pair.supported_parts.size()&&pair.supported_parts[k])continue;
                    impl->manifold.clear();double gap=a[ha].low.z-impl->p.ground_z;
                    if(pair.b>=0)collide_hulls(a[ha],b[hb],impl->p.skin+1e-8,impl->scratch,impl->manifold,gap);
                    if(gap<=impl->p.skin+1e-7)message<<" part="<<ha<<","<<hb<<" gap="<<gap<<" witnesses="<<impl->manifold.size();
                }
            }
            throw std::runtime_error(message.str()+"; simulation was not advanced past unresolved events");}
        impl->now=std::max(impl->now,event.t);impl->stats.events++;
        if(event.kind)impl->handle_pair(event.id);else if(impl->bodies[event.id].alive)impl->handle_body(event.id);
    }
    impl->now=end;
}
double World::time()const{return impl->now;}
BodyState World::state(BodyId id)const{return impl->predicted(id,impl->now);}
Statistics World::statistics()const{return impl->stats;}
double World::kinetic_energy()const{double e=0;for(std::size_t i=0;i<impl->bodies.size();i++){const Body& b=impl->bodies[i];if(!b.alive||b.fixed)continue;BodyState s=state(int(i));e+=.5*b.shape->mass*dot(s.v,s.v)+.5*dot(s.w,mul(rotated(to_matrix(s.pose.q),b.inertia_local),s.w));}return e;}
double World::equilibrium_residual(BodyId id)const {
    const Body& body=impl->bodies[id];if(!body.equilibrium)return length(body.acceleration)+body.shape->radius*length(body.alpha);
    Vec3 force=impl->p.gravity*body.shape->mass,torque{};
    for(int index:body.pairs){const Pair& pair=impl->pairs[index];if(!pair.standing)continue;for(const Atomic& atom:pair.atoms){
        Vec3 n=rotate(impl->bodies[pair.a].pose.q,atom.normal_a),f=n*atom.normal_force+atom.tangent_force;
        Vec3 pa=impl->bodies[pair.a].pose.p+rotate(impl->bodies[pair.a].pose.q,atom.local_a),pb=atom.local_b;
        if(pair.b>=0)pb=impl->bodies[pair.b].pose.p+rotate(impl->bodies[pair.b].pose.q,atom.local_b);
        Vec3 r=(pa+pb)*.5-body.pose.p;if(pair.a!=id)f=f*(-1);force=force+f;torque=torque+cross(r,f);}}
    return length(force)*body.inv_mass+body.shape->radius*length(mul(body.inverse,torque));
}
std::vector<Contact> World::contacts()const{
    std::vector<Contact> out;for(const Pair& pair:impl->pairs)if(pair.standing)for(const Atomic& atom:pair.atoms){
        BodyState a=state(pair.a);Vec3 pa=a.pose.p+rotate(a.pose.q,atom.local_a),pb=atom.local_b,n=rotate(a.pose.q,atom.normal_a);
        if(pair.b>=0){BodyState b=state(pair.b);pb=b.pose.p+rotate(b.pose.q,atom.local_b);}
        out.push_back({pair.a,pair.b,(pa+pb)*.5,n,dot(pa-pb,n),atom.normal_force*impl->p.contact_step});
    }
    return out;
}
void World::apply_impulse(BodyId id,Vec3 j,Vec3 point){
    impl->materialize(id);Body& b=impl->bodies.at(id);if(!b.alive||b.fixed)return;
    b.equilibrium=false;impl->impulse(id,point-b.pose.p,j);impl->solve_group(id);
}
void World::set_pose(BodyId id,Pose pose){
    Body& b=impl->bodies.at(id);impl->materialize(id);std::vector<int> affected=impl->connected(id);
    b.pose=pose;b.v={};b.w={};b.angular_momentum={};b.stamp=impl->now;b.equilibrium=false;impl->tensors(b);b.revision++;
    for(int index:b.pairs){impl->pairs[index].standing=false;impl->pairs[index].atoms.clear();impl->pairs[index].clock++;}
    if(b.leaf>=0)impl->tree.erase(b.leaf);b.bounds=impl->envelope(id,0);b.leaf=impl->tree.insert(id,b.bounds);
    for(int other:affected)if(!impl->bodies[other].fixed)impl->solve_group(other);
    if(b.fixed){std::vector<int> near;impl->tree.query(impl->envelope(id,0),[&](int other){if(other!=id&&!impl->bodies[other].fixed)near.push_back(other);});for(int other:near)impl->plan_pair(impl->ensure_pair(other,id));}
}
void World::remove_body(BodyId id){
    Body& b=impl->bodies.at(id);if(!b.alive)return;
    std::vector<int> affected=impl->connected(id);if(b.fixed)for(int index:b.pairs){const Pair& pair=impl->pairs[index];if(pair.standing)affected.push_back(pair.a);}
    b.alive=false;b.clock++;if(b.leaf>=0){impl->tree.erase(b.leaf);b.leaf=-1;}
    for(int index:b.pairs){impl->pairs[index].standing=false;impl->pairs[index].atoms.clear();impl->pairs[index].clock++;}
    std::sort(affected.begin(),affected.end());affected.erase(std::unique(affected.begin(),affected.end()),affected.end());
    for(int other:affected)if(other!=id&&impl->bodies[other].alive&&!impl->bodies[other].fixed)impl->solve_group(other);
}
}
