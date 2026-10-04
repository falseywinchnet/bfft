#include "engine.hpp"
#include "broadphase.hpp"
#include "collide.hpp"
#include "threadpool_atomic_fast.hpp"
#include <atomic>
#include <chrono>
#include <exception>
#include <thread>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace tape {
namespace {
constexpr double inf=std::numeric_limits<double>::infinity();
Quat turn(Quat q,Vec3 r){double a=length(r);if(a<1e-14)return q;double s=std::sin(a*.5)/a;
    Quat v=multiply({std::cos(a*.5),r.x*s,r.y*s,r.z*s},q);double n=std::sqrt(v.w*v.w+v.x*v.x+v.y*v.y+v.z*v.z);return {v.w/n,v.x/n,v.y/n,v.z/n};}
bool contains(Bounds a,Bounds b){return a.low.x<=b.low.x&&a.low.y<=b.low.y&&a.low.z<=b.low.z&&a.high.x>=b.high.x&&a.high.y>=b.high.y&&a.high.z>=b.high.z;}
// Any separating direction is a valid rejection witness. The centre direction
// cheaply rejects diagonally separated compound pieces before full feature SAT.
bool centre_separates(const WorldHull& a,const WorldHull& b,double margin){
    Vec3 axis=a.centroid-b.centroid;double norm2=dot(axis,axis);if(norm2<1e-24)return false;
    double low=inf,high=-inf;
    for(Vec3 v:a.vertices)low=std::min(low,dot(v,axis));
    for(Vec3 v:b.vertices)high=std::max(high,dot(v,axis));
    return low-high>(margin+1e-10)*std::sqrt(norm2);
}
struct alignas(128) BodyLock {std::atomic_flag flag;};
struct Body {
    std::shared_ptr<const Shape> shape;Pose pose;Vec3 v,w,L;
    Mat3 local,inverse_local,inverse;
    double inverse_mass=0;bool fixed=false,alive=true;
    int leaf=-1,place=-1;
    std::uint64_t velocity_revision=0;
    Bounds bounds{},fat{};
    std::vector<WorldHull> hulls;
    std::vector<int> contacts,groups;
    std::unique_ptr<BodyLock> lock=std::make_unique<BodyLock>();
};
struct Memory {Vec3 a,b,n,tangent;double normal=0;};
struct Relation {std::vector<Memory> points;double dt=0;std::uint64_t seen=0;};
struct Row {
    int a,b;Memory* memory;
    Vec3 n,ra,rb,t1,t2,tangent,response_n;
    double change=0;
    double normal=0,kn=0,k11=0,k12=0,k22=0,mu=0,target=0;
    double inverse_kn=0,inverse11=0,inverse12=0,inverse22=0;
};
}
struct World::Impl {
    Parameters p;double now=0;Statistics stats;
    fileman::worker::atomic_thread_pool pool;
    explicit Impl(Parameters parameters):p(parameters),pool(parameters.workers){}
    struct Task {Impl* world;int group;bool reverse;};
    struct Group {int first,last;std::uint64_t exchanges=0,seen_a=0,seen_b=0,skips=0;bool quiet=false;};
    std::vector<Group> groups;
    std::vector<Task> arguments;std::vector<threadpool_task_t> tasks;
    std::vector<Body> bodies;
    std::vector<int> tape;
    std::vector<Row> rows;
    std::unordered_map<std::uint64_t,Relation> relations;
    struct PairWork {Impl* world=nullptr;int a=0,b=0;double dt=0;
        std::vector<ManifoldPoint> points;std::vector<Row> rows;
        std::vector<Memory> next_points;
        std::uint64_t narrow=0,key=0;std::exception_ptr error;
        Relation fresh;Relation* relation=nullptr;
    };
    std::vector<PairWork> pairs;std::vector<threadpool_task_t> collision_tasks;
    Tree space;


    void tensor(Body& b){b.inverse=rotated(to_matrix(b.pose.q),b.inverse_local);b.w=mul(b.inverse,b.L);}
    Vec3 velocity(int id,Vec3 r)const{return id<0?Vec3{}:bodies[id].v+cross(bodies[id].w,r);}
    Vec3 response(int id,Vec3 r,Vec3 j)const{if(id<0)return {};const Body& b=bodies[id];return j*b.inverse_mass+cross(mul(b.inverse,cross(r,j)),r);}
    double impulse(int id,Vec3 r,Vec3 j){if(id<0||bodies[id].fixed)return 0;Body& b=bodies[id];
        b.velocity_revision++;Vec3 dv=j*b.inverse_mass,dw=mul(b.inverse,cross(r,j));b.v=b.v+dv;b.w=b.w+dw;b.L=b.L+cross(r,j);
        return dot(dv,dv)+b.shape->radius*b.shape->radius*dot(dw,dw);
    }
    void transform_geometry(int id){Body& b=bodies[id];
        b.hulls.resize(b.shape->hulls.size());b.bounds={{inf,inf,inf},{-inf,-inf,-inf}};
        for(std::size_t k=0;k<b.hulls.size();k++){transform_hull(b.shape->hulls[k],b.pose.p,b.pose.q,b.hulls[k]);b.bounds=joined(b.bounds,{b.hulls[k].low,b.hulls[k].high});}
    }
    struct GeometryTask {Impl* world;int id;std::exception_ptr error;};
    std::vector<GeometryTask> geometry_arguments;std::vector<threadpool_task_t> geometry_tasks;
    static void geometry_task(void* argument){auto& task=*static_cast<GeometryTask*>(argument);
        try{task.world->transform_geometry(task.id);}catch(...){task.error=std::current_exception();}
    }
    static void warm_task(void* argument){auto& task=*static_cast<GeometryTask*>(argument);Impl& world=*task.world;Body& body=world.bodies[task.id];
        Vec3 momentum{},torque{};
        for(int index:body.contacts){const Row& row=world.rows[index];Vec3 j=row.n*row.normal+row.tangent;
            Vec3 r=row.ra;if(row.a!=task.id){j=j*(-1);r=row.rb;}momentum=momentum+j;torque=torque+cross(r,j);}
        body.v=body.v+momentum*body.inverse_mass;body.w=body.w+mul(body.inverse,torque);body.L=body.L+torque;
    }
    void geometry(int id,double dt,bool refresh=true){Body& b=bodies[id];
        if(refresh&&(!b.fixed||b.hulls.empty()))transform_geometry(id);
        Bounds swept=b.bounds;
        if(!b.fixed){Vec3 d=b.v*dt;swept=joined(swept,{b.bounds.low+d,b.bounds.high+d});
            double r=b.shape->radius*std::min(2.,length(b.w)*dt);Vec3 padding{r,r,r};swept.low=swept.low-padding;swept.high=swept.high+padding;}
        Vec3 skin{p.skin,p.skin,p.skin};swept.low=swept.low-skin;swept.high=swept.high+skin;
        if(b.leaf<0||!contains(b.fat,swept)){
            if(b.leaf>=0)space.erase(b.leaf);Vec3 pad{.002,.002,.002};b.fat={swept.low-pad,swept.high+pad};b.leaf=space.insert(id,b.fat);
        }
        b.bounds=swept;
    }
    static void collide_task(void* argument){
        PairWork& task=*static_cast<PairWork*>(argument);
        try{task.world->pair_geometry(task);task.world->build_rows(task);}catch(...){task.error=std::current_exception();}
    }
    void pair_geometry(PairWork& task){
        const Body& a=bodies[task.a];const Body* b=task.b<0?nullptr:&bodies[task.b];
        auto& points=task.points;points.clear();task.rows.clear();task.narrow=0;task.error=nullptr;
        thread_local CollideScratch scratch;
        thread_local std::vector<ManifoldPoint> part_points;
        double sweep=length(a.v-(b?b->v:Vec3{}))*task.dt+a.shape->radius*length(a.w)*task.dt+(b?b->shape->radius*length(b->w)*task.dt:0);
        double margin=p.skin+sweep;
        for(const WorldHull& ha:a.hulls){
            if(!b){if(ha.low.z-p.ground_z>margin)continue;part_points.clear();collide_ground(ha,p.ground_z,margin,part_points);
                for(const auto& m:part_points)points.push_back(m);}
            else for(const WorldHull& hb:b->hulls){if(!bounds_overlap(ha,hb,margin)||centre_separates(ha,hb,margin))continue;
                part_points.clear();double gap;task.narrow++;collide_hulls(ha,hb,margin,scratch,part_points,gap);
                for(const auto& m:part_points)points.push_back(m);}
        }
    }
    void build_rows(PairWork& task){int ia=task.a,ib=task.b;double dt=task.dt;
        const Body& a=bodies[ia];const Body* b=ib<0?nullptr:&bodies[ib];
        const auto& points=task.points;auto& rows=task.rows;
        if(points.empty())return;
        Relation& relation=*task.relation;double scale=relation.dt>0?dt/relation.dt:0;
        auto& next=task.next_points;next.clear();next.reserve(points.size());
        for(const ManifoldPoint& m:points){Memory x;x.a=rotate(conjugate(a.pose.q),m.point_a-a.pose.p);
            x.b=b?rotate(conjugate(b->pose.q),m.point_b-b->pose.p):m.point_b;x.n=m.normal;
            bool duplicate=false;for(const Memory& prior:next)if(dot(x.a-prior.a,x.a-prior.a)<1e-14&&dot(x.n,prior.n)>.9999){duplicate=true;break;}if(duplicate)continue;
            if(m.separation<=p.skin*4){double best=1e-6;for(const Memory& old:relation.points){double distance=dot(x.a-old.a,x.a-old.a);
                if(distance<best&&dot(x.n,old.n)>.98){best=distance;x.normal=old.normal*scale;x.tangent=(old.tangent-x.n*dot(old.tangent,x.n))*scale;}}}
            next.push_back(x);
        }
        // Recycle the previous witness allocation after matching. The rows
        // below point into the new relation buffer until the next interval.
        relation.points.swap(next);relation.dt=dt;relation.seen=stats.steps;
        for(Memory& memory:relation.points){Row r;r.a=ia;r.b=ib;r.memory=&memory;r.n=memory.n;
            Vec3 pa=a.pose.p+rotate(a.pose.q,memory.a),pb=b?b->pose.p+rotate(b->pose.q,memory.b):memory.b,shared=(pa+pb)*.5;
            r.ra=shared-a.pose.p;if(b)r.rb=shared-b->pose.p;double gap=dot(pa-pb,r.n);
            r.t1=normalized(cross(r.n,std::abs(r.n.x)<.7?Vec3{1,0,0}:Vec3{0,1,0}));r.t2=cross(r.n,r.t1);
            r.response_n=response(ia,r.ra,r.n)+response(ib,r.rb,r.n);r.kn=dot(r.n,r.response_n);r.inverse_kn=r.kn>0?1/r.kn:0;
            Vec3 t1=response(ia,r.ra,r.t1)+response(ib,r.rb,r.t1),t2=response(ia,r.ra,r.t2)+response(ib,r.rb,r.t2);
            r.k11=dot(r.t1,t1);r.k12=dot(r.t1,t2);r.k22=dot(r.t2,t2);
            double det=r.k11*r.k22-r.k12*r.k12;if(det>1e-30){r.inverse11=r.k22/det;r.inverse12=-r.k12/det;r.inverse22=r.k11/det;}
            r.mu=std::sqrt(a.shape->friction*(b?b->shape->friction:p.ground_friction));r.normal=memory.normal;r.tangent=memory.tangent;
            double limit=r.mu*r.normal,size=length(r.tangent);if(size>limit)r.tangent=r.tangent*(limit/size);
            double vn=dot(velocity(ia,r.ra)-velocity(ib,r.rb),r.n);
            r.target=-std::max(0.,gap-p.skin)/dt;
            double restitution=std::max(a.shape->restitution,b?b->shape->restitution:0.);
            if(gap<=p.skin*2&&vn<-.1)r.target=std::max(r.target,-restitution*vn);
            if(r.kn<=0)continue;rows.push_back(r);
        }
    }
    double exchange(Row& r,int& recipient,int reader){
        Vec3 u=velocity(r.a,r.ra)-velocity(r.b,r.rb);double vn=dot(u,r.n);
        if(r.normal==0&&vn>=r.target-p.velocity_tolerance)return 0;
        if(std::abs(vn-r.target)<=p.velocity_tolerance){Vec3 slip=u-r.n*vn;
            if(dot(slip,slip)<=p.velocity_tolerance*p.velocity_tolerance)return 0;
        }
        double normal=std::max(0.,r.normal+(r.target-vn)*r.inverse_kn);Vec3 dn=r.n*(normal-r.normal);
        Vec3 after=u+r.response_n*(normal-r.normal);
        double x=dot(after,r.t1),y=dot(after,r.t2);
        Vec3 tangent=r.tangent;
        tangent=tangent-r.t1*(r.inverse11*x+r.inverse12*y)-r.t2*(r.inverse12*x+r.inverse22*y);
        double bound=r.mu*normal,size=length(tangent);
        if(size>bound){
            if(bound<=0)tangent={};
            else {
                // Project in the contact's effective-mass metric. Radial
                // clamping is only correct for an isotropic tangent matrix.
                double old1=dot(r.tangent,r.t1),old2=dot(r.tangent,r.t2);
                double f1=r.k11*old1+r.k12*old2-x,f2=r.k12*old1+r.k22*old2-y;
                double lower=0,upper=std::hypot(f1,f2)/bound,lambda=upper*.5;
                double tx=0,ty=0;
                for(int iteration=0;iteration<16;iteration++){
                    double a=r.k11+lambda,c=r.k22+lambda,d=a*c-r.k12*r.k12;
                    tx=(c*f1-r.k12*f2)/d;ty=(a*f2-r.k12*f1)/d;
                    double norm=std::hypot(tx,ty),error=norm-bound;
                    if(std::abs(error)<=bound*1e-9)break;
                    if(error>0)lower=lambda;else upper=lambda;
                    double zx=(c*tx-r.k12*ty)/d,zy=(a*ty-r.k12*tx)/d;
                    double slope=-(tx*zx+ty*zy)/norm,next=lambda-error/slope;
                    lambda=(next>lower&&next<upper)?next:(lower+upper)*.5;
                }
                double norm=std::hypot(tx,ty),scale=norm>bound?bound/norm:1;
                tangent=(r.t1*tx+r.t2*ty)*scale;
            }
        }
        Vec3 j=dn+tangent-r.tangent;r.normal=normal;r.tangent=tangent;
        if(dot(j,j)<1e-36)return 0;
        double change_a=impulse(r.a,r.ra,j),change_b=impulse(r.b,r.rb,j*(-1));
        recipient=reader==r.a?r.b:r.a;
        return std::max(change_a,change_b);
    }
    void swap_cells(int a,int b){if(a==b)return;std::swap(tape[a],tape[b]);bodies[tape[a]].place=a;bodies[tape[b]].place=b;stats.swaps++;}
    static void contact_task(void* argument){
        auto& task=*static_cast<Task*>(argument);Impl& world=*task.world;Group& group=world.groups[task.group];Row& row=world.rows[group.first];
        // Lock both downstream records in ID order. A contact sees one coherent
        // incoming state and atomically publishes both sides of the exchange.
        // Disjoint contacts run concurrently; shared-body ripples serialize.
        int first=row.a,second=row.b;
        if(second>=0&&world.bodies[second].fixed)second=-1;
        if(second>=0&&second<first)std::swap(first,second);
        auto acquire=[&](int id){if(id>=0)while(world.bodies[id].lock->flag.test_and_set(std::memory_order_acquire)){while(world.bodies[id].lock->flag.test(std::memory_order_relaxed)){};}};
        acquire(first);acquire(second);
        auto revision=[&](int id){return id<0?std::uint64_t(0):world.bodies[id].velocity_revision;};
        if(world.p.skip_unchanged&&group.quiet&&group.seen_a==revision(row.a)&&group.seen_b==revision(row.b))group.skips++;
        else {
            bool quiet=true;
            for(int offset=0;offset<group.last-group.first;offset++){
                Row& contact=world.rows[task.reverse?group.last-1-offset:group.first+offset];
                int recipient=-1;contact.change=world.exchange(contact,recipient,contact.a);
                if(contact.change>0){group.exchanges++;quiet=false;}
            }
            group.quiet=quiet;group.seen_a=revision(row.a);group.seen_b=revision(row.b);
        }
        if(second>=0)world.bodies[second].lock->flag.clear(std::memory_order_release);
        world.bodies[first].lock->flag.clear(std::memory_order_release);
    }
    void move_tape(){
        if(!p.moving_tape)return;
        for(int cell=0;cell+1<int(tape.size());cell++){
            double strongest=p.velocity_tolerance*p.velocity_tolerance;int partner=-1;
            for(int index:bodies[tape[cell]].contacts){const Row& row=rows[index];int other=row.a==tape[cell]?row.b:row.a;
                if(other>=0&&!bodies[other].fixed&&bodies[other].place>cell+1&&row.change>strongest){strongest=row.change;partner=other;}}
            if(partner>=0)swap_cells(cell+1,bodies[partner].place);
        }
    }
    void read_tape(){
        if(p.streamed&&groups.size()>640){
            // All exchanges belong to the same physical interval. Only shared
            // body state imposes a dependency: the two-body transaction enforces
            // it. A global barrier after each numerical sweep is unnecessary.
            arguments.resize(groups.size()*p.reads);tasks.resize(arguments.size());std::size_t cursor=0;
            for(int pass=0;pass<p.reads;pass++)for(std::size_t offset=0;offset<tape.size();offset++){
                int id=tape[(pass&1)?tape.size()-1-offset:offset];
                for(std::size_t k=0;k<bodies[id].groups.size();k++){
                    int index=bodies[id].groups[(pass&1)?bodies[id].groups.size()-1-k:k];
                    arguments[cursor]={this,index,(pass&1)!=0};tasks[cursor]={contact_task,&arguments[cursor]};cursor++;
                }
            }
            pool.run(tasks);stats.passes+=p.reads;stats.reads+=tape.size()*p.reads;
            if(p.workers>1&&tasks.size()>64*p.workers)stats.parallel_batches++;
            for(const Group& group:groups)stats.exchanges+=group.exchanges;
            move_tape();return;
        }
        arguments.resize(groups.size());tasks.resize(groups.size());
        for(int pass=0;pass<p.reads;pass++){
            std::size_t cursor=0;
            for(std::size_t offset=0;offset<tape.size();offset++){
                int id=tape[(pass&1)?tape.size()-1-offset:offset];
                for(std::size_t k=0;k<bodies[id].groups.size();k++){
                    int index=bodies[id].groups[(pass&1)?bodies[id].groups.size()-1-k:k];
                    arguments[cursor]={this,index,(pass&1)!=0};tasks[cursor]={contact_task,&arguments[cursor]};cursor++;
                }
            }
            stats.passes++;stats.reads+=tape.size();
            if(tasks.size()>64*p.workers&&p.workers>1)stats.parallel_batches++;
            pool.run(tasks);
            double largest=0;
            for(const Row& row:rows){largest=std::max(largest,row.change);if(row.change>0)stats.exchanges++;}
            // Position is mutable state too. Actual exchange strength moves
            // the strongest downstream neighbor alongside each tape entry.
            move_tape();
            if(largest<=p.velocity_tolerance*p.velocity_tolerance)break;
        }
    }
    void position_read(){
        for(int pass=0;pass<p.position_reads;pass++){
            bool changed=false;
            for(int id:tape)for(int index:bodies[id].contacts){Row& row=rows[index];if(row.a!=id)continue;Body& a=bodies[row.a];Body* b=row.b<0?nullptr:&bodies[row.b];
                Vec3 pa=a.pose.p+rotate(a.pose.q,row.memory->a),pb=b?b->pose.p+rotate(b->pose.q,row.memory->b):row.memory->b;
                double gap=dot(pa-pb,row.n);if(gap>=-p.skin*.25)continue;
                Vec3 shared=(pa+pb)*.5,ra=shared-a.pose.p,rb=b?shared-b->pose.p:Vec3{};
                double k=dot(row.n,response(row.a,ra,row.n)+response(row.b,rb,row.n));Vec3 j=row.n*(.5*(-gap-p.skin*.25)/k);
                for(int side=0;side<2;side++){int body_id=side?row.b:row.a;if(body_id<0||bodies[body_id].fixed)continue;Body& body=bodies[body_id];
                    Vec3 delta=side?j*(-1):j,r=side?rb:ra;body.pose.p=body.pose.p+delta*body.inverse_mass;body.pose.q=turn(body.pose.q,mul(body.inverse,cross(r,delta)));tensor(body);}
                changed=true;stats.position_exchanges++;
            }
            if(!changed)break;
        }
    }
    void step(double dt){using Clock=std::chrono::steady_clock;auto start=Clock::now();stats.steps++;rows.clear();groups.clear();
        geometry_arguments.resize(tape.size());geometry_tasks.resize(tape.size());
        for(std::size_t i=0;i<tape.size();i++){int id=tape[i];Body& b=bodies[id];b.contacts.clear();b.groups.clear();b.v=b.v+p.gravity*dt;
            geometry_arguments[i]={this,id,{}};geometry_tasks[i]={geometry_task,&geometry_arguments[i]};}
        pool.run(geometry_tasks);
        for(std::size_t i=0;i<tape.size();i++){if(geometry_arguments[i].error)std::rethrow_exception(geometry_arguments[i].error);geometry(tape[i],dt,false);}
        std::size_t count=0;
        auto candidate=[&](int a,int b){if(count>=pairs.size())pairs.emplace_back();
            PairWork& task=pairs[count++];task.world=this;task.a=a;task.b=b;task.dt=dt;
            task.key=(std::uint64_t(std::uint32_t(a))<<32)|std::uint32_t(b+1);
        };
        for(int id:tape){Body& b=bodies[id];space.query(b.bounds,[&](int other){if(other==id||!bodies[other].alive)return;
                if(bodies[other].fixed||other>id)candidate(id,other);});
            if(b.bounds.low.z<=p.ground_z+p.skin)candidate(id,-1);}
        collision_tasks.resize(count);
        for(std::size_t i=0;i<count;i++){
            PairWork& task=pairs[i];auto previous=relations.find(task.key);
            if(previous!=relations.end())task.relation=&previous->second;
            else {task.fresh=Relation{};task.relation=&task.fresh;}
            collision_tasks[i]={collide_task,&task};
        }
        pool.run(collision_tasks);
        for(std::size_t i=0;i<count;i++){PairWork& task=pairs[i];if(task.error)std::rethrow_exception(task.error);
            stats.narrowphase+=task.narrow;
            if(task.rows.empty())continue;
            if(task.relation==&task.fresh)relations.emplace(task.key,std::move(task.fresh));
            int first=int(rows.size());
            for(const Row& row:task.rows){int index=int(rows.size());rows.push_back(row);bodies[row.a].contacts.push_back(index);
                if(row.b>=0&&!bodies[row.b].fixed)bodies[row.b].contacts.push_back(index);}
            bodies[task.a].groups.push_back(int(groups.size()));groups.push_back({first,int(rows.size())});
        }
        for(auto it=relations.begin();it!=relations.end();){if(it->second.seen!=stats.steps)it=relations.erase(it);else ++it;}
        stats.geometry_seconds+=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();
        for(std::size_t i=0;i<tape.size();i++)geometry_tasks[i]={warm_task,&geometry_arguments[i]};
        pool.run(geometry_tasks);
        read_tape();for(const Group& group:groups)stats.unchanged_tasks+=group.skips;
        stats.solve_seconds+=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();
        double residual=0;
        for(Row& r:rows){r.memory->normal=r.normal;r.memory->tangent=r.tangent;
            residual=std::max(residual,r.target-dot(velocity(r.a,r.ra)-velocity(r.b,r.rb),r.n));}
        stats.max_residual=std::max(stats.max_residual,residual);if(residual>p.velocity_tolerance*8)stats.unconverged++;
        for(int id:tape){Body& b=bodies[id];b.pose.p=b.pose.p+b.v*dt;
            // In unopposed flight, recover the exact constant-gravity integral.
            if(b.contacts.empty())b.pose.p=b.pose.p-p.gravity*(.5*dt*dt);
            // Only this record pays for its angular integration accuracy.
            // A fast spinner must not shorten every other object's interval.
            double remaining=dt;
            while(remaining>1e-14){double speed=length(b.w),h=std::min(remaining,p.angular_step/std::max(speed,1e-12));
                if(h<1e-12)throw std::runtime_error("angular integration interval underflow");
                Quat initial=b.pose.q,midpoint=turn(initial,b.w*(h*.5));
                Vec3 middle_w=mul(rotated(to_matrix(midpoint),b.inverse_local),b.L);
                b.pose.q=turn(initial,middle_w*h);tensor(b);remaining-=h;stats.rotation_reads++;
            }
        }

        position_read();now+=dt;stats.integration_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
    }
    int add(const Shape& shape,Pose pose,Vec3 v,Vec3 w,bool fixed){
        if(!(shape.mass>0)||shape.hulls.empty())throw std::invalid_argument("cooked finite-volume geometry required");
        Body b;b.shape=std::make_shared<Shape>(shape);b.pose=pose;b.v=v;b.w=w;b.fixed=fixed;
        for(int k=0;k<9;k++)b.local.m[k]=shape.inertia[k];
        if(!fixed){b.inverse_mass=1/shape.mass;b.inverse_local=inverse(b.local);}
        b.L=mul(rotated(to_matrix(pose.q),b.local),w);tensor(b);int id=int(bodies.size());bodies.push_back(std::move(b));
        if(fixed)geometry(id,0);else {bodies[id].place=int(tape.size());tape.push_back(id);}return id;
    }
};
World::World(Parameters p){if(p.workers==0)p.workers=std::max(1u,std::thread::hardware_concurrency());if(!(p.step>0&&p.angular_step>0&&p.skin>0&&p.reads>0&&p.workers>0))throw std::invalid_argument("invalid tape parameters");impl=std::make_unique<Impl>(p);}
World::~World()=default;
BodyId World::add_body(const Shape& s,Pose p,Vec3 v,Vec3 w){return impl->add(s,p,v,w,false);}
BodyId World::add_static_body(const Shape& s,Pose p){return impl->add(s,p,{},{},true);}
void World::advance(double seconds){if(!std::isfinite(seconds)||seconds<0)throw std::invalid_argument("finite nonnegative duration required");
    double end=impl->now+seconds;
    while(impl->now<end-1e-13){double dt=std::min(impl->p.step,end-impl->now);
        impl->step(dt);
    }impl->now=end;
}
double World::time()const{return impl->now;}
BodyState World::state(BodyId id)const{const Body& b=impl->bodies.at(id);return {b.pose,b.v,b.w,false};}
Statistics World::statistics()const{return impl->stats;}
Parameters World::parameters()const{return impl->p;}
std::vector<BodyId> World::order()const{return impl->tape;}
std::vector<Contact> World::contacts()const{std::vector<Contact> out;for(const Row& r:impl->rows){const Body& a=impl->bodies[r.a];Vec3 pa=a.pose.p+rotate(a.pose.q,r.memory->a),pb=r.memory->b;
        if(r.b>=0){const Body& b=impl->bodies[r.b];pb=b.pose.p+rotate(b.pose.q,r.memory->b);}out.push_back({r.a,r.b,(pa+pb)*.5,r.n,dot(pa-pb,r.n),r.normal});}return out;}
double World::kinetic_energy()const{double result=0;for(int id:impl->tape){const Body& b=impl->bodies[id];result+=.5*b.shape->mass*dot(b.v,b.v)+.5*dot(b.w,b.L);}return result;}
void World::apply_impulse(BodyId id,Vec3 momentum,Vec3 point){Body& b=impl->bodies.at(id);if(b.alive&&!b.fixed)impl->impulse(id,point-b.pose.p,momentum);}
void World::set_pose(BodyId id,Pose pose){Body& b=impl->bodies.at(id);b.pose=pose;b.v={};b.w={};b.L={};impl->tensor(b);b.hulls.clear();impl->rows.clear();impl->relations.clear();if(b.fixed)impl->geometry(id,0);}
void World::remove_body(BodyId id){Body& b=impl->bodies.at(id);if(!b.alive)return;b.alive=false;if(b.leaf>=0)impl->space.erase(b.leaf);
    if(!b.fixed){impl->tape.erase(impl->tape.begin()+b.place);for(std::size_t i=0;i<impl->tape.size();i++)impl->bodies[impl->tape[i]].place=int(i);}impl->rows.clear();impl->relations.clear();}
}
