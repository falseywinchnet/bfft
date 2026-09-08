// A continuous, power-independent boundary operator for one homogeneous convex
// dielectric child. Only this owned face set is visible to its internal query.
// Parent rays are returned at egress; no internal ray enters the parent queue.
struct ChildBoundaryPort { Ray ray{}; double weight=0; int primitive=-1; };
struct ChildBoundaryResponse {
    std::vector<ChildBoundaryPort> ports;
    double unresolved_weight=0,source_weight=0; int internal_hits=0;bool cache_hit=false;
};
struct ConvexOpticalChild {
    std::vector<std::pair<int,Primitive>> faces;
    Material material,internal_material; bool participating=false,exact_states=false;
    using ResponseKey=std::array<std::uint64_t,12>;
    struct KeyHash {std::size_t operator()(const ResponseKey& key)const{
        std::uint64_t h=1469598103934665603ULL;for(auto value:key){h^=value;h*=1099511628211ULL;h^=h>>29;}return h;}};
    struct ResponseShard {std::mutex mutex;std::unordered_map<ResponseKey,ChildBoundaryResponse,KeyHash> entries;};
    std::unique_ptr<std::array<ResponseShard,32>> responses;
    mutable std::atomic<std::uint64_t> evaluations{0},evaluated_internal_hits{0},classification_evaluations{0};
    mutable std::atomic<std::uint64_t> unit_compilations{0},unit_internal_hits{0},unit_cache_bytes{0};
    explicit ConvexOpticalChild(const Scene& scene,int primitive) {
        participating=primitive==scene.jelly_volume;exact_states=scene.exact_child_states;
        if(exact_states)responses=std::make_unique<std::array<ResponseShard,32>>();
        material=scene.materials[scene.primitives[primitive].material];
        internal_material=participating&&scene.jelly_core>=0?
            scene.materials[scene.primitives[scene.jelly_core].material]:material;
        if(scene.prism_volume>=0&&primitive>=scene.prism_volume&&primitive<=scene.prism_top) {
            for(int id=scene.prism_volume;id<=scene.prism_top;++id)faces.emplace_back(id,scene.primitives[id]);
        } else faces.emplace_back(primitive,scene.primitives[primitive]);
    }
    Hit next(const Ray& ray,int ignore) const {
        Hit nearest; double limit=std::numeric_limits<double>::infinity();
        if(faces.size()>1){
            // The child is convex: its first outward supporting-plane crossing
            // is the exit. No finite-face epsilon can skip a grazing wedge.
            for(const auto& face:faces)if(face.first!=ignore){const Vec3 n=face.second.normal;
                const double den=dot(ray.direction,n);if(den<=1e-14)continue;
                const double t=dot(face.second.center-ray.origin,n)/den;
                if(t< -1e-9||t>=limit)continue;
                limit=std::max(0.,t);nearest={true,face.first,limit,ray.origin+ray.direction*limit,-n,n,false};}
            return nearest;
        }
        for(const auto& face:faces)if(face.first!=ignore){
            const Hit hit=intersect_primitive(face.second,face.first,ray,limit);
            if(hit.valid){nearest=hit;limit=hit.t;}}
        // Spheres have entry and exit on the same boundary primitive.
        if(faces.size()==1&&faces.front().second.shape==Shape::Sphere)
            nearest=intersect_primitive(faces.front().second,faces.front().first,ray);
        return nearest;
    }
    ChildBoundaryResponse evaluate_law(const Ray& incident,const Hit& entry,int channel,
        double coordinate,double cutoff,int ceiling,double solid_path=.24,double thin_path=.10,int port_limit=0) const {
        ChildBoundaryResponse result;
        const double index=spectral_ior(material,coordinate);
        const double boundary=material.base[channel]*std::exp(-material.absorption[channel]*(participating?0:(material.thin?thin_path:solid_path)));
        auto port=[&](const Hit& hit,Vec3 direction,double weight){
            if(weight>0)result.ports.push_back({{hit.position+direction*3e-4,direction},weight,hit.primitive});};
        Vec3 direction=incident.direction;double weight=1;
        Ray internal=incident;int previous=-1;
        if(entry.front||material.thin){
            const bool transmitted=material.thin||refract(incident.direction,entry.normal,1/index,direction);
            const double fresnel=transmitted?schlick(std::abs(dot(incident.direction,entry.normal)),1,index):1;
            port(entry,unit(reflect(incident.direction,entry.normal)),boundary*fresnel);
            weight=boundary*(1-fresnel);
            if(material.thin){port(entry,incident.direction,weight);return result;}
            internal={entry.position,direction};previous=entry.primitive;
        }
        while(weight>cutoff&&result.internal_hits<ceiling&&
            (port_limit==0||int(result.ports.size())<port_limit)){
            const Hit hit=next(internal,previous);
            if(!hit.valid){result.unresolved_weight+=weight;return result;}
            ++result.internal_hits;
            if(participating){const double retention=std::exp(-material.absorption[channel]*hit.t);
                result.source_weight+=weight*(1-retention);weight*=retention;}
            Vec3 external;
            const bool exit=refract(internal.direction,hit.normal,index,external);
            const double reflected=exit?schlick(std::abs(dot(internal.direction,hit.normal)),index,1):1;
            weight*=boundary;
            if(exit)port(hit,external,weight*(1-reflected));
            weight*=reflected;direction=unit(reflect(internal.direction,hit.normal));
            internal={hit.position,direction};previous=hit.primitive;
        }
        result.unresolved_weight+=weight;return result;
    }
    ChildBoundaryResponse classification_response(const Ray& ray,const Hit& hit,int channel,double coordinate,
        double cutoff,int ceiling)const {
        classification_evaluations.fetch_add(1,std::memory_order_relaxed);
        if(exact_states)return response(ray,hit,channel,coordinate,cutoff,ceiling);
        evaluations.fetch_add(1,std::memory_order_relaxed);
        // The existing classifier reads reflection and first transmission only.
        // Later internal exits cannot change these already determined ports.
        auto result=evaluate_law(ray,hit,channel,coordinate,material.thin?cutoff:1e-12,
            material.thin?ceiling:128,.24,.10,2);
        evaluated_internal_hits.fetch_add(result.internal_hits,std::memory_order_relaxed);return result;
    }
    ChildBoundaryResponse response(const Ray& incident,const Hit& entry,int channel,double coordinate,
        double cutoff,int ceiling,double solid_path=.24,double thin_path=.10)const {
        evaluations.fetch_add(1,std::memory_order_relaxed);
        // A child is registered once. Coordinates and wavelength are arguments
        // of its response law; querying the law cannot register another object.
        // Preserve the exact-state oracle's unit budget for matched comparison.
        if(!exact_states){auto result=evaluate_law(incident,entry,channel,coordinate,
                material.thin?cutoff:1e-12,material.thin?ceiling:128,solid_path,thin_path);
            evaluated_internal_hits.fetch_add(result.internal_hits,std::memory_order_relaxed);return result;}
        // Thin sheets already have a closed two-term response and no march.
        if(material.thin)return evaluate_law(incident,entry,channel,coordinate,cutoff,ceiling,solid_path,thin_path);
        // A unit response has its own numerical budget, independent of parent
        // population, parent recursion depth, camera identity, and source power.
        const Vec3 point=entry.front?entry.position:incident.origin;
        auto bits=[](double x){return std::bit_cast<std::uint64_t>(x==0?0.:x);};
        const ResponseKey key{{bits(point.x),bits(point.y),bits(point.z),bits(incident.direction.x),
            bits(incident.direction.y),bits(incident.direction.z),bits(coordinate),std::uint64_t(channel),
            std::uint64_t(entry.primitive),std::uint64_t(entry.front),bits(solid_path),bits(thin_path)}};
        auto& shard=(*responses)[KeyHash{}(key)%responses->size()];std::lock_guard<std::mutex> guard(shard.mutex);
        const auto found=shard.entries.find(key);
        if(found!=shard.entries.end()){auto result=found->second;result.cache_hit=true;return result;}
        auto result=evaluate_law(incident,entry,channel,coordinate,1e-12,128,solid_path,thin_path);
        unit_internal_hits.fetch_add(result.internal_hits,std::memory_order_relaxed);
        unit_compilations.fetch_add(1,std::memory_order_relaxed);
        unit_cache_bytes.fetch_add(sizeof(ResponseKey)+sizeof(ChildBoundaryResponse)+
            result.ports.size()*sizeof(ChildBoundaryPort),std::memory_order_relaxed);
        shard.entries.emplace(key,result);return result;
    }
};

bool same_child_definition(const ConvexOpticalChild& a,const ConvexOpticalChild& b){
    auto vec=[](Vec3 x,Vec3 y){return x.x==y.x&&x.y==y.y&&x.z==y.z;};
    auto material=[](const Material& x,const Material& y){return x.kind==y.kind&&x.base==y.base&&
        x.ior_rgb==y.ior_rgb&&x.absorption==y.absorption&&x.thin==y.thin&&x.diffuse==y.diffuse;};
    if(a.exact_states!=b.exact_states||a.participating!=b.participating||a.faces.size()!=b.faces.size()||
        !material(a.material,b.material)||!material(a.internal_material,b.internal_material))return false;
    for(std::size_t i=0;i<a.faces.size();++i){const auto& x=a.faces[i];const auto& y=b.faces[i];
        if(x.first!=y.first||x.second.shape!=y.second.shape||x.second.radius2!=y.second.radius2||
            !vec(x.second.center,y.second.center)||!vec(x.second.normal,y.second.normal)||
            !vec(x.second.origin,y.second.origin)||!vec(x.second.u,y.second.u)||!vec(x.second.v,y.second.v)||
            !vec(x.second.a,y.second.a)||!vec(x.second.b,y.second.b)||!vec(x.second.c,y.second.c))return false;}
    return true;
}
void refresh_optical_children(Scene& scene){
    auto previous=std::move(scene.optical_children);
    scene.optical_children.clear();scene.optical_owner.assign(scene.primitives.size(),-1);
    auto add=[&](int id){if(id<0||id>=int(scene.primitives.size()))return;
        const int owner=int(scene.optical_children.size());
        std::shared_ptr<const ConvexOpticalChild> child=std::make_shared<ConvexOpticalChild>(scene,id);
        for(const auto& old:previous)if(same_child_definition(*old,*child)){child=old;break;}
        for(const auto& face:child->faces)scene.optical_owner[face.first]=owner;
        scene.optical_children.push_back(std::move(child));};
    add(scene.prism_volume);add(scene.glass_sheet);add(scene.jelly_volume);
}
const ConvexOpticalChild* optical_child(const Scene& scene,int primitive){
    if(!scene.child_boundaries||primitive<0||primitive>=int(scene.optical_owner.size()))return nullptr;
    const int owner=scene.optical_owner[primitive];return owner<0?nullptr:scene.optical_children[owner].get();
}

// Boundary closure of the existing straight source-visibility model. This is
// deliberately separate from the refracting optical response: changing the
// source integration model is not an ownership operation.
struct ChildSegmentResponse { bool valid=false;Hit egress{};RGB transfer{1,1,1},source{}; };
ChildSegmentResponse child_segment_response(const ConvexOpticalChild& child,const Hit& entry,
    Vec3 receiver,Vec3 source){
    ChildSegmentResponse result;const Vec3 delta=source-receiver;
    const double length=norm(delta);if(length<1e-8)return result;
    const Vec3 direction=delta/length;const Material& material=child.material;
    result.egress=entry;result.valid=true;
    auto boundary=[&](const Hit& hit,double path){for(int c=0;c<3;++c)
        result.transfer[c]*=material.base[c]*(1-schlick(std::abs(dot(direction,hit.normal)),1,material.ior_rgb[c]))*
            std::exp(-material.absorption[c]*path);};
    if(child.participating){
        const double path=segment_sphere_length(child.faces.front().second,receiver,source);
        const Hit exit=child.next({entry.position,direction},entry.primitive);
        if(exit.valid&&dot(exit.position-receiver,direction)<length)result.egress=exit;
        for(int c=0;c<3;++c){const double b=material.base[c]*(1-schlick(std::abs(dot(direction,entry.normal)),1,material.ior_rgb[c]));
            const double retention=std::exp(-material.absorption[c]*path);
            result.transfer[c]=b*b*retention;result.source[c]=b*(1-retention);}
        return result;
    }
    boundary(entry,material.thin?.08:.16);
    if(!material.thin&&entry.front){
        const Hit exit=child.next({entry.position,direction},entry.primitive);
        if(!exit.valid){result.valid=false;return result;}
        if(dot(exit.position-receiver,direction)<length){boundary(exit,.16);result.egress=exit;}
    }
    return result;
}
