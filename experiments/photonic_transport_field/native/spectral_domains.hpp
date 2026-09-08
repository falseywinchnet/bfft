// Experimental event-wise wavelength integration. The scalar packet tracer
// remains the pointwise oracle; no cutoff or material law is changed here.
struct SpectralDomainConfig{
    double absolute_error=1e-5,relative_error=1e-3;
    double topology_step=1.0/8192;
    int shading_depth=7,seed_intervals=8;
};
SpectralDomainConfig spectral_domain_config;
struct SpectralDomainStats{
    std::uint64_t steps=0,source_evaluations=0,domains=0,topology_limits=0,shading_limits=0;
    double estimated_error=0;
};
struct SpectralState{
    OpticalPacket packet{};double outer=1;bool valid=false,primary=false;
};
enum class SpectralTerm{None,Surface,Emission,Volume};
struct SpectralEvent{
    int owner=-1;Hit hit{};Vec3 view{};double coefficient=0,path_weight=1;
    SpectralTerm term=SpectralTerm::None;
    std::array<SpectralState,2> child{};
};

SpectralEvent spectral_event(const TraceContext& ctx,SpectralState state,int channel,double coordinate,
                              SpectralDomainStats& stats){
    ++stats.steps;SpectralEvent event;if(!state.valid||state.outer==0)return event;
    OpticalPacket packet=state.packet;
    if(!state.primary&&(packet.weight<=ctx.optical_cutoff||packet.interactions>=28))return event;
    const Hit hit=packet.has_prefetched?packet.prefetched:optical_first_hit(ctx,packet.ray,
        std::numeric_limits<double>::infinity(),packet.ignore);
    if(!hit.valid)return event;event.owner=hit.primitive;event.hit=hit;event.view=-packet.ray.direction;
    event.path_weight=packet.weight;
    const Material& material=ctx.scene.materials[ctx.scene.primitives[hit.primitive].material];
    if(state.primary){
        if(material.kind!=MaterialKind::Dielectric)throw std::logic_error("spectral root must be dielectric");
        const double index=spectral_ior(material,coordinate),ni=hit.front?1:index,nt=hit.front?index:1;
        const double fresnel=schlick(std::abs(dot(packet.ray.direction,hit.normal)),ni,nt);
        const double common=material.base[channel]*(hit.primitive==ctx.scene.jelly_volume?1.0:
            std::exp(-material.absorption[channel]*(material.thin?.10:.24)));
        auto root=[&](int branch,Ray ray,double outer){SpectralState child;child.valid=true;child.outer=outer;
            child.packet.ray=ray;child.packet.ignore=hit.primitive;child.packet.previous=hit.primitive;
            event.child[branch]=child;};
        const Vec3 reflected=unit(reflect(packet.ray.direction,hit.normal));
        root(0,{hit.position+reflected*3e-4,reflected},common*fresnel);
        Vec3 transmitted=packet.ray.direction;
        if(material.thin||refract(packet.ray.direction,hit.normal,ni/nt,transmitted)){
            if(material.thin)transmitted=packet.ray.direction;
            const VolumeChord chord=jelly_volume_chord(ctx.scene,hit,transmitted,index);
            if(chord.valid){const double retention=std::exp(-material.absorption[channel]*chord.length);
                event.term=SpectralTerm::Volume;event.coefficient=common*(1-fresnel)*(1-retention);
                root(1,chord.exit_ray,common*(1-fresnel)*retention);
            }else root(1,{hit.position+transmitted*3e-4,transmitted},common*(1-fresnel));
        }
        return event;
    }
    const bool direction_return=norm2(packet.previous_previous_direction)>.5&&
        dot(unit(packet.ray.direction),unit(packet.previous_previous_direction))>1-1e-9;
    if(packet.previous_previous==hit.primitive&&direction_return){const double ratio=packet.previous_previous_weight>0?
            packet.weight/packet.previous_previous_weight:0;
        packet.feedback=true;if(ratio>0&&ratio<1)packet.feedback_ratio=ratio;}
    auto continuation=[&](int branch,Vec3 direction,double weight,bool loop_remainder=false,
                          bool exclude_emitter=false,const Hit* prefetched=nullptr){
        if(weight<=0)return;OpticalPacket child=packet;child.ray={hit.position+direction*3e-4,direction};
        child.weight=weight;child.ignore=hit.primitive;child.interactions=packet.interactions+1;
        child.previous_previous=packet.previous;child.previous_previous_weight=packet.previous_weight;
        child.previous=hit.primitive;child.previous_weight=packet.weight;
        child.previous_previous_direction=packet.previous_direction;child.previous_direction=packet.ray.direction;
        child.exclude_area_emitters=packet.exclude_area_emitters||exclude_emitter;
        child.has_prefetched=prefetched!=nullptr;if(prefetched)child.prefetched=*prefetched;
        if(loop_remainder&&child.feedback&&child.feedback_ratio>0&&child.feedback_ratio<1&&
            child.weight/(1-child.feedback_ratio)<=ctx.optical_cutoff)return;
        event.child[branch]={child,state.outer,true,false};
    };
    if(material.kind==MaterialKind::Emissive){
        if(!(packet.exclude_area_emitters&&is_area_emitter(ctx.scene,hit.primitive))){
            event.term=SpectralTerm::Emission;event.coefficient=state.outer*packet.weight;}
        return event;}
    if(material.kind==MaterialKind::Dielectric){
        const double index=spectral_ior(material,coordinate),ni=hit.front?1:index,nt=hit.front?index:1;
        const double fresnel=schlick(std::abs(dot(packet.ray.direction,hit.normal)),ni,nt);
        const double common=packet.weight*material.base[channel]*(hit.primitive==ctx.scene.jelly_volume?1.0:
            std::exp(-material.absorption[channel]*(material.thin?.10:.24)));
        const Vec3 reflected=unit(reflect(packet.ray.direction,hit.normal));
        continuation(0,reflected,common*fresnel,packet.feedback);
        Vec3 transmitted=packet.ray.direction;
        if(material.thin||refract(packet.ray.direction,hit.normal,ni/nt,transmitted)){
            if(material.thin)transmitted=packet.ray.direction;
            const VolumeChord chord=jelly_volume_chord(ctx.scene,hit,transmitted,index);
            if(chord.valid){const double retention=std::exp(-material.absorption[channel]*chord.length);
                event.term=SpectralTerm::Volume;event.coefficient=state.outer*common*(1-fresnel)*(1-retention);
                continuation(1,chord.exit_ray.direction,common*(1-fresnel)*retention);
                if(event.child[1].valid)event.child[1].packet.ray=chord.exit_ray;
            }else continuation(1,transmitted,common*(1-fresnel));
        }
        return event;
    }
    if(material.kind==MaterialKind::Mirror){const Vec3 direction=unit(reflect(packet.ray.direction,hit.normal));
        continuation(0,direction,packet.weight*material.base[channel],packet.feedback);return event;}
    event.term=SpectralTerm::Surface;event.coefficient=state.outer*packet.weight;
    if(material.kind==MaterialKind::Metal||material.kind==MaterialKind::Glossy){
        if(material.kind==MaterialKind::Metal)event.coefficient*=.12;
        const Vec3 direction=unit(reflect(packet.ray.direction,hit.normal));const RoughTerminalRelation terminal=
            rough_terminal_relation(ctx.scene,{hit.position+direction*3e-4,direction},hit.primitive,material.roughness);
        if(!terminal.area_emitter)continuation(0,direction,packet.weight*terminal.coverage*
            (material.kind==MaterialKind::Metal?.88*material.base[channel]:(.04+.20*(1-material.roughness))),
            packet.feedback,true,&terminal.terminal);
    }
    return event;
}

double evaluate_spectral_term(const TraceContext& ctx,const SpectralEvent& event,int channel,SpectralDomainStats& stats){
    if(event.coefficient==0||event.term==SpectralTerm::None)return 0;
    if(event.term==SpectralTerm::Emission)return event.coefficient*emitted_radiance(ctx.scene,event.hit,event.view)[channel];
    if(event.term==SpectralTerm::Volume)return event.coefficient*jelly_source_radiance(ctx,channel);
    ++stats.source_evaluations;
    return event.coefficient*base_radiance(ctx,event.hit,event.view,false,event.path_weight)[channel];
}

SpectralState spectral_root(const Ray& ray,const Hit& hit){SpectralState root;root.valid=true;root.primary=true;
    root.packet.ray=ray;root.packet.prefetched=hit;root.packet.has_prefetched=true;return root;}

double spectral_event_point(const TraceContext& ctx,const Ray& ray,const Hit& hit,int channel,double coordinate,
                            SpectralDomainStats& stats){
    std::vector<SpectralState> stack{spectral_root(ray,hit)};double result=0;
    while(!stack.empty()){const auto state=stack.back();stack.pop_back();const auto event=spectral_event(ctx,state,channel,coordinate,stats);
        result+=evaluate_spectral_term(ctx,event,channel,stats);
        for(const auto& child:event.child)if(child.valid)stack.push_back(child);
    }
    return result;
}

class SpectralDomainIntegrator{
    const TraceContext& ctx;int channel;SpectralDomainConfig config;SpectralDomainStats& stats;
    std::uint64_t* signature;
    struct Sample{SpectralEvent event;double value=0;bool shaded=false;};
    using StateFunction=std::function<SpectralState(double)>;
    struct Node{
        SpectralDomainIntegrator& integration;StateFunction incoming;std::map<double,Sample> samples;
        Sample& at(double coordinate){auto [it,inserted]=samples.try_emplace(coordinate);
            if(inserted)it->second.event=spectral_event(integration.ctx,incoming(coordinate),integration.channel,coordinate,integration.stats);
            return it->second;}
        double value(double coordinate,int owner){Sample& sample=at(coordinate);
            if(sample.event.owner!=owner)return 0;
            if(!sample.shaded){sample.value=evaluate_spectral_term(integration.ctx,sample.event,integration.channel,integration.stats);
                sample.shaded=true;}
            return sample.value;}
    };
    struct Region{double begin,end;int owner;};
    void partition(Node& node,double begin,double end,std::vector<Region>& regions,int depth=0){
        const double middle=(begin+end)*.5;
        const int a=node.at(begin).event.owner,b=node.at(middle).event.owner,c=node.at(end).event.owner;
        if((a==b&&b==c)||end-begin<=config.topology_step||depth>=20){
            if(a!=b||b!=c)++stats.topology_limits;
            if(!regions.empty()&&regions.back().owner==b&&regions.back().end==begin)regions.back().end=end;
            else regions.push_back({begin,end,b});
            return;}
        partition(node,begin,middle,regions,depth+1);partition(node,middle,end,regions,depth+1);
    }
    double integrate_term(Node& node,const Region& region,int depth=0){
        const double width=region.end-region.begin,middle=.5*(region.begin+region.end),half=.5*width;
        double coarse=0,fine=0;
        for(int i=0;i<2;++i)coarse+=node.value(middle+half*emitter_gauss2_x[i],region.owner)*half;
        for(int i=0;i<3;++i)fine+=node.value(middle+half*emitter_gauss3_x[i],region.owner)*half*emitter_gauss3_w[i];
        const double error=std::abs(fine-coarse);
        if(error<=config.absolute_error*width+config.relative_error*std::abs(fine)||depth>=config.shading_depth){
            stats.estimated_error+=error;
            if(depth>=config.shading_depth&&error>config.absolute_error*width+config.relative_error*std::abs(fine))++stats.shading_limits;
            return fine;}
        return integrate_term(node,{region.begin,middle,region.owner},depth+1)+
            integrate_term(node,{middle,region.end,region.owner},depth+1);
    }
    double integrate_node(StateFunction incoming,double begin,double end,int depth){
        if(depth>29)throw std::logic_error("spectral event depth exceeded");
        Node node{*this,std::move(incoming),{}};std::vector<Region> regions;
        const int seeds=std::max(1,int(std::ceil((end-begin)*config.seed_intervals)));
        for(int seed=0;seed<seeds;++seed)partition(node,begin+(end-begin)*seed/seeds,begin+(end-begin)*(seed+1)/seeds,regions);
        double result=0;
        for(const Region& region:regions){if(region.owner<0)continue;++stats.domains;
            signature_push(signature,region.owner,channel+depth*3);
            // Shade this event before its descendants partition their supports.
            result+=integrate_term(node,region);
            for(int branch=0;branch<2;++branch){const int owner=region.owner;
                result+=integrate_node([&node,branch,owner](double coordinate){const SpectralEvent& event=node.at(coordinate).event;
                    return event.owner==owner?event.child[branch]:SpectralState{};},region.begin,region.end,depth+1);}
        }
        return result;
    }
public:
    SpectralDomainIntegrator(const TraceContext& context,int band,SpectralDomainConfig settings,SpectralDomainStats& metrics,
                             std::uint64_t* path_signature=nullptr):
        ctx(context),channel(band),config(settings),stats(metrics),signature(path_signature){}
    double integrate(const Ray& ray,const Hit& hit){const SpectralState root=spectral_root(ray,hit);
        return integrate_node([root](double){return root;},channel-.5,channel+.5,0);}
};

double integrate_spectral_domains(const TraceContext& ctx,const Ray& ray,const Hit& hit,int channel,std::uint64_t* signature,
                                  SpectralDomainStats* output=nullptr){
    SpectralDomainStats stats;SpectralDomainIntegrator integration(ctx,channel,spectral_domain_config,stats,signature);
    const double result=integration.integrate(ray,hit);if(output)*output=stats;return result;
}
