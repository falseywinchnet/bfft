// Experimental return extinction and direct scalar feedback closure.
// Included after OpticalPacket, before the sealed tracer.
struct ReturnEvent {
    OpticalPacket incoming{};
    Hit hit{};
    double source=0;
    std::array<OpticalPacket,2> children{};
    int count=0;
    bool pruned=false;
};

bool same_return_vector(Vec3 a,Vec3 b){return a.x==b.x&&a.y==b.y&&a.z==b.z;}
bool same_return_state(const ReturnEvent& event,const OpticalPacket& packet,const Hit& hit){
    return event.hit.primitive==hit.primitive&&event.hit.front==hit.front&&
        event.incoming.exclude_area_emitters==packet.exclude_area_emitters&&
        same_return_vector(event.hit.position,hit.position)&&same_return_vector(event.hit.normal,hit.normal)&&
        same_return_vector(event.hit.geometric_normal,hit.geometric_normal)&&
        same_return_vector(event.incoming.ray.direction,packet.ray.direction);
}

// Only terminal side exits are admitted. A second recurrent/branching exit
// would need a richer return relation and is left to the ordinary marcher.
bool return_terminal_source(const TraceContext& ctx,const OpticalPacket& packet,int channel,double& value){
    if(packet.weight<=ctx.optical_cutoff)return false;
    const Hit hit=packet.has_prefetched?packet.prefetched:optical_first_hit(ctx,packet.ray,
        std::numeric_limits<double>::infinity(),packet.ignore);
    value=0;if(!hit.valid)return true;
    const Material& material=ctx.scene.materials[ctx.scene.primitives[hit.primitive].material];
    if(material.kind==MaterialKind::Emissive){
        if(!(packet.exclude_area_emitters&&is_area_emitter(ctx.scene,hit.primitive)))
            value=packet.weight*emitted_radiance(ctx.scene,hit,-packet.ray.direction)[channel];
        return true;
    }
    if(material.kind!=MaterialKind::Diffuse)return false;
    value=packet.weight*base_radiance(ctx,hit,-packet.ray.direction,false,packet.weight)[channel];return true;
}

bool close_return_cycle(const TraceContext& ctx,const std::vector<ReturnEvent>& history,
    const OpticalPacket& packet,int ancestor,int channel,double& added){
    const double gain=packet.weight/history[ancestor].incoming.weight;
    if(!(gain>0&&gain<=1))return false;
    double cycle_source=0;int id=packet.return_parent,edge=packet.return_edge;
    while(id>=0){const ReturnEvent& event=history[id];
        if(event.pruned)return false;
        cycle_source+=event.source;
        for(int branch=0;branch<event.count;++branch)if(branch!=edge){double exit_source=0;
            if(!return_terminal_source(ctx,event.children[branch],channel,exit_source))return false;
            cycle_source+=exit_source;}
        if(id==ancestor){
            if(gain==1){if(cycle_source!=0)return false;added=0;return true;}
            added=cycle_source*(gain/(1-gain));return std::isfinite(added);
        }
        edge=event.incoming.return_edge;id=event.incoming.return_parent;
    }
    return false;
}
