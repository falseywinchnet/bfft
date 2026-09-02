#define main regime_scene_application_main
#include "regime_scene_native.cpp"
#undef main

namespace {

struct PrimarySplit {
    RGB reflected{};
    RGB transmitted{};
    RGB other{};
    int owner=-1;
};

struct ComposedComponents {
    RGB area{};
    RGB beam{};
    RGB bounce{};
    int terminal=-1;
};

PrimarySplit split_primary(const TraceContext& context,const Ray& ray);
Hit composed_reflection_hit(const TraceContext& context,const Ray& primary_ray);
ComposedComponents composed_reflection_components(const TraceContext& context,const Ray& primary_ray);

struct PathContribution{std::string path;int endpoint=-1,channel=0;double value=0;};

void enumerate_dielectric_paths(const TraceContext& context,const Ray& ray,int depth,int channel,double weight,
                                const std::string& path,std::vector<PathContribution>& contributions,int ignore=-1){
    if(depth<=0||weight<=1e-12)return;const Hit hit=first_hit(context.scene,ray,
        std::numeric_limits<double>::infinity(),ignore);if(!hit.valid)return;const Material& material=
        context.scene.materials[context.scene.primitives[hit.primitive].material];
    if(material.kind==MaterialKind::Mirror){const Vec3 reflected=unit(reflect(ray.direction,hit.normal));
        enumerate_dielectric_paths(context,{hit.position+reflected*3e-4,reflected},depth-1,channel,
            weight*material.base[channel],path+">M("+context.scene.primitives[hit.primitive].name+")",
            contributions,hit.primitive);return;}
    if(material.kind!=MaterialKind::Dielectric){const double value=trace_channel(context,ray,depth,channel,nullptr,ignore);
        contributions.push_back({path+">"+context.scene.primitives[hit.primitive].name,hit.primitive,channel,weight*value});return;}
    const double ni=hit.front?1:material.ior_rgb[channel],nt=hit.front?material.ior_rgb[channel]:1;
    const double fresnel=schlick(std::abs(dot(ray.direction,hit.normal)),ni,nt);const double attenuation=
        material.base[channel]*std::exp(-material.absorption[channel]*(material.thin?.10:.24));
    const Vec3 reflected=unit(reflect(ray.direction,hit.normal));enumerate_dielectric_paths(context,
        {hit.position+reflected*3e-4,reflected},depth-1,channel,weight*attenuation*fresnel,
        path+">R("+context.scene.primitives[hit.primitive].name+")",contributions,hit.primitive);
    Vec3 transmitted=ray.direction;if(material.thin||refract(ray.direction,hit.normal,ni/nt,transmitted)){
        if(material.thin)transmitted=ray.direction;enumerate_dielectric_paths(context,
            {hit.position+transmitted*3e-4,transmitted},depth-1,channel,weight*attenuation*(1-fresnel),
            path+">T("+context.scene.primitives[hit.primitive].name+")",contributions,hit.primitive);}
}

void print_probe(const TraceContext& context,const Camera& camera,int sensor_width,int sensor_height,int x,int y){
    const Ray ray=camera_ray(camera,sensor_width,sensor_height,x,y);const Hit primary=first_hit(context.scene,ray);
    std::cout<<"probe "<<x<<","<<y<<" owner="<<(primary.valid?context.scene.primitives[primary.primitive].name:"miss")<<"\n";
    if(!primary.valid)return;const Material& material=context.scene.materials[context.scene.primitives[primary.primitive].material];
    const RGB base=base_radiance(context,primary,-ray.direction);std::cout<<"  material="<<material.name
        <<" position="<<primary.position.x<<","<<primary.position.y<<","<<primary.position.z
        <<" base="<<base[0]<<","<<base[1]<<","<<base[2]<<"\n";
    if(material.kind==MaterialKind::Glossy||material.kind==MaterialKind::Metal||material.kind==MaterialKind::Mirror){
        const Vec3 direction=unit(reflect(ray.direction,primary.normal));const Hit terminal=first_hit(context.scene,
            {primary.position+direction*3e-4,direction},std::numeric_limits<double>::infinity(),primary.primitive);
        std::cout<<"  delta_reflection_terminal="<<(terminal.valid?context.scene.primitives[terminal.primitive].name:"miss");
        if(terminal.valid){const Material& terminal_material=context.scene.materials[
            context.scene.primitives[terminal.primitive].material];std::cout<<" terminal_material="<<terminal_material.name;}
        std::cout<<"\n";}
    if(material.kind!=MaterialKind::Dielectric)return;const PrimarySplit split=split_primary(context,ray);
    std::cout<<"  reflected="<<split.reflected[0]<<","<<split.reflected[1]<<","<<split.reflected[2]
        <<" transmitted="<<split.transmitted[0]<<","<<split.transmitted[1]<<","<<split.transmitted[2]<<"\n";
    const Hit composed=composed_reflection_hit(context,ray);if(composed.valid){const RGB area=area_irradiance(context.scene,
        composed.position+composed.normal*2e-4,composed.normal,composed.primitive);const RGB beam=
        beam_irradiance(context.beams,composed.primitive,composed.position);RGB bounce{};const int node=
        context.field.index_by_primitive[composed.primitive];if(node>=0)bounce=context.field.bounce[node];
        const Material& composed_material=context.scene.materials[context.scene.primitives[composed.primitive].material];
        const Vec3 reflected_direction=unit(reflect(ray.direction,primary.normal));const RGB composed_base=
            base_radiance(context,composed,-reflected_direction);const RGB composed_specular=
            composed_material.kind==MaterialKind::Glossy||composed_material.kind==MaterialKind::Metal?
                specular_area(context,composed,-reflected_direction,composed_material):RGB{};
        std::cout<<"  composed_terminal="<<context.scene.primitives[composed.primitive].name<<" position="
            <<composed.position.x<<","<<composed.position.y<<","<<composed.position.z<<" area="
            <<area[0]<<","<<area[1]<<","<<area[2]<<" beam="<<beam[0]<<","<<beam[1]<<","<<beam[2]
            <<" bounce="<<bounce[0]<<","<<bounce[1]<<","<<bounce[2]<<" base="<<composed_base[0]<<","
            <<composed_base[1]<<","<<composed_base[2]<<" specular="<<composed_specular[0]<<","
            <<composed_specular[1]<<","<<composed_specular[2]<<"\n";}
    if(composed.valid){const Material& composed_material=context.scene.materials[
            context.scene.primitives[composed.primitive].material];
        if(composed_material.kind==MaterialKind::Metal||composed_material.kind==MaterialKind::Glossy){
            const Vec3 incoming=unit(reflect(ray.direction,primary.normal));const Vec3 continuation_direction=
                unit(reflect(incoming,composed.normal));const Ray continuation{
                    composed.position+continuation_direction*3e-4,continuation_direction};
            const RoughTerminalRelation relation=rough_terminal_relation(context.scene,continuation,
                composed.primitive,composed_material.roughness);const Hit terminal=first_hit(context.scene,continuation,
                    std::numeric_limits<double>::infinity(),composed.primitive);
            std::cout<<"  composed_delta_terminal="<<(terminal.valid?
                context.scene.primitives[terminal.primitive].name:"miss")<<" coverage="<<relation.coverage
                <<" area_emitter="<<relation.area_emitter;
            for(int channel=0;channel<3;++channel)std::cout<<(channel==0?" traced=":",")<<trace_channel(
                context,continuation,6,channel,nullptr,composed.primitive);
            for(int channel=0;channel<3;++channel)std::cout<<(channel==0?" area_sealed=":",")<<trace_channel(
                context,continuation,6,channel,nullptr,composed.primitive,
                std::numeric_limits<double>::quiet_NaN(),true);
            std::cout<<"\n";}}
    for(int channel=0;channel<3;++channel){const double ni=
        primary.front?1:material.ior_rgb[channel],nt=primary.front?material.ior_rgb[channel]:1;const double fresnel=
        schlick(std::abs(dot(ray.direction,primary.normal)),ni,nt);const double attenuation=material.base[channel]*
        std::exp(-material.absorption[channel]*(material.thin?.10:.24));Vec3 transmitted=ray.direction;
        std::vector<PathContribution> paths;const Vec3 reflected=unit(reflect(ray.direction,primary.normal));
        enumerate_dielectric_paths(context,{primary.position+reflected*3e-4,reflected},7,channel,
            attenuation*fresnel,"R("+context.scene.primitives[primary.primitive].name+")",paths,primary.primitive);
        if(material.thin||refract(ray.direction,primary.normal,ni/nt,transmitted)){
            if(material.thin)transmitted=ray.direction;enumerate_dielectric_paths(context,
                {primary.position+transmitted*3e-4,transmitted},7,channel,attenuation*(1-fresnel),
                "T("+context.scene.primitives[primary.primitive].name+")",paths,primary.primitive);}
        std::sort(paths.begin(),paths.end(),[](const PathContribution& left,const PathContribution& right){
            return left.value>right.value;});std::cout<<"  channel "<<channel<<"\n";
        for(std::size_t i=0;i<std::min<std::size_t>(paths.size(),8);++i)
            std::cout<<"    "<<std::setprecision(9)<<paths[i].value<<" "<<paths[i].path<<"\n";}
    for(int channel=0;channel<3;++channel){const double ni=primary.front?1:material.ior_rgb[channel];
        const double nt=primary.front?material.ior_rgb[channel]:1;Vec3 transmitted=ray.direction;
        if(!(material.thin||refract(ray.direction,primary.normal,ni/nt,transmitted)))continue;
        if(material.thin)transmitted=ray.direction;const Hit inner=first_hit(context.scene,
            {primary.position+transmitted*3e-4,transmitted},std::numeric_limits<double>::infinity(),primary.primitive);
        if(!inner.valid)continue;const Material& inner_material=context.scene.materials[
            context.scene.primitives[inner.primitive].material];
        if(inner_material.kind!=MaterialKind::Glossy&&inner_material.kind!=MaterialKind::Metal)continue;
        const Vec3 reflected=unit(reflect(transmitted,inner.normal));
        const Hit terminal=first_hit(context.scene,{inner.position+reflected*3e-4,reflected},
            std::numeric_limits<double>::infinity(),inner.primitive);std::cout<<"  transmitted_delta_terminal channel "<<channel
            <<" via "<<context.scene.primitives[inner.primitive].name<<" -> "
            <<(terminal.valid?context.scene.primitives[terminal.primitive].name:"miss")<<"\n";}
}

PrimarySplit split_primary(const TraceContext& context,const Ray& ray){
    PrimarySplit split;const Hit hit=first_hit(context.scene,ray);if(!hit.valid)return split;
    split.owner=hit.primitive;const Material& material=
        context.scene.materials[context.scene.primitives[hit.primitive].material];
    if(material.kind!=MaterialKind::Dielectric){split.other=trace_primary(context,ray,hit,nullptr);return split;}
    for(int channel=0;channel<3;++channel){const double ni=hit.front?1:material.ior_rgb[channel];
        const double nt=hit.front?material.ior_rgb[channel]:1;const double fresnel=
            schlick(std::abs(dot(ray.direction,hit.normal)),ni,nt);const double scale=material.base[channel]*
            std::exp(-material.absorption[channel]*(material.thin?.10:.24));
        const Vec3 reflected_direction=unit(reflect(ray.direction,hit.normal));split.reflected[channel]=scale*fresnel*
            trace_channel(context,{hit.position+reflected_direction*3e-4,reflected_direction},7,channel,nullptr,hit.primitive);
        Vec3 transmitted_direction=ray.direction;if(material.thin||refract(ray.direction,hit.normal,ni/nt,transmitted_direction)){
            if(material.thin)transmitted_direction=ray.direction;split.transmitted[channel]=scale*(1-fresnel)*
                trace_channel(context,{hit.position+transmitted_direction*3e-4,transmitted_direction},7,channel,nullptr,hit.primitive);}}
    return split;
}

RGB owner_colour(int owner){
    if(owner<0)return {};std::uint64_t x=static_cast<std::uint64_t>(owner+1)*0x9e3779b97f4a7c15ULL;
    x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;
    return {0.18+0.82*((x>>0)&255)/255.0,0.18+0.82*((x>>8)&255)/255.0,
        0.18+0.82*((x>>16)&255)/255.0};
}

int composed_reflection_terminal(const TraceContext& context,const Ray& primary_ray){
    Hit hit=first_hit(context.scene,primary_ray);if(!hit.valid)return -1;const Material& primary_material=
        context.scene.materials[context.scene.primitives[hit.primitive].material];
    if(primary_material.kind!=MaterialKind::Dielectric)return -1;Vec3 direction=unit(reflect(primary_ray.direction,hit.normal));
    Ray ray{hit.position+direction*3e-4,direction};int previous=hit.primitive;
    for(int depth=0;depth<6;++depth){hit=first_hit(context.scene,ray,std::numeric_limits<double>::infinity(),previous);
        if(!hit.valid)return -1;const Material& material=context.scene.materials[context.scene.primitives[hit.primitive].material];
        if(material.kind!=MaterialKind::Mirror)return hit.primitive;direction=unit(reflect(ray.direction,hit.normal));
        ray={hit.position+direction*3e-4,direction};previous=hit.primitive;}
    return -1;
}

Hit composed_reflection_hit(const TraceContext& context,const Ray& primary_ray){
    Hit hit=first_hit(context.scene,primary_ray);if(!hit.valid)return {};const Material& primary_material=
        context.scene.materials[context.scene.primitives[hit.primitive].material];if(primary_material.kind!=MaterialKind::Dielectric)return {};
    Vec3 direction=unit(reflect(primary_ray.direction,hit.normal));Ray ray{hit.position+direction*3e-4,direction};int previous=hit.primitive;
    for(int depth=0;depth<6;++depth){hit=first_hit(context.scene,ray,std::numeric_limits<double>::infinity(),previous);
        if(!hit.valid)return {};const Material& material=context.scene.materials[context.scene.primitives[hit.primitive].material];
        if(material.kind!=MaterialKind::Mirror)return hit;direction=unit(reflect(ray.direction,hit.normal));
        ray={hit.position+direction*3e-4,direction};previous=hit.primitive;}
    return {};
}

ComposedComponents composed_reflection_components(const TraceContext& context,const Ray& primary_ray){
    ComposedComponents components;Hit hit=first_hit(context.scene,primary_ray);if(!hit.valid)return components;
    const Material& primary_material=context.scene.materials[context.scene.primitives[hit.primitive].material];
    if(primary_material.kind!=MaterialKind::Dielectric)return components;
    RGB throughput{};for(int channel=0;channel<3;++channel){const double ni=hit.front?1:primary_material.ior_rgb[channel];
        const double nt=hit.front?primary_material.ior_rgb[channel]:1;const double fresnel=
            schlick(std::abs(dot(primary_ray.direction,hit.normal)),ni,nt);throughput[channel]=primary_material.base[channel]*
            std::exp(-primary_material.absorption[channel]*(primary_material.thin?.10:.24))*fresnel;}
    Vec3 direction=unit(reflect(primary_ray.direction,hit.normal));Ray ray{hit.position+direction*3e-4,direction};
    int previous=hit.primitive;
    for(int depth=0;depth<6;++depth){hit=first_hit(context.scene,ray,std::numeric_limits<double>::infinity(),previous);
        if(!hit.valid)return {};const Material& material=context.scene.materials[context.scene.primitives[hit.primitive].material];
        if(material.kind==MaterialKind::Mirror){throughput=multiply(throughput,material.base);
            direction=unit(reflect(ray.direction,hit.normal));ray={hit.position+direction*3e-4,direction};previous=hit.primitive;continue;}
        if(material.kind==MaterialKind::Dielectric||material.kind==MaterialKind::Emissive)return {};
        components.terminal=hit.primitive;const RGB diffuse_scale=multiply(throughput,material.base)*(material.diffuse/pi);
        const Vec3 sample=hit.position+hit.normal*2e-4;components.area=multiply(diffuse_scale,
            area_irradiance(context.scene,sample,hit.normal,hit.primitive));components.beam=multiply(diffuse_scale,
            beam_irradiance(context.beams,hit.primitive,hit.position));const int node=context.field.index_by_primitive[hit.primitive];
        if(node>=0)components.bounce=multiply(diffuse_scale,context.field.bounce[node]);return components;}
    return {};
}

RGB base_beam_radiance(const TraceContext& context,const Hit& hit){const Material& material=
    context.scene.materials[context.scene.primitives[hit.primitive].material];return multiply(
        beam_irradiance(context.beams,hit.primitive,hit.position),material.base)*(material.diffuse/pi);
}

double trace_beam_channel(const TraceContext& context,const Ray& ray,int depth,int channel,int ignore=-1){
    if(depth<=0)return 0;const Hit hit=first_hit(context.scene,ray,std::numeric_limits<double>::infinity(),ignore);
    if(!hit.valid)return 0;const Material& material=context.scene.materials[context.scene.primitives[hit.primitive].material];
    if(material.kind==MaterialKind::Emissive)return 0;if(material.kind==MaterialKind::Dielectric){const double ni=
        hit.front?1:material.ior_rgb[channel],nt=hit.front?material.ior_rgb[channel]:1;const double fresnel=
        schlick(std::abs(dot(ray.direction,hit.normal)),ni,nt);const Vec3 reflected=unit(reflect(ray.direction,hit.normal));
        const double reflected_value=trace_beam_channel(context,{hit.position+reflected*3e-4,reflected},depth-1,channel,hit.primitive);
        Vec3 transmitted=ray.direction;double transmitted_value=0;if(material.thin||refract(ray.direction,hit.normal,ni/nt,transmitted)){
            if(material.thin)transmitted=ray.direction;transmitted_value=trace_beam_channel(context,
                {hit.position+transmitted*3e-4,transmitted},depth-1,channel,hit.primitive);}
        return material.base[channel]*std::exp(-material.absorption[channel]*(material.thin?.10:.24))*
            ((1-fresnel)*transmitted_value+fresnel*reflected_value);}
    if(material.kind==MaterialKind::Mirror){const Vec3 direction=unit(reflect(ray.direction,hit.normal));return material.base[channel]*
        trace_beam_channel(context,{hit.position+direction*3e-4,direction},depth-1,channel,hit.primitive);}
    const RGB base=base_beam_radiance(context,hit);if(material.kind==MaterialKind::Metal){const Vec3 direction=
        unit(reflect(ray.direction,hit.normal));return .12*base[channel]+.88*material.base[channel]*trace_beam_channel(
            context,{hit.position+direction*3e-4,direction},depth-1,channel,hit.primitive);}
    if(material.kind==MaterialKind::Glossy){const Vec3 direction=unit(reflect(ray.direction,hit.normal));return base[channel]+
        (.04+.20*(1-material.roughness))*trace_beam_channel(context,
            {hit.position+direction*3e-4,direction},depth-1,channel,hit.primitive);}return base[channel];
}

RGB trace_primary_beam(const TraceContext& context,const Ray& ray,const Hit& hit){if(!hit.valid)return {};
    const Material& material=context.scene.materials[context.scene.primitives[hit.primitive].material];
    if(material.kind==MaterialKind::Diffuse)return base_beam_radiance(context,hit);if(material.kind==MaterialKind::Emissive)return {};
    RGB result{};for(int channel=0;channel<3;++channel){if(material.kind==MaterialKind::Dielectric){const double ni=
            hit.front?1:material.ior_rgb[channel],nt=hit.front?material.ior_rgb[channel]:1;const double fresnel=
            schlick(std::abs(dot(ray.direction,hit.normal)),ni,nt);const Vec3 reflected=unit(reflect(ray.direction,hit.normal));
            const double reflected_value=trace_beam_channel(context,{hit.position+reflected*3e-4,reflected},7,channel,hit.primitive);
            Vec3 transmitted=ray.direction;double transmitted_value=0;if(material.thin||refract(ray.direction,hit.normal,ni/nt,transmitted)){
                if(material.thin)transmitted=ray.direction;transmitted_value=trace_beam_channel(context,
                    {hit.position+transmitted*3e-4,transmitted},7,channel,hit.primitive);}
            result[channel]=material.base[channel]*std::exp(-material.absorption[channel]*(material.thin?.10:.24))*
                ((1-fresnel)*transmitted_value+fresnel*reflected_value);
        }else if(material.kind==MaterialKind::Mirror){const Vec3 direction=unit(reflect(ray.direction,hit.normal));result[channel]=
            material.base[channel]*trace_beam_channel(context,{hit.position+direction*3e-4,direction},7,channel,hit.primitive);
        }else if(material.kind==MaterialKind::Metal){const RGB base=base_beam_radiance(context,hit);const Vec3 direction=
            unit(reflect(ray.direction,hit.normal));result[channel]=.12*base[channel]+.88*material.base[channel]*trace_beam_channel(
                context,{hit.position+direction*3e-4,direction},6,channel,hit.primitive);
        }else{const RGB base=base_beam_radiance(context,hit);const Vec3 direction=unit(reflect(ray.direction,hit.normal));
            result[channel]=base[channel]+(.04+.20*(1-material.roughness))*trace_beam_channel(
                context,{hit.position+direction*3e-4,direction},5,channel,hit.primitive);}}
    return result;
}

double trace_bottom_emitter_ghost_channel(const TraceContext& context,const Ray& primary_ray,int channel){
    struct GhostPacket{Ray ray{};double weight=0;int ignore=-1,interactions=0;bool bottom_reflected=false;};
    constexpr double cutoff=4e-8;constexpr int maximum_interactions=28;std::vector<GhostPacket> packets;
    auto split_dielectric=[&](const GhostPacket& packet,const Hit& hit){const Material& material=
        context.scene.materials[context.scene.primitives[hit.primitive].material];const double ni=
        hit.front?1:material.ior_rgb[channel],nt=hit.front?material.ior_rgb[channel]:1;const double fresnel=
        schlick(std::abs(dot(packet.ray.direction,hit.normal)),ni,nt);const double common=packet.weight*
        material.base[channel]*std::exp(-material.absorption[channel]*(material.thin?.10:.24));
        const Vec3 reflected=unit(reflect(packet.ray.direction,hit.normal));packets.push_back({
            {hit.position+reflected*3e-4,reflected},common*fresnel,hit.primitive,packet.interactions+1,
            packet.bottom_reflected||hit.primitive==context.scene.prism_bottom});Vec3 transmitted=packet.ray.direction;
        if(material.thin||refract(packet.ray.direction,hit.normal,ni/nt,transmitted)){
            if(material.thin)transmitted=packet.ray.direction;
            packets.push_back({{hit.position+transmitted*3e-4,transmitted},common*(1-fresnel),
                hit.primitive,packet.interactions+1,packet.bottom_reflected});}};
    const Hit primary=first_hit(context.scene,primary_ray);if(!primary.valid)return 0;const Material& primary_material=
        context.scene.materials[context.scene.primitives[primary.primitive].material];if(primary_material.kind!=MaterialKind::Dielectric)return 0;
    split_dielectric({primary_ray,1,-1,0,false},primary);double result=0;
    while(!packets.empty()){const GhostPacket packet=packets.back();packets.pop_back();
        if(packet.weight<=cutoff||packet.interactions>=maximum_interactions)continue;
        const Hit hit=first_hit(context.scene,packet.ray,std::numeric_limits<double>::infinity(),packet.ignore);
        if(!hit.valid)continue;const Material& material=
            context.scene.materials[context.scene.primitives[hit.primitive].material];
        if(material.kind==MaterialKind::Emissive){if(packet.bottom_reflected)result+=packet.weight*material.emission[channel];continue;}
        if(material.kind==MaterialKind::Dielectric){split_dielectric(packet,hit);continue;}
        if(material.kind==MaterialKind::Mirror){const Vec3 reflected=unit(reflect(packet.ray.direction,hit.normal));packets.push_back({
            {hit.position+reflected*3e-4,reflected},packet.weight*material.base[channel],hit.primitive,
            packet.interactions+1,packet.bottom_reflected});}
    }
    return result;
}

RGB trace_bottom_emitter_ghost(const TraceContext& context,const Ray& ray){RGB result{};
    for(int channel=0;channel<3;++channel)result[channel]=trace_bottom_emitter_ghost_channel(context,ray,channel);return result;}

std::vector<std::uint8_t> to_image(const std::vector<RGB>& values){
    std::vector<std::uint8_t> image(values.size()*3);for(std::size_t i=0;i<values.size();++i){
        for(int channel=0;channel<3;++channel)image[i*3+channel]=tone_byte(values[i][channel]);}
    return image;
}

} // namespace

int main(int argc,char** argv)try{
    int sensor_width=1920,sensor_height=1280,x0=500,y0=500,width=520,height=520;
    int probe_x=-1,probe_y=-1,fibres=19;double beam_cell=.095,deposit_radius=.14;
    bool supersample_beam=false,ghost_only=false;
    std::string prefix="/tmp/regime_branch",camera_mode="default";
    for(int i=1;i<argc;++i){const std::string argument=argv[i];if(argument=="--supersample-beam"){
            supersample_beam=true;continue;}if(argument=="--ghost-only"){ghost_only=true;continue;}
        if(i+1>=argc)throw std::runtime_error("missing value");
        if(argument=="--sensor-width")sensor_width=std::stoi(argv[++i]);
        else if(argument=="--sensor-height")sensor_height=std::stoi(argv[++i]);
        else if(argument=="--x0")x0=std::stoi(argv[++i]);else if(argument=="--y0")y0=std::stoi(argv[++i]);
        else if(argument=="--width")width=std::stoi(argv[++i]);else if(argument=="--height")height=std::stoi(argv[++i]);
        else if(argument=="--probe-x")probe_x=std::stoi(argv[++i]);else if(argument=="--probe-y")probe_y=std::stoi(argv[++i]);
        else if(argument=="--fibres")fibres=std::stoi(argv[++i]);else if(argument=="--beam-cell")beam_cell=std::stod(argv[++i]);
        else if(argument=="--deposit-radius")deposit_radius=std::stod(argv[++i]);
        else if(argument=="--prefix")prefix=argv[++i];else if(argument=="--camera")camera_mode=argv[++i];
        else throw std::runtime_error("unknown argument: "+argument);}
    Scene scene=build_regime_scene();for(BeamBundle& beam:scene.beams)beam.fibres=fibres;
    const BeamField beams=compile_beam_field(scene,{beam_cell,deposit_radius});
    const TransportField field=compile_transport_field(scene,beams);const TraceContext context{scene,beams,field,nullptr};
    const Camera camera=make_camera_mode(sensor_width,sensor_height,camera_mode);const std::size_t count=static_cast<std::size_t>(width)*height;
    std::vector<RGB> full(count),reflected(count),transmitted(count),owners(count),reflection_terminals(count),
        composed_area(count),composed_beam(count),composed_bounce(count),primary_beam(count),primary_beam_supersampled(count);
    std::vector<RGB> prism_edge_ground_truth(count),bottom_emitter_ghost(count);
    std::atomic<int> next_row{0};
    const unsigned workers=std::max(1u,std::thread::hardware_concurrency());std::vector<std::thread> threads;
    const auto start=Clock::now();for(unsigned worker=0;worker<workers;++worker)threads.emplace_back([&]{
        for(;;){const int y=next_row.fetch_add(1);if(y>=height)break;for(int x=0;x<width;++x){const Ray ray=
            camera_ray(camera,sensor_width,sensor_height,x0+x,y0+y);const std::size_t index=static_cast<std::size_t>(y)*width+x;
            bottom_emitter_ghost[index]=trace_bottom_emitter_ghost(context,ray);if(ghost_only)continue;
            const PrimarySplit split=split_primary(context,ray);reflected[index]=split.reflected;
            transmitted[index]=split.transmitted;full[index]=split.reflected+split.transmitted+split.other;
            owners[index]=owner_colour(split.owner);reflection_terminals[index]=
                owner_colour(composed_reflection_terminal(context,ray));const ComposedComponents components=
                composed_reflection_components(context,ray);composed_area[index]=components.area;
            composed_beam[index]=components.beam;composed_bounce[index]=components.bounce;
            const Hit hit=first_hit(context.scene,ray);primary_beam[index]=trace_primary_beam(context,ray,hit);
            prism_edge_ground_truth[index]=full[index];if(supersample_beam){constexpr int side=16;RGB sum{};
                bool has_prism_top=false,has_other=false;for(int sy=0;sy<side;++sy)for(int sx=0;sx<side;++sx){
                    const Ray subray=camera_ray(camera,sensor_width,sensor_height,x0+x-.5+(sx+.5)/side,
                        y0+y-.5+(sy+.5)/side);const Hit subhit=first_hit(context.scene,subray);
                    if(subhit.valid){sum+=trace_primary_beam(context,subray,subhit);has_prism_top=has_prism_top||
                        subhit.primitive==context.scene.prism_top;has_other=has_other||subhit.primitive!=context.scene.prism_top;}}
                primary_beam_supersampled[index]=sum*(1.0/(side*side));if(has_prism_top&&has_other){RGB full_sum{};
                    for(int sy=0;sy<side;++sy)for(int sx=0;sx<side;++sx){const Ray subray=camera_ray(camera,sensor_width,
                            sensor_height,x0+x-.5+(sx+.5)/side,y0+y-.5+(sy+.5)/side);const Hit subhit=
                            first_hit(context.scene,subray);if(subhit.valid)full_sum+=trace_primary(context,subray,subhit,nullptr);}
                    prism_edge_ground_truth[index]=full_sum*(1.0/(side*side));}}}}});
    for(auto& thread:threads)thread.join();
    write_ppm(prefix+"_bottom_emitter_ghost.ppm",to_image(bottom_emitter_ghost),width,height);
    if(!ghost_only){write_ppm(prefix+"_full.ppm",to_image(full),width,height);
        write_ppm(prefix+"_reflection.ppm",to_image(reflected),width,height);
        write_ppm(prefix+"_transmission.ppm",to_image(transmitted),width,height);
        write_ppm(prefix+"_owners.ppm",to_image(owners),width,height);
        write_ppm(prefix+"_reflection_terminals.ppm",to_image(reflection_terminals),width,height);
        write_ppm(prefix+"_composed_area.ppm",to_image(composed_area),width,height);
        write_ppm(prefix+"_composed_beam.ppm",to_image(composed_beam),width,height);
        write_ppm(prefix+"_composed_bounce.ppm",to_image(composed_bounce),width,height);
        write_ppm(prefix+"_primary_beam.ppm",to_image(primary_beam),width,height);
        if(supersample_beam)write_ppm(prefix+"_primary_beam_supersampled.ppm",to_image(primary_beam_supersampled),width,height);
        if(supersample_beam)write_ppm(prefix+"_prism_edge_ground_truth.ppm",to_image(prism_edge_ground_truth),width,height);}
    if(probe_x>=0&&probe_y>=0)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)
        print_probe(context,camera,sensor_width,sensor_height,probe_x+dx,probe_y+dy);
    std::cout<<std::fixed<<std::setprecision(3)<<"{\n  \"sensor\": ["<<sensor_width<<","<<sensor_height
        <<"],\n  \"camera\": \""<<camera_mode<<"\",\n  \"crop\": ["<<x0<<","<<y0<<","<<width<<","<<height<<"],\n  \"elapsed_ms\": "
        <<std::chrono::duration<double,std::milli>(Clock::now()-start).count()<<",\n  \"fibres\": "<<fibres
        <<",\n  \"beam_cell\": "<<beam_cell<<",\n  \"deposits\": "<<beams.deposits.size()
        <<",\n  \"prefix\": \""<<prefix<<"\"\n}\n";
    return 0;
}catch(const std::exception& error){std::cerr<<"error: "<<error.what()<<"\n";return 2;}
