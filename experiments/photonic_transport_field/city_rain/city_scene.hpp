// Static city geometry and materials for the retained observer-volume test.
struct CityScene {
    Scene scene;
    std::vector<std::uint8_t> facade;
    std::vector<float> facade_phase;
    int buildings=0;
};
CityScene make_rain_city(){
    CityScene city;Scene& s=city.scene;
    const int ground=add_material(s,{"slate avenue",MaterialKind::Diffuse,{.19,.24,.29},{},.9,.8});
    const int roof=add_material(s,{"roof stone",MaterialKind::Diffuse,{.40,.44,.48},{},.88,.8});
    std::array<int,6> walls;
    const std::array<RGB,6> colors{{{.49,.36,.27},{.26,.39,.46},{.54,.48,.38},{.24,.31,.38},{.39,.43,.46},{.58,.39,.26}}};
    for(int i=0;i<6;++i)walls[i]=add_material(s,{"city facade "+std::to_string(i),MaterialKind::Diffuse,colors[i],{},.91,.6});
    const int sun=add_material(s,{"late afternoon area sky",MaterialKind::Emissive,{1,1,1},{20,17,14},0,0});
    s.floor=add_rect(s,"city plane",{-32,0,-42},{64,0,0},{0,0,68},{0,1,0},ground);
    const int light=add_rect(s,"large sky source",{-18,22,-2},{26,0,0},{0,0,20},{0,-1,0},sun,false);
    s.area_lights.push_back({light,s.materials[sun].emission});
    auto building=[&](double x,double z,double w,double d,double h,int material,int phase){
        const int begin=int(s.primitives.size());
        add_rect(s,"building front",{x-w/2,0,z+d/2},{w,0,0},{0,h,0},{0,0,1},material);
        add_rect(s,"building east",{x+w/2,0,z+d/2},{0,0,-d},{0,h,0},{1,0,0},material);
        add_rect(s,"building back",{x+w/2,0,z-d/2},{-w,0,0},{0,h,0},{0,0,-1},material);
        add_rect(s,"building west",{x-w/2,0,z-d/2},{0,0,d},{0,h,0},{-1,0,0},material);
        add_rect(s,"building roof",{x-w/2,h,z+d/2},{w,0,0},{0,0,-d},{0,1,0},roof);
        city.facade.resize(s.primitives.size());city.facade_phase.resize(s.primitives.size());
        for(int i=0;i<4;++i){city.facade[begin+i]=1;city.facade_phase[begin+i]=float(phase+i*3);}
        ++city.buildings;
    };
    for(int row=0;row<5;++row)for(int col=0;col<4;++col){
        const double x=(col<2?col-2:col-1)*4.2+(col<2?1.0:-1.0);
        const double z=1.0-row*5.0;
        const double h=2.1+double((row*7+col*11)%11)*.42+(row>=3?.8:0);
        building(x,z,2.75+.2*((row+col)%2),3.2,h,walls[(row*3+col)%6],row*13+col*7);
    }
    // A distant landmark provides narrow silhouettes and strong occlusion tests.
    building(-1.3,-26,3.3,3.3,10.5,walls[3],73);
    building(-1.3,-26,2.25,2.25,12.4,walls[3],75);
    const int warm=add_material(s,{"street bollard light",MaterialKind::Emissive,{1,1,1},{3.5,2.1,.8},0,0});
    for(int i=0;i<8;++i)for(int side:{-1,1})add_sphere(s,"street light",{side*.9,.24,2-i*3.1},.085,warm,false);
    city.facade.resize(s.primitives.size());city.facade_phase.resize(s.primitives.size());
    build_scene_bvh(s);return city;
}
RGB rain_sky(Vec3 direction){
    const double h=std::clamp(direction.y*.9+.15,0.,1.);
    RGB horizon{1.25,.89,.65},zenith{.18,.37,.70};RGB value{};
    for(int c=0;c<3;++c)value[c]=horizon[c]*(1-h)+zenith[c]*h;
    const Vec3 sun=unit(Vec3{-.7,.32,.4});const double glow=std::pow(std::max(dot(direction,sun),0.),32);
    value+=RGB{1.2,.65,.20}*glow;return value;
}
// All appearance is evaluated while the static position/normal field is built.
RGB evaluate_city(const CityScene& city,const TraceContext& ctx,const Ray& ray){
    if(rain_city_frozen.load(std::memory_order_relaxed)){++rain_forbidden_city_calls;throw std::logic_error("city evaluation after freeze");}
    const Hit hit=first_hit(city.scene,ray);if(!hit.valid)return rain_sky(ray.direction);
    RGB value=base_radiance(ctx,hit,-ray.direction);
    if(city.facade[hit.primitive]){
        const Primitive& p=city.scene.primitives[hit.primitive];
        const double u=dot(hit.position-p.origin,unit(p.u));
        const double v=hit.position.y,phase=city.facade_phase[hit.primitive];
        const double col=std::floor(u/.48),row=std::floor(v/.58);
        const double fx=u/.48-col,fy=v/.58-row;
        if(fx>.16&&fx<.80&&fy>.18&&fy<.82&&v>.38){
            // A static facade texture, baked into the outgoing field. Lit panes
            // are local surface emission; they are not extra area-light solves.
            const int hash=(int(col)*17+int(row)*29+int(phase)*7)%13;
            if(hash<4)value=RGB{1.95,1.15,.48}*(.72+.07*hash)+value*.16;
            else value=multiply(value,RGB{.20,.33,.45})+RGB{.10,.19,.29};
        }else if(fy>.92)value=value*.48;
        if(v<.40)value=value*.65;
    }else if(hit.primitive==city.scene.floor){
        const double x=hit.position.x,z=hit.position.z;
        const double stripe=std::abs(x)<.07&&std::fmod(std::abs(z)+.7,2.4)<1.15?1:0;
        if(stripe)value=value*2.7+RGB{.12,.10,.05};
        // Quiet paving and crossing marks make the refraction readable.
        if(std::abs(z+3.0)<.85&&std::abs(x)<1.15&&std::fmod(std::abs(z+3.0),.27)<.13)value=value*2.0;

    }
    return value;
}
