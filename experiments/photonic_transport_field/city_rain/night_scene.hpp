// Distant celestial geometry and a static illuminated city for pinhole capture.
struct NightPoint {Vec3 position;RGB intensity;};
struct NightCity {CityScene city;std::vector<NightPoint> lights;int vehicles=0,beacons=0;};
NightCity make_night_city(){
    NightCity n;n.city=make_rain_city();auto& s=n.city.scene;
    for(auto& m:s.materials){if(m.name=="late afternoon area sky")m.emission={.20,.28,.55};}
    for(auto& l:s.area_lights)l.radiance=s.materials[s.primitives[l.primitive].material].emission;
    const int extra_roof=add_material(s,{"outer roof",MaterialKind::Diffuse,{.20,.25,.32},{},.85,.4});
    std::array<int,4> outer_walls;
    for(int i=0;i<4;++i)outer_walls[i]=add_material(s,{"outer facade",MaterialKind::Diffuse,RGB{.24+i*.055,.29+i*.025,.34-i*.035},{},.88,.5});
    auto outer_building=[&](double x,double z,double h,int phase){
        double w=2.8+.15*(phase%3),d=3.2;int m=outer_walls[phase%4],begin=int(s.primitives.size());
        add_rect(s,"building front",{x-w/2,0,z+d/2},{w,0,0},{0,h,0},{0,0,1},m);
        add_rect(s,"building east",{x+w/2,0,z+d/2},{0,0,-d},{0,h,0},{1,0,0},m);
        add_rect(s,"building back",{x+w/2,0,z-d/2},{-w,0,0},{0,h,0},{0,0,-1},m);
        add_rect(s,"building west",{x-w/2,0,z-d/2},{0,0,d},{0,h,0},{-1,0,0},m);
        add_rect(s,"building roof",{x-w/2,h,z+d/2},{w,0,0},{0,0,-d},{0,1,0},extra_roof);
        n.city.facade.resize(s.primitives.size());n.city.facade_phase.resize(s.primitives.size());
        for(int i=0;i<4;++i){n.city.facade[begin+i]=1;n.city.facade_phase[begin+i]=phase+i*3;}++n.city.buildings;
    };
    for(int row=0;row<8;++row)for(int side:{-1,1})for(int col=0;col<2;++col)
        outer_building(side*(11.6+col*4.2),1-row*5,3.2+.5*((row*7+col*5+(side+1)*3)%13),101+row*11+col*3+(side+1));
    for(int row=5;row<9;++row)for(int col=0;col<4;++col){if((row==5||row==6)&&col==1)continue;
        double x=(col<2?col-2:col-1)*4.2+(col<2?1.0:-1.0);
        outer_building(x,1-row*5,4+.5*((row*3+col*7)%12),203+row*7+col);}
    const int metal=add_material(s,{"dark architectural metal",MaterialKind::Diffuse,{.16,.23,.31},{},.8,.3});
    const int red=add_material(s,{"aviation red",MaterialKind::Emissive,{1,1,1},{12,.025,.01},0,0});
    const int white=add_material(s,{"headlamp",MaterialKind::Emissive,{1,1,1},{12,13,16},0,0});
    const int amber=add_material(s,{"warm streetlamp",MaterialKind::Emissive,{1,1,1},{9,4.5,1.3},0,0});
    auto box=[&](Vec3 p,double w,double h,double d,int m){
        add_rect(s,"detail front",p,{w,0,0},{0,h,0},{0,0,1},m,false);
        add_rect(s,"detail side",p+Vec3{w,0,0},{0,0,-d},{0,h,0},{1,0,0},m,false);
        add_rect(s,"detail back",p+Vec3{w,0,-d},{-w,0,0},{0,h,0},{0,0,-1},m,false);
        add_rect(s,"detail side",p+Vec3{0,0,-d},{0,0,d},{0,h,0},{-1,0,0},m,false);
        add_rect(s,"detail roof",p+Vec3{0,h,0},{w,0,0},{0,0,-d},{0,1,0},m,false);
    };
    // Continue the plane beyond the miniature block, below the original street.
    add_rect(s,"distant ground",{-2000,-.002,-2000},{4000,0,0},{0,0,4000},{0,1,0},s.primitives[s.floor].material,false);
    const auto original=s.primitives;
    for(std::size_t i=0;i<original.size();++i){const auto& p=original[i];if(p.name!="building roof")continue;
        const Vec3 middle=p.origin+p.u*.5+p.v*.5;
        box(middle+Vec3{-.40,.02,.35},.8,.35,.7,metal);
        if(i%3==0){box(middle+Vec3{-.035,.35,.035},.07,.85,.07,metal);
            Vec3 q=middle+Vec3{0,1.23,0};add_sphere(s,"red roof beacon",q,.065,red,false);n.lights.push_back({q,{.28,.001,.0002}});++n.beacons;}
        // Cornices make the roof line legible against the sky.
        box(p.origin+Vec3{-.05,-.08,.05},norm(p.u)+.1,.13,norm(p.v)+.1,metal);
    }
    for(double avenue:{-9.5,0.,9.5})for(int i=0;i<12;++i)for(int side:{-1,1}){
        const Vec3 q{avenue+side*(avenue==0?1.16:.56),1.35,3-i*3.1};box(q+Vec3{-.025,-1.35,.025},.05,1.3,.05,metal);
        add_sphere(s,"streetlamp globe",q,.065,amber,false);n.lights.push_back({q,{1.7,.8,.23}});
    }
    for(double avenue:{-9.5,0.,9.5})for(int i=0;i<12;++i){double x=avenue+(i%2?.43:-.43),z=4-i*2.35+(avenue==0?0:1.1);int direction=i%2?1:-1;
        int paint=add_material(s,{"vehicle paint",MaterialKind::Diffuse,i%3==0?RGB{.5,.07,.035}:i%3==1?RGB{.10,.23,.34}:RGB{.52,.52,.46},{},.85,.2});
        box({x-.21,.10,z+.43},.42,.20,.86,paint);box({x-.17,.30,z+.17},.34,.17,.40,metal);
        for(int side:{-1,1}){Vec3 q{x+side*.145,.235,z+direction*.445};add_sphere(s,"vehicle headlamp",q,.034,white,false);n.lights.push_back({q,{.25,.28,.38}});
            Vec3 rear{x+side*.145,.235,z-direction*.445};add_sphere(s,"vehicle taillamp",rear,.031,red,false);n.lights.push_back({rear,{.12,.001,.0003}});}
        ++n.vehicles;
    }
    n.city.facade.resize(s.primitives.size());n.city.facade_phase.resize(s.primitives.size());build_scene_bvh(s);return n;
}
// Skybox is a cube at +/- 1e7 scene units. Moon and sun lie inside it.
RGB night_background(const Ray& ray){
    const Vec3 moon_center{-90000,115000,-420000};constexpr double radius=25500;
    const Vec3 sun_position{6e6,2.5e6,7e6};
    const Vec3 oc=ray.origin-moon_center;double b=dot(oc,ray.direction),disc=b*b-norm2(oc)+radius*radius;
    if(disc>0){double t=-b-std::sqrt(disc);if(t>0){Vec3 p=ray.origin+ray.direction*t,N=unit(p-moon_center);
        auto noise=[](Vec3 q){int ix=int(std::floor(q.x)),iy=int(std::floor(q.y)),iz=int(std::floor(q.z));double x=q.x-ix,y=q.y-iy,z=q.z-iz;x=x*x*(3-2*x);y=y*y*(3-2*y);z=z*z*(3-2*z);double v=0;
            for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i){std::uint32_t h=std::uint32_t(ix+i)*73856093u^std::uint32_t(iy+j)*19349663u^std::uint32_t(iz+k)*83492791u;h^=h>>13;h*=1274126177u;v+=(h&65535)/65535.*(i?x:1-x)*(j?y:1-y)*(k?z:1-z);}return v;};
        double maria=.42+.32*noise(N*6)+.12*noise(N*17)+.06*noise(N*45);
        Vec3 bump=N;std::uint32_t rng=71831;auto random=[&](){rng=rng*1664525u+1013904223u;return (rng>>8)/double(1u<<24);};
        for(int i=0;i<80;++i){double a=random()*2*pi,z=random()*2-1,r=.015+.11*random()*random();Vec3 center{std::sqrt(1-z*z)*std::cos(a),z,std::sqrt(1-z*z)*std::sin(a)};
            Vec3 delta=N-center;double d=norm(delta);if(d>r*1.6||d<1e-7)continue;double q=d/r;
            double rim=std::exp(-std::pow((q-1)/.16,2)),basin=std::exp(-q*q*3);maria+=.12*rim-.13*basin;
            Vec3 tangent=delta-N*dot(delta,N);bump=bump+tangent/d*(.14*(q-1)/.16*rim-.09*q*basin);}
        bump=unit(bump);double illumination=std::max(0.,dot(bump,unit(sun_position-p)));
        return RGB{1.45,1.55,1.8}*(std::clamp(maria,.2,.95)*illumination+.003);
    }}
    double edge=1e7;Vec3 d=ray.direction;double t=std::min({edge/std::max(std::abs(d.x),1e-12),edge/std::max(std::abs(d.y),1e-12),edge/std::max(std::abs(d.z),1e-12)});
    Vec3 p=ray.origin+d*t;double h=std::clamp(d.y*2.0,0.,1.);RGB sky=RGB{.025,.039,.085}*(1-h)+RGB{.003,.009,.029}*h;
    double cloud=std::sin(d.x*15+d.z*6)+.45*std::sin(d.x*39-d.y*27);sky+=RGB{.009,.013,.023}*(std::max(cloud,0.)*(1-h));
    // Stable fine skybox texels, with rare stars; no frame-dependent noise.
    double u=std::atan2(p.x,-p.z)*1800,v=std::asin(std::clamp(unit(p).y,-1.,1.))*1800;
    int a=int(std::floor(u)),c=int(std::floor(v));std::uint32_t hash=std::uint32_t(a)*73856093u^std::uint32_t(c)*19349663u;hash^=hash>>13;hash*=1274126177u;
    double dx=u-std::floor(u)-.5,dy=v-std::floor(v)-.5;if(hash%1700==0&&d.y>0)sky+=RGB{.7,.8,1.}*(std::exp(-18*(dx*dx+dy*dy))*.6);
    return sky;
}
RGB evaluate_night(const NightCity& n,const TraceContext& ctx,const Ray& ray){
    if(rain_city_frozen){++rain_forbidden_city_calls;throw std::logic_error("night city evaluated after screen freeze");}
    const Hit hit=first_hit(n.city.scene,ray);if(!hit.valid)return night_background(ray);
    const auto& s=n.city.scene;const auto& p=s.primitives[hit.primitive];const auto& material=s.materials[p.material];
    RGB value=base_radiance(ctx,hit,-ray.direction);
    if(material.kind==MaterialKind::Emissive)return value;
    value+=multiply(material.base,RGB{.012,.023,.047})*(.4+.6*std::max(hit.normal.y,0.));
    for(const auto& light:n.lights){Vec3 delta=light.position-hit.position;double r2=norm2(delta);if(r2>110||r2<1e-8)continue;double r=std::sqrt(r2);Vec3 l=delta/r;
        double nd=std::max(dot(hit.normal,l),0.);if(nd<=0)continue;
        if(first_hit(s,{hit.position+hit.normal*2e-4,l},r-.085,hit.primitive).valid)continue;
        value+=multiply(material.base,light.intensity)*(nd/(r2+.04));
        if(hit.primitive==s.floor){Vec3 halfway=unit(l-ray.direction);double gloss=std::pow(std::max(dot(hit.normal,halfway),0.),75);value+=light.intensity*(gloss*5/(r2+.2));}
    }
    if(n.city.facade[hit.primitive]){
        double u=dot(hit.position-p.origin,unit(p.u)),v=hit.position.y;
        double col=std::floor(u/.48),row=std::floor(v/.58),fx=u/.48-col,fy=v/.58-row;
        if(fx>.14&&fx<.83&&fy>.17&&fy<.82&&v>.38){
            int hash=(int(col)*17+int(row)*29+int(n.city.facade_phase[hit.primitive])*7)%23;
            if(hash<12){RGB bulb=hash%4==0?RGB{.75,1.35,2.}:RGB{2.5,1.35,.48};
                // An analytic room behind each aperture: bounded walls and an
                // interior point lamp with a small visible luminous globe.
                constexpr double ww=.69*.48,hh=.65*.58,depth=.28;
                Vec3 q{(fx-.14)*.48,(fy-.17)*.58,0};
                Vec3 d{dot(ray.direction,unit(p.u)),ray.direction.y,-dot(ray.direction,hit.normal)};
                double t=depth/std::max(d.z,1e-8);Vec3 wallnormal{0,0,-1};
                auto wall=[&](double candidate,Vec3 normal){if(candidate>0&&candidate<t){t=candidate;wallnormal=normal;}};
                if(d.x>1e-8)wall((ww-q.x)/d.x,{-1,0,0});else if(d.x< -1e-8)wall(-q.x/d.x,{1,0,0});
                if(d.y>1e-8)wall((hh-q.y)/d.y,{0,-1,0});else if(d.y< -1e-8)wall(-q.y/d.y,{0,1,0});
                Vec3 lamp{ww*.5,hh*.77,.08},surface=q+d*t,delta=lamp-surface;double r2=norm2(delta);
                double illumination=.16+.09*std::max(dot(wallnormal,unit(delta)),0.)/(r2+.025);
                Vec3 oc=q-lamp;double a=norm2(d),b=dot(oc,d),disc=b*b-a*(norm2(oc)-.0004);
                if(disc>0&&(-b-std::sqrt(disc))/a>0&&(-b-std::sqrt(disc))/a<t)illumination=3.5;
                double curtain=hash%3==0?.65+.35*std::pow(std::sin(fx*24),2):1.;
                value=bulb*((.30+.035*hash)*curtain*illumination)+value*.12;
                if(std::abs(fx-.485)<.025||std::abs(fy-.49)<.018)value=value*.22;
            }else value=RGB{.012,.030,.057}+value*.22;
        }else if(fy>.92)value=value*.45;
    }
    if(hit.primitive==s.floor){double worldx=hit.position.x,x=worldx-9.5*std::round(worldx/9.5),z=hit.position.z;
        if(std::abs(x)<.035&&std::fmod(std::abs(z)+.7,2.4)<1.15)value+=RGB{.16,.12,.045};
        if(std::abs(x)>.93&&std::abs(x)<.96)value+=RGB{.13,.15,.18};
        if(std::abs(z+3)<.85&&std::abs(x)<1.15&&std::fmod(std::abs(z+3),.27)<.13)value+=RGB{.17,.19,.22};
    }
    double fog=1-std::exp(-hit.t*.006);return value*(1-fog)+RGB{.025,.040,.073}*fog;
}
