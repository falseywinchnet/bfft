// Focused geometry-only diagnostic; no transport compilation required.
#define main regime_scene_program_main
#include "regime_scene_native.cpp"
#undef main
std::vector<int> spectral_words(const TraceContext& ctx,const Ray& primary_ray,const Hit& primary_hit,
    double coordinate){
    if(!primary_hit.valid)return {};
    if(ctx.expansion)++ctx.expansion->pixel.probes;
    std::vector<int> signature;signature.push_back(primary_hit.primitive);Ray ray=primary_ray;Hit hit=primary_hit;
    for(int depth=0;depth<8;++depth){const Material& material=
            ctx.scene.materials[ctx.scene.primitives[hit.primitive].material];Vec3 direction{};
        if(material.kind==MaterialKind::Mirror||material.kind==MaterialKind::Metal||material.kind==MaterialKind::Glossy)
            direction=unit(reflect(ray.direction,hit.normal));
        else if(material.kind==MaterialKind::Dielectric){const Vec3 reflected=unit(reflect(ray.direction,hit.normal));
            const Hit reflected_hit=optical_first_hit(ctx,{hit.position+reflected*3e-4,reflected},
                std::numeric_limits<double>::infinity(),hit.primitive);
            if(reflected_hit.valid)signature.push_back(1000+reflected_hit.primitive);
            if(material.thin)direction=ray.direction;else{const double index=spectral_ior(material,coordinate);
                const double ni=hit.front?1:index,nt=hit.front?index:1;
                if(!refract(ray.direction,hit.normal,ni/nt,direction))direction=reflected;
                else{const VolumeChord chord=jelly_volume_chord(ctx.scene,hit,direction,index);if(chord.valid){
                    ray=chord.exit_ray;hit=optical_first_hit(ctx,ray,std::numeric_limits<double>::infinity(),hit.primitive);
                    if(!hit.valid)break;signature.push_back(hit.primitive);continue;}}}}
        else break;ray={hit.position+direction*3e-4,direction};
        hit=optical_first_hit(ctx,ray,std::numeric_limits<double>::infinity(),hit.primitive);if(!hit.valid)break;
        signature.push_back(hit.primitive);}
    return signature;
}

int main(int argc,char** argv){
    const int px=argc>1?std::stoi(argv[1]):275,py=argc>2?std::stoi(argv[2]):324;
    if(px<0||px>=800||py<0||py>=600){std::cerr<<"pixel must be inside 800x600\n";return 2;}
    Scene scene=build_demonstrator_scene("aperture-canyon");scene.use_bvh=false;
    BeamField beams;TransportField field;TraceContext ctx{scene,beams,field};Camera camera=make_camera(800,600);
    for(int i=0;i<int(scene.primitives.size());++i)std::cout<<"primitive "<<i<<" "<<scene.primitives[i].name<<"\n";
    struct Bucket{TerminalLabel label;double x=0,y=0,fx=0,fy=0;int n=0;};std::vector<Bucket> buckets;
    for(int sy=0;sy<16;++sy)for(int sx=0;sx<16;++sx){double x=px-.5+(sx+.5)/16,y=py-.5+(sy+.5)/16;
        Ray ray=camera_ray(camera,800,600,x,y);Hit hit=first_hit(scene,ray);TerminalLabel label{hit.primitive,visible_path_signature(ctx,ray,hit)};
        auto it=std::find_if(buckets.begin(),buckets.end(),[&](const Bucket& b){return b.label==label;});
        if(it==buckets.end()){buckets.push_back({label,0,0,x,y,0});it=buckets.end()-1;}it->x+=x;it->y+=y;++it->n;}
    std::cout<<std::setprecision(17)<<"buckets "<<buckets.size()<<"\n";
    int bi=0;for(const auto& b:buckets){double x=b.x/b.n,y=b.y/b.n;
        Ray ray=camera_ray(camera,800,600,x,y);Hit hit=first_hit(scene,ray);
        if(!(TerminalLabel{hit.primitive,visible_path_signature(ctx,ray,hit)}==b.label)){x=b.fx;y=b.fy;ray=camera_ray(camera,800,600,x,y);hit=first_hit(scene,ray);}
        std::vector<int> previous;int count=0;std::cout<<"bucket "<<bi++<<" "<<x<<" "<<y<<" weight "<<b.n<<"\n";
        for(int i=0;i<=24576;++i){double t=-.5+double(i)/8192;auto words=spectral_words(ctx,ray,hit,t);
            if(words!=previous){{std::cout<<t;for(int w:words)std::cout<<" "<<w;std::cout<<"\n";}previous=words;++count;}}
        std::cout<<"transitions "<<count<<"\n";
    }
}
