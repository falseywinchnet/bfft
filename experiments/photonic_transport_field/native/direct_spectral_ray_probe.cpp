#define main regime_scene_program_main
#include "/tmp/direct_spectral_scene.cpp"
#undef main
int main(int argc,char** argv){
 Scene scene=build_demonstrator_scene("aperture-canyon");scene.use_bvh=false;scene.child_boundaries=true;refresh_optical_children(scene);
 auto beams=compile_beam_field(scene);auto field=compile_transport_field(scene,beams,true);TraceContext ctx{scene,beams,field};
 const auto camera=make_camera(800,600);std::vector<Vec2> candidates;
 for(int y=8;y<600;y+=8)for(int x=8;x<800;x+=8){auto ray=camera_ray(camera,800,600,x,y);auto hit=first_hit(scene,ray);
  if(hit.valid&&scene.materials[scene.primitives[hit.primitive].material].kind==MaterialKind::Dielectric)candidates.push_back({double(x),double(y)});}
 std::vector<Vec2> sites;for(int i=0;i<48;++i)sites.push_back(candidates[i*(candidates.size()-1)/47]);
 for(Vec2 p:std::array<Vec2,8>{{{270,369},{271,368},{320,330},{296,335},{315,331},{310,332},{291,336},{286,337}}})sites.push_back(p);
 std::ofstream out(argv[1]);out<<std::setprecision(17)<<"{\"rays\":[";bool comma=false;
 for(auto p:sites){auto ray=camera_ray(camera,800,600,p.x,p.y);auto hit=first_hit(scene,ray);
  if(comma)out<<",";comma=true;out<<"{\"x\":"<<p.x<<",\"y\":"<<p.y<<",\"primitive\":"<<hit.primitive<<",\"samples\":[";bool sub=false;
  for(int n:{0,4,8,16,32,64,128,256,512,1024}){direct_spectral_samples=n;std::uint64_t before=0;for(auto c:scene.optical_children)before+=c->classification_evaluations;
   auto value=trace_primary(ctx,ray,hit,nullptr);std::uint64_t after=0;for(auto c:scene.optical_children)after+=c->classification_evaluations;
   update_require(!n||before==after,"direct spectral transport ran classification");
   for(auto v:value)update_require(std::isfinite(v)&&v>=0,"invalid spectral radiance");
   if(sub)out<<",";sub=true;out<<"{\"n\":"<<n<<",\"value\":["<<value[0]<<","<<value[1]<<","<<value[2]<<"],\"classes\":"<<after-before<<"}";
  }out<<"]}";out.flush();}out<<"]}\n";
}
