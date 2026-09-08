#define main regime_scene_program_main
#include "/tmp/reflected_gather_scene.cpp"
#undef main
int main(int argc,char** argv){
 const std::string name=argc>2?argv[2]:"mirror-relay";
 Scene scene=build_demonstrator_scene(name);scene.use_bvh=false;scene.child_boundaries=true;refresh_optical_children(scene);
 auto beams=compile_beam_field(scene);auto field=compile_transport_field(scene,beams,true);TraceContext ctx{scene,beams,field};
 auto camera=make_camera(800,600);direct_spectral_samples=1;
 std::array<CameraSpecularField,2> primary;
 for(auto& f:primary)f=compile_viewer_origin_specular_field(ctx,camera.origin);
 const auto build_start=Clock::now();std::vector<CameraSpecularField> reflected;
 for(const auto& p:scene.primitives){auto kind=scene.materials[p.material].kind;
  if(p.shape!=Shape::Rectangle||kind!=MaterialKind::Mirror)continue;
  const Vec3 origin=camera.origin-p.normal*(2*dot(camera.origin-p.origin,p.normal));
  // Verify the planar virtual-eye identity on an interior point.
  Vec3 point=p.origin+p.u*.37+p.v*.61;point=point-p.normal*dot(point-p.origin,p.normal);
  const Vec3 incoming=unit(point-camera.origin),outgoing=unit(reflect(incoming,p.normal));
  update_require(norm2(unit(point-origin)-outgoing)<1e-24,"virtual-eye reflection identity failed");
  reflected.push_back(compile_viewer_origin_specular_field(ctx,origin));
 }
 const double build_ms=std::chrono::duration<double,std::milli>(Clock::now()-build_start).count();
 std::ofstream out(argv[1]);out<<std::setprecision(17)<<"{\"scene\":\""<<name<<"\",\"origins\":"<<reflected.size()<<",\"allocation_ms\":"<<build_ms<<",\"frames\":[";bool comma=false;
 std::vector<std::uint8_t> reference;std::array<std::vector<std::uint8_t>,2> stable;
 for(int repeat=0;repeat<3;++repeat)for(int slot=0;slot<2;++slot){int mode=(repeat+slot)%2;TraceContext local=ctx;local.reflected_fields=mode?&reflected:nullptr;
  RenderStats stats;std::uint64_t quad_before=0;for(auto& f:reflected)quad_before+=camera_demand_counts(f)[1];auto before=reflected_field_gathers.load();
  const auto start=Clock::now();auto pixels=render_visible_edge_field(local,800,600,1,stats,&camera,&primary[mode]);
  double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();std::uint64_t quad_after=0,samples=0;for(auto& f:reflected){auto c=camera_demand_counts(f);quad_after+=c[1];samples+=c[0];}
  if(reference.empty())reference=pixels;if(stable[mode].empty())stable[mode]=pixels;else update_require(stable[mode]==pixels,"retained frame changed on reuse");double absolute=0,square=0;int maximum=0;std::uint64_t over1=0;
  for(std::size_t i=0;i<pixels.size();++i){int delta=std::abs(int(pixels[i])-reference[i]);absolute+=delta;square+=delta*delta;maximum=std::max(maximum,delta);over1+=delta>1;}
  if(!mode)update_require(maximum==0,"baseline changed between repeats");
  if(comma)out<<",";comma=true;out<<"{\"mode\":"<<mode<<",\"repeat\":"<<repeat<<",\"ms\":"<<ms<<",\"reflected_gathers\":"<<reflected_field_gathers.load()-before
   <<",\"source_quadrature\":"<<stats.trace.emitter_quadrature_samples<<",\"primary_field_quadrature\":"<<stats.camera_specular_quadrature_samples
   <<",\"reflected_field_quadrature\":"<<quad_after-quad_before<<",\"reflected_samples_total\":"<<samples
   <<",\"specular_integrations\":"<<stats.trace.specular_area_calls<<",\"diffuse_fallbacks\":"<<stats.trace.direct_exact_calls
   <<",\"rms_byte_error\":"<<std::sqrt(square/pixels.size())<<",\"max_byte_error\":"<<maximum<<",\"over1_bytes\":"<<over1<<"}";out.flush();
  if(repeat==0){std::ofstream image(std::string(argv[1])+"."+std::to_string(mode)+".ppm",std::ios::binary);image<<"P6\n800 600\n255\n";image.write(reinterpret_cast<const char*>(pixels.data()),pixels.size());}
 }out<<"]}\n";
 // A changed illumination generation must not reuse a stale virtual-view field.
 if(!reflected.empty())for(int id=0;id<int(scene.primitives.size());++id){
  const auto& p=scene.primitives[id];const auto& material=scene.materials[p.material];
  if(p.shape!=Shape::Sphere||material.kind!=MaterialKind::Glossy)continue;
  const Vec3 normal=unit(reflected[0].origin-p.center);const Ray ray{p.center+normal*(p.radius+1),-normal};
  const Hit hit=intersect_primitive(p,id,ray);update_require(hit.valid,"invalid stale-field probe");
  TraceContext check=ctx;check.reflected_fields=&reflected;
  const auto before=reflected_field_gathers.load();++field.generation;
  const RGB value=compiled_specular_area(check,hit,normal,material,false,1);
  update_require(reflected_field_gathers.load()==before,"stale field was gathered");
  for(double x:value)update_require(std::isfinite(x),"invalid stale-field fallback");
  --field.generation;break;
 }

}
