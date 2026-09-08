struct PackedRadiance { _Float16 r,g,b; };
static_assert(sizeof(PackedRadiance)==6);
struct LensSite { Vec3 point,internal,incident;double entrance_transmission=1;RGB entrance_reflection{}; };
struct FrozenVolumeField {
    int width=0,height=0,normals=0,output_width=0,output_height=0;
    double normal_extent=.65,index=1.333,front_distance=1.20,glass_thickness=.006,volume_thickness=.045;
    Camera camera{};
    std::vector<PackedRadiance> response; // Owned storage exists only during baking.
    std::shared_ptr<void> mapped_owner;
    const PackedRadiance* mapped_response=nullptr;
    std::size_t sample_count()const{return plane_size()*normals*normals;}
    const PackedRadiance* data()const{return mapped_response?mapped_response:response.data();}
    std::vector<LensSite> sites;
    std::uint64_t generation=0,city_queries=0;
    std::size_t plane_size()const{return std::size_t(width)*height;}
    std::size_t sample_index(int x,int y,int sx,int sy)const{
        return (std::size_t(sy)*normals+sx)*plane_size()+std::size_t(y)*width+x;
    }
    LensSite site(double x,double y)const{
        LensSite s;const Ray ray=camera_ray(camera,output_width,output_height,x,y);s.incident=ray.direction;
        const Vec3 n=-camera.forward;
        Vec3 glass,inside;
        update_require(refract(ray.direction,n,1/1.52,glass),"air/glass refraction failed");
        update_require(refract(glass,n,1.52/index,inside),"glass/volume refraction failed");
        const Vec3 entry=ray.origin+ray.direction*(front_distance/dot(ray.direction,camera.forward));
        const Vec3 sheet_back=entry+glass*(glass_thickness/dot(glass,camera.forward));
        s.point=sheet_back+inside*(volume_thickness/dot(inside,camera.forward));s.internal=inside;
        const double a=schlick(-dot(ray.direction,n),1,1.52),b=schlick(-dot(glass,n),1.52,index);
        const double denominator=1-a*b;
        s.entrance_transmission=(1-a)*(1-b)/denominator;
        const double reflected=a+(1-a)*(1-a)*b/denominator;
        s.entrance_reflection=rain_sky(reflect(ray.direction,n))*reflected;
        return s;
    }
    RGB gather(double x,double y,double nx,double ny)const{
        const double gx=std::clamp(x/(output_width-1)*(width-1),0.,double(width-1));
        const double gy=std::clamp(y/(output_height-1)*(height-1),0.,double(height-1));
        const double ax=std::clamp((nx/normal_extent+1)*.5*(normals-1),0.,double(normals-1));
        const double ay=std::clamp((ny/normal_extent+1)*.5*(normals-1),0.,double(normals-1));
        const int ix=std::min(int(gx),width-2),iy=std::min(int(gy),height-2);
        const int ia=std::min(int(ax),normals-2),ib=std::min(int(ay),normals-2);
        const double wx[2]{1-(gx-ix),gx-ix},wy[2]{1-(gy-iy),gy-iy};
        const double wa[2]{1-(ax-ia),ax-ia},wb[2]{1-(ay-ib),ay-ib};
        RGB value{};
        for(int b=0;b<2;++b)for(int a=0;a<2;++a){
            const double angular=wa[a]*wb[b];const std::size_t base=(std::size_t(ib+b)*normals+ia+a)*plane_size()+std::size_t(iy)*width+ix;
            for(int j=0;j<2;++j)for(int i=0;i<2;++i){
                const auto& c=data()[base+std::size_t(j)*width+i];const double w=angular*wx[i]*wy[j];
                value[0]+=float(c.r)*w;value[1]+=float(c.g)*w;value[2]+=float(c.b)*w;
            }
        }return value;
    }
    std::uint64_t checksum()const{
        const auto* bytes=reinterpret_cast<const std::uint8_t*>(data());std::uint64_t h=1469598103934665603ULL;
        for(std::size_t i=0;i<sample_count()*sizeof(PackedRadiance);++i){h^=bytes[i];h*=1099511628211ULL;}return h;
    }
};
RGB evaluate_lens_response(const FrozenVolumeField& field,const LensSite& site,double nx,double ny,
    const CityScene& city,const TraceContext& ctx,std::uint64_t& queries){
    const Vec3 normal=unit(-field.camera.forward+field.camera.right*nx+field.camera.up*ny);
    Vec3 transmitted;
    const bool transmission=refract(site.internal,normal,field.index,transmitted);
    const double fresnel=transmission?schlick(std::clamp(-dot(site.internal,normal),0.,1.),field.index,1):1;
    RGB back=rain_sky(reflect(site.internal,normal))*fresnel;
    if(transmission){++queries;back+=evaluate_city(city,ctx,{site.point+transmitted*3e-4,transmitted})*(1-fresnel);}
    return site.entrance_reflection+multiply(back,RGB{.997,.999,1.0})*site.entrance_transmission;
}
void compile_volume_field(FrozenVolumeField& field,const CityScene& city,const TraceContext& ctx){
    update_require(!field.mapped_owner,"cannot rebuild a read-only response field");
    field.generation=ctx.field.generation;
    field.sites.resize(field.plane_size());
    for(int y=0;y<field.height;++y)for(int x=0;x<field.width;++x)
        field.sites[std::size_t(y)*field.width+x]=field.site(double(x)*(field.output_width-1)/(field.width-1),double(y)*(field.output_height-1)/(field.height-1));
    field.response.resize(field.plane_size()*field.normals*field.normals);
    const unsigned count=std::min(10u,std::max(1u,std::thread::hardware_concurrency()));
    std::atomic<int> next{0},completed{0};std::vector<std::thread> workers;std::vector<std::uint64_t> queries(count);
    for(unsigned worker=0;worker<count;++worker)workers.emplace_back([&,worker]{
        for(;;){const int layer=next.fetch_add(1);if(layer>=field.normals*field.normals)break;
            const double nx=(2.*(layer%field.normals)/(field.normals-1)-1)*field.normal_extent;
            const double ny=(2.*(layer/field.normals)/(field.normals-1)-1)*field.normal_extent;
            for(std::size_t p=0;p<field.plane_size();++p){const RGB value=evaluate_lens_response(field,field.sites[p],nx,ny,city,ctx,queries[worker]);
                field.response[std::size_t(layer)*field.plane_size()+p]={_Float16(value[0]),_Float16(value[1]),_Float16(value[2])};}
            const int done=completed.fetch_add(1)+1;if(done%25==0||done==field.normals*field.normals)
                std::cerr<<"field layers "<<done<<"/"<<field.normals*field.normals<<"\n";
        }
    });
    for(auto& w:workers)w.join();for(auto n:queries)field.city_queries+=n;
}
void save_volume_field(const FrozenVolumeField& field,const std::string& path){
    std::ofstream out(path,std::ios::binary);update_require(bool(out),"cannot write frozen field");
    const std::uint64_t header[]{0x31564c464e494152ULL,std::uint64_t(field.width),std::uint64_t(field.height),std::uint64_t(field.normals),
        std::uint64_t(field.output_width),std::uint64_t(field.output_height),field.generation,field.city_queries};
    out.write(reinterpret_cast<const char*>(header),sizeof(header));out.write(reinterpret_cast<const char*>(&field.normal_extent),sizeof(double));
    out.write(reinterpret_cast<const char*>(field.data()),field.sample_count()*sizeof(PackedRadiance));
    update_require(bool(out),"frozen field write failed");
}
