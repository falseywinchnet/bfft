"""Generate a diagnostic-only translation unit; never instruments the timed renderer."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parent / 'native'
s = (root / 'regime_scene_native.cpp').read_text()
def replace(a,b):
    global s
    assert s.count(a)==1, (a,s.count(a))
    s=s.replace(a,b)
replace('ExpansionAudit* expansion=nullptr;', 'ExpansionAudit* expansion=nullptr; int classification_stage=0;')
replace('const VisibleEdgeField edges=\n        discover_visible_edges(camera_context,camera,width,height,ownership);', 'TraceContext discovery_context=camera_context; discovery_context.classification_stage=1; const VisibleEdgeField edges=\n        discover_visible_edges(discovery_context,camera,width,height,ownership);')
replace('ctx.optical_queries=boundary_path_capacity?&query_memo:nullptr;\n        std::array<Ray,16>', 'ctx.optical_queries=boundary_path_capacity?&query_memo:nullptr; ctx.classification_stage=2;\n        std::array<Ray,16>')
replace('TraceContext ctx=camera_context;\n        ctx.stats=', 'TraceContext ctx=camera_context; ctx.classification_stage=3;\n        ctx.stats=')
audit=r'''
struct ClassificationKeyHash{std::size_t operator()(const std::array<std::uint64_t,16>& k)const{
    std::size_t h=0;for(auto v:k)h^=v+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);return h;}};
std::array<std::atomic<std::uint64_t>,4> spectral_queries{},band_queries{},region_counts{},leaf_counts{},mixed_counts{};
std::mutex classification_mutex;
std::unordered_map<std::array<std::uint64_t,16>,unsigned,ClassificationKeyHash> classification_keys;
std::array<std::uint64_t,8> classification_calls{},classification_unique{};
void audit_classification(const TraceContext& ctx,const Ray& ray,const Hit& hit,int channel,double coordinate,bool spectral){
    const int stage=ctx.classification_stage+(spectral?4:0);
    const std::array<std::uint64_t,16> key{{std::bit_cast<std::uint64_t>(ray.origin.x),std::bit_cast<std::uint64_t>(ray.origin.y),
        std::bit_cast<std::uint64_t>(ray.origin.z),std::bit_cast<std::uint64_t>(ray.direction.x),std::bit_cast<std::uint64_t>(ray.direction.y),
        std::bit_cast<std::uint64_t>(ray.direction.z),std::uint64_t(hit.primitive),std::uint64_t(channel),std::bit_cast<std::uint64_t>(coordinate),std::bit_cast<std::uint64_t>(hit.position.x),std::bit_cast<std::uint64_t>(hit.position.y),std::bit_cast<std::uint64_t>(hit.position.z),std::bit_cast<std::uint64_t>(hit.normal.x),std::bit_cast<std::uint64_t>(hit.normal.y),std::bit_cast<std::uint64_t>(hit.normal.z),std::uint64_t(hit.front)}};
    std::lock_guard<std::mutex> lock(classification_mutex);++classification_calls[stage];auto& mask=classification_keys[key];
    if(!(mask&(1u<<stage)))++classification_unique[stage];mask|=1u<<stage;
}
'''
replace('std::uint64_t spectral_path_signature(',audit+'\nstd::uint64_t spectral_path_signature(')
replace('const auto response=child->classification_response(ray,hit,1,coordinate,', 'audit_classification(ctx,ray,hit,1,coordinate,true); const auto response=child->classification_response(ray,hit,1,coordinate,')
replace('const auto response=child->classification_response(ray,hit,channel,channel,', 'audit_classification(ctx,ray,hit,channel,channel,false); const auto response=child->classification_response(ray,hit,channel,channel,')
replace('double coordinate){\n    if(ctx.expansion)++ctx.expansion->pixel.probes;', 'double coordinate){\n    ++spectral_queries[ctx.classification_stage]; if(ctx.expansion)++ctx.expansion->pixel.probes;')
replace('if(ctx.expansion)++ctx.expansion->pixel.bands;', '++band_queries[ctx.classification_stage]; if(ctx.expansion)++ctx.expansion->pixel.bands;')
replace('double integral=0;for(const Interval& region:regions)', 'region_counts[ctx.classification_stage]+=regions.size(); leaf_counts[ctx.classification_stage]+=leaves.size(); double integral=0;for(const Interval& region:regions)')
replace('if(ctx.expansion&&mixed_spectrum)++ctx.expansion->pixel.mixed;', 'if(mixed_spectrum)++mixed_counts[ctx.classification_stage]; if(ctx.expansion&&mixed_spectrum)++ctx.expansion->pixel.mixed;')
s='#define main original_main\n'+s+'\n#undef main\n'+r'''
int main(){Scene scene=build_demonstrator_scene("aperture-canyon");scene.use_bvh=false;scene.child_boundaries=true;
    refresh_optical_children(scene);auto beams=compile_beam_field(scene);auto field=compile_transport_field(scene,beams,true);
    TraceContext ctx{scene,beams,field};RenderStats stats;auto pixels=render_visible_edge_field(ctx,800,600,1,stats);
    std::cout<<"{\"stages\":[";for(int i=0;i<8;++i){if(i)std::cout<<",";std::cout<<"{\"stage\":"<<i<<",\"calls\":"<<classification_calls[i]<<",\"unique\":"<<classification_unique[i]<<"}";}
    std::cout<<"],\"spectral\":[";for(int i=0;i<4;++i){if(i)std::cout<<",";std::cout<<"{\"stage\":"<<i<<",\"queries\":"<<spectral_queries[i]<<",\"bands\":"<<band_queries[i]<<",\"regions\":"<<region_counts[i]<<",\"leaves\":"<<leaf_counts[i]<<",\"mixed_rays\":"<<mixed_counts[i]<<"}";}
    std::cout<<"],\"unique_total\":"<<classification_keys.size()<<",\"boundary_pixels\":"<<stats.boundary_pixels<<",\"boundary_topology_samples\":"<<stats.boundary_topology_samples
        <<",\"shared_filter_queries\":"<<stats.shared_filter_queries<<",\"shared_filter_requests\":"<<stats.shared_filter_requests
        <<",\"sampled_boundary_pixels\":"<<stats.sampled_boundary_pixels<<",\"topology_queries\":"<<stats.topology_queries<<"}\n";
}
'''
Path(sys.argv[1]).write_text(s)
