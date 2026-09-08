"""Add retained planar virtual-view fields to the N=1 comparison source."""
from pathlib import Path
import sys
s=Path('/tmp/direct_spectral_scene.cpp').read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
replace('ExpansionAudit* expansion=nullptr;', 'ExpansionAudit* expansion=nullptr;\n    const std::vector<CameraSpecularField>* reflected_fields=nullptr;')
replace('RGB compiled_specular_area(', 'std::atomic<std::uint64_t> reflected_field_gathers{0};\nRGB compiled_specular_area(')
replace('if(atlas_index<0)return specular_area(ctx,hit,view,material,camera_primary,path_weight);', '''if(atlas_index<0){
        if(ctx.reflected_fields)for(const auto& field:*ctx.reflected_fields){
            if(field.field_generation!=ctx.field.generation)continue;
            const int id=hit.primitive>=0&&hit.primitive<int(field.by_primitive.size())?field.by_primitive[hit.primitive]:-1;
            if(id<0||dot(unit(field.origin-hit.position),view)<=1-1e-11)continue;
            TraceContext reflected=ctx;reflected.camera_specular=&field;
            reflected_field_gathers.fetch_add(1,std::memory_order_relaxed);
            if(ctx.stats)++ctx.stats->camera_specular_gathers;
            return gather_camera_specular(reflected,*field.demand_atlas[id],hit.position);
        }
        return specular_area(ctx,hit,view,material,camera_primary,path_weight);
    }''')
Path(sys.argv[1]).write_text(s)
