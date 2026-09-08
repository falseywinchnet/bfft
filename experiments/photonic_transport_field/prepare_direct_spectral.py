"""Create a comparison translation unit without changing production defaults."""
from pathlib import Path
import sys
s=(Path(__file__).resolve().parent/'native/regime_scene_native.cpp').read_text()
def replace(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a))
    s=s.replace(a,b)
replace('std::uint64_t spectral_path_signature(', '''int direct_spectral_samples=0;
std::uint64_t spectral_path_signature(''')
replace('if(prism_face)for(int band=0;band<3;++band)', 'if(prism_face&&!direct_spectral_samples)for(int band=0;band<3;++band)')
replace('if(mixed_spectrum)result[c]=integrate_spectral_band(local,ray,hit,m,c,signature);', '''if(direct_spectral_samples){
                // Unconditional midpoint quadrature over the entire band.
                // Every sample runs the complete existing dielectric transport.
                double sum=0;for(int k=0;k<direct_spectral_samples;++k){
                    const double coordinate=c-.5+(k+.5)/direct_spectral_samples;
                    const auto sample=trace_primary_dielectric_sample(local,ray,hit,m,c,coordinate);
                    merge_signature(signature,sample.signature);sum+=sample.value;}
                result[c]=sum/direct_spectral_samples;
            }else if(mixed_spectrum)result[c]=integrate_spectral_band(local,ray,hit,m,c,signature);''')
Path(sys.argv[1]).write_text(s)
