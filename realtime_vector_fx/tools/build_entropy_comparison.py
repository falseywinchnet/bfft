#!/usr/bin/env python3
"""Build isolated instrumented before/after OBS modules, never install them.

Run on the M4 with the same OBS headers and framework as build_obs_macos.sh.
A deterministic *test-only* profile clock lets the comparison require identical
frame hashes, without confusing faster execution with a different elapsed time.
Production code has no clock override or benchmark selector.
"""
import argparse
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--out',default='/tmp/entropy-gpu-comparison')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
obs=Path(os.environ.get('OBS_SOURCE_DIR','/tmp/entropy-obs-deps/obs-studio-32.2.1'))
config=Path(os.environ.get('OBS_CONFIG_INCLUDE_DIR','/tmp/entropy-obs-deps'))
simde=Path(os.environ.get('SIMDE_INCLUDE_DIR','/tmp/entropy-obs-deps/simde-0.8.2'))
app=Path(os.environ.get('OBS_APP','/Applications/OBS.app'))
framework=app/'Contents/Frameworks/libobs.framework/libobs'
flags=['-std=c++17','-O3','-DNDEBUG','-fPIC','-I'+str(config),'-I'+str(obs/'libobs'),'-I'+str(simde)]
def run(args):subprocess.run(args,check=True)
profiler=r'''
#include <cstdio>
struct ESMeasurement {
    static double sums[7];static unsigned counts[7];
    int slot;std::chrono::steady_clock::time_point start;
    explicit ESMeasurement(int s):slot(s),start(std::chrono::steady_clock::now()){}
    ~ESMeasurement(){sums[slot]+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();++counts[slot];}
};
double ESMeasurement::sums[7]{};unsigned ESMeasurement::counts[7]{};
template<class Fn> auto es_measure(int slot,Fn fn)->decltype(fn()){ESMeasurement t(slot);return fn();}
void es_report(){const char* labels[]={"capture","stage","analyze","advance","pack","upload","draw"};
    for(int i=0;i<7;++i)std::printf("PHASE %s %u %.6f\n",labels[i],ESMeasurement::counts[i],ESMeasurement::sums[i]);}
'''
for name in ['before','after']:
    work=out/name;inc=work/'include/rvfx';inc.mkdir(parents=True,exist_ok=True)
    if name=='before':
        header=(root/'tests/reference/entropy_stretch_v1.hpp').read_text().replace('rvfx::entropy_v1','rvfx::entropy')
        core=(root/'tests/reference/entropy_stretch_v1.cpp').read_text().replace('rvfx::entropy_v1','rvfx::entropy').replace('"entropy_stretch_v1.hpp"','"rvfx/entropy_stretch.hpp"')
        filt=(root/'tests/reference/entropy_filter_v1.cpp').read_text()
    else:
        header=(root/'include/rvfx/entropy_stretch.hpp').read_text()
        core=(root/'src/entropy_stretch.cpp').read_text()
        filt=(root/'obs/entropy-filter.cpp').read_text()
    (inc/'entropy_stretch.hpp').write_text(header);(work/'core.cpp').write_text(core)
    filt=filt.replace('namespace {',profiler+'\nnamespace {',1)
    filt=filt.replace('auto now=std::chrono::steady_clock::now();',
       'auto now=f->previous.time_since_epoch().count()?f->previous+std::chrono::nanoseconds(16666667):std::chrono::steady_clock::time_point(std::chrono::seconds(1));')
    calls=[(0,'capture(f,target)'),(1,'gs_stage_texture(f->stages[f->slot],gs_texrender_get_texture(f->lattice))'),
       (2,'f->profile.analyze(mapped,f->aw,f->ah,stride,s.config)'),(3,'f->profile.advance(dt,s.config)'),
       (3,'f->profile.advance_and_pack(dt,s.config,f->upload)'),(4,'pack_profile(f)'),
       (5,'gs_texture_set_image(f->lut,reinterpret_cast<const uint8_t*>(f->upload.data()),1089*4*sizeof(uint16_t),false)'),
       (6,'gs_draw_sprite(gs_texrender_get_texture(f->capture),0,w,h)')]
    for slot,call in calls:filt=filt.replace(call,'es_measure('+str(slot)+',[&]{return '+call+';})')
    filt=filt.replace('obs_leave_graphics();delete f;}','obs_leave_graphics();es_report();delete f;}')
    (work/'filter.cpp').write_text(filt)
    objects=[]
    for file in [root/'src/engine.cpp',root/'obs/plugin-main.cpp',root/'obs/gpu-filter.cpp',work/'core.cpp',work/'filter.cpp']:
        obj=work/(file.stem+'.o');run(['c++',*flags,'-I'+str(work/'include'),'-I'+str(root/'include'),'-c',str(file),'-o',str(obj)]);objects.append(str(obj))
    run(['c++','-bundle',*objects,str(framework),'-Wl,-rpath,'+str(app/'Contents/Frameworks'),'-o',str(work/'realtime-vector-fx.so')])
    run(['/usr/bin/codesign','--force','--sign','-',str(work/'realtime-vector-fx.so')])
# The driver shares the exact existing smoke source/renderer.
run(['clang++',*flags,'-DES_SMOKE_WIDTH=1920','-DES_SMOKE_HEIGHT=1080','-I'+str(root/'include'),'-c',str(root/'tools/entropy_gpu_benchmark.mm'),'-o',str(out/'driver.o')])
run(['clang++',str(out/'driver.o'),str(out/'after/core.o'),str(framework),'-framework','AppKit','-Wl,-rpath,'+str(app/'Contents/Frameworks'),'-o',str(out/'driver')])
print(out)
