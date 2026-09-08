"""Verify the frozen recording and create local visual diagnostics."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
p=root/'output/final'
bake=json.loads((p/'precompute.json').read_text());play=json.loads((p/'playback.json').read_text());reference=json.loads((p/'reference.json').read_text());video=json.loads((p/'video.json').read_text())
assert play['city_evaluations_during_playback']==play['static_transport_updates']==play['field_rebuilds']==0
assert bake['field_checksum']==play['field_checksum_before']==play['field_checksum_after']
assert play['normal_texture_updates']==play['frames']==1080
stream=video['streams'][0]
assert stream['width']==1280 and stream['height']==720 and stream['nb_frames']=='1080'
assert stream['avg_frame_rate']==stream['r_frame_rate']=='60/1'
assert float(video['format']['duration'])==18
ms=np.array(play['frame_compute_ms']);assert len(ms)==1080 and np.isfinite(ms).all() and (ms>0).all()
summary={'scene':{'buildings':bake['buildings'],'primitives':bake['primitives'],'transport_nodes':bake['transport_nodes']},
 'precomputation':{'static_transport_seconds':bake['static_transport_ms']/1000,'response_seconds':bake['response_precompute_ms']/1000,
  'surface_sites':bake['field_width']*bake['field_height'],'normal_states_per_site':bake['normal_axis_samples']**2,
  'stored_responses':bake['response_count'],'bytes':bake['field_payload_bytes'],'GiB':bake['field_payload_bytes']/2**30,
  'city_evaluations':bake['city_queries']},
 'playback':{k:v for k,v in play.items() if k!='frame_compute_ms'},
 'frame_budget':{'frames_over_16_667ms':int((ms>1000/60).sum()),'maximum_compute_ms':float(ms.max()),'end_to_end_frames_per_second':1080/(play['sequence_wall_ms']/1000)},
 'reference_comparison':reference,'video':video}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for file in p.glob('*.ppm'):
 Image.open(file).save(file.with_suffix('.png'))
# Storyboard of the encoded sequence's source frames.
fig,axes=plt.subplots(2,2,figsize=(14,8.4),layout='constrained')
for ax,frame,title in zip(axes.flat,[0,356,712,1079],['Clear glass · frozen city','Droplets and trails','Fine rain and mist','Full normal-texture simulation']):
 ax.imshow(Image.open(p/f'frame_{frame}.ppm'));ax.set_title(f'{frame/60:.2f} s — {title}');ax.axis('off')
fig.savefig(p/'storyboard.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
for ax,name,title in zip(axes,['reference_12s_direct','reference_12s_gather','reference_12s_normals'],['Direct city evaluation — diagnostic only','Frozen field gather — recorded path','Dynamic normal texture']):
 ax.imshow(Image.open(p/f'{name}.ppm'));ax.set_title(title);ax.axis('off')
fig.savefig(p/'reference_comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,3.7),layout='constrained')
ax.plot(np.arange(len(ms))/60,ms,lw=.7,color='#176976',label='Normal simulation + gather + tone mapping')
ax.axhline(1000/60,color='#b96239',ls='--',lw=1,label='60 fps frame budget')
ax.set(xlabel='Simulation time (s)',ylabel='Compute time (ms)',title='Frozen city playback · 1280 × 720 · M4 Mini CPU')
ax.set_ylim(0,max(20,float(ms.max())*1.05));ax.grid(alpha=.2);ax.legend(frameon=False)
fig.savefig(p/'frame_times.png',dpi=160);plt.close(fig)
files=['city_rain.cpp','city_scene.hpp','volume_field.hpp','rain_normals.hpp','build_m4.sh','run_m4.sh','summarize.py']
manifest={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
manifest['../native/regime_scene_native.cpp']=hashlib.sha256((root/'../native/regime_scene_native.cpp').read_bytes()).hexdigest()
manifest['video_sha256']=hashlib.sha256((p/'city_rain_frozen_transport_720p60.mp4').read_bytes()).hexdigest()
(p/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['video','playback']},indent=2))
print('Verified 1080 frames at 60 fps, immutable field checksum, and zero frozen city evaluations.')
