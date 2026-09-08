"""Validate the recorded screen deformation and produce readable diagnostics."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
out = root / 'output/night_screen'
bake = json.loads((out / 'bake.json').read_text())
play = json.loads((out / 'playback.json').read_text())
video = json.loads((out / 'video.json').read_text())
assert bake['checksum'] == play['checksum_before'] == play['checksum_after']
assert play['city_evaluations'] == play['screen_updates'] == 0
assert play['frames'] == 1080 and play['fps'] == 60 and play['duration'] == 18
stream = video['streams'][0]
assert (stream['width'], stream['height'], stream['nb_frames']) == (1280, 720, '1080')
assert stream['r_frame_rate'] == stream['avg_frame_rate'] == '60/1'
assert float(video['format']['duration']) == 18
ms = np.asarray(play['frame_ms'])
assert ms.shape == (1080,) and np.isfinite(ms).all() and (ms > 0).all()
assert (out / 'illumination.screen').stat().st_size == bake['screen_bytes'] + 28
summary = {
    'scene_and_capture': bake,
    'playback': {k: v for k, v in play.items() if k != 'frame_ms'},
    'frames_over_60fps_budget': int(np.sum(ms > 1000/60)),
    'mean_compute_ms': float(ms.mean()),
    'prior_4d_field_bytes': 3333562806,
    'reduction_factor_base': 3333562806 / bake['screen_bytes'],
    'reduction_factor_with_mips': 3333562806 / bake['screen_with_mips_bytes'],
    'video': video,
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for path in out.glob('*.ppm'):
    Image.open(path).save(path.with_suffix('.png'))
fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), layout='constrained')
for ax, frame, title in zip(axes.flat, [0, 360, 720, 1079],
                          ['Captured night city', 'Large and small droplets',
                           'Mist and trails', 'Full rain deformation']):
    ax.imshow(Image.open(out / f'frame_{frame}.png'))
    ax.set_title(f'{frame/60:.2f} s · {title}')
    ax.axis('off')
fig.savefig(out / 'storyboard.png', dpi=150)
plt.close(fig)
fig, ax = plt.subplots(figsize=(11, 3.5), layout='constrained')
ax.plot(np.arange(1080)/60, ms, color='#256978', lw=.7,
        label='Rain + screen footprint + lookup + tone mapping')
ax.axhline(1000/60, color='#b66043', ls='--', lw=1, label='60 fps budget')
ax.set(xlabel='Animation time (s)', ylabel='Compute time (ms)',
       title='Frozen illumination screen · M4 Mini CPU · 1280 × 720')
ax.legend(frameon=False)
ax.grid(alpha=.2)
fig.savefig(out / 'frame_times.png', dpi=160)
plt.close(fig)
files = ['night_screen.cpp', 'night_scene.hpp', 'city_rain.cpp', 'city_scene.hpp',
         'rain_normals.hpp', 'volume_field.hpp', 'run_night_screen.sh',
         'summarize_night_screen.py', 'NIGHT_SCREEN.md',
         '../native/regime_scene_native.cpp', '../native/retained_transport.cpp',
         '../native/retained_transport.h',
         '../../../standalone_conv_resize_demo/native/conv_native.c']
manifest = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files}
for name in ['night_city_screen_rain_720p60.mp4', 'illumination.screen']:
    manifest[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
(out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'video'}, indent=2))
print('Verified: 1080 frames, 60 fps, 18 seconds, frozen screen and zero city calls.')
