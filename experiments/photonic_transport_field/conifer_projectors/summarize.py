"""Local presentation and a planar-reprojection counterexample from saved data."""
from pathlib import Path
import json,struct
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output'
for p in OUT.glob('*.ppm'):Image.open(p).save(p.with_suffix('.png'))
bake=json.loads((OUT/'bake.json').read_text());tests=json.loads((OUT/'tests.json').read_text());replay=json.loads((OUT/'replay.json').read_text())
raw=(OUT/'conifer.sheet').read_bytes();header=struct.unpack_from('<9Q',raw);W,H=header[1:3];data=np.frombuffer(raw,dtype='<f4',offset=72).reshape(H,W,8)
linear=data[:,:,1:4]+data[:,:,0,None]*data[:,:,4:7]
def frame(eye):
 eye=np.array(eye,dtype=float);target=np.array([0,3.3,1]);forward=target-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,forward);return eye,target,forward,right,up
a,t,fa,ra,ua=frame([10,5.2,13]);b,_,fb,rb,ub=frame([12,5.2,13]);w,h=320,180
xx,yy=np.meshgrid(np.arange(w)+.5,np.arange(h)+.5);directions=fb+(2*xx[...,None]/w-1)*(.48*w/h)*rb+(1-2*yy[...,None]/h)*.48*ub
distance=np.dot(t-b,fa)/np.sum(directions*fa,axis=-1);p=b+directions*distance[...,None];delta=p-a;z=np.sum(delta*fa,axis=-1)
gx=(np.sum(delta*ra,axis=-1)/z/(.48*W/H)+1)*.5*W-.5;gy=(1-np.sum(delta*ua,axis=-1)/z/.48)*.5*H-.5
valid=(gx>=0)&(gy>=0)&(gx<W-1)&(gy<H-1);x=np.clip(np.floor(gx).astype(int),0,W-2);y=np.clip(np.floor(gy).astype(int),0,H-2);u=np.clip(gx-x,0,1)[...,None];v=np.clip(gy-y,0,1)[...,None]
warped=(1-v)*((1-u)*linear[y,x]+u*linear[y,x+1])+v*((1-u)*linear[y+1,x]+u*linear[y+1,x+1]);tone=(255*np.power(np.clip(1-np.exp(-.75*np.maximum(warped,0)),0,1),1/2.2)).astype('uint8');tone[~valid]=[35,20,35]
Image.fromarray(tone).save(OUT/'planar_reprojection.png');actual=np.asarray(Image.open(OUT/'moved_camera.ppm'));err=np.abs(tone.astype(float)-actual.astype(float))[valid];warp_stats={'coverage':float(valid.mean()),'mae_bytes':float(err.mean()),'rmse_bytes':float(np.sqrt(np.mean(err**2))),'p99_bytes':float(np.quantile(err,.99)),'maximum_bytes':float(err.max()),'mapping':'Reproject the saved linear-RGB sheet through the target plane into the moved pinhole. Errors measured only inside the captured domain; purple is unavailable support.'};(OUT/'planar_reprojection.json').write_text(json.dumps(warp_stats,indent=2))
assert np.array_equal(np.asarray(Image.open(OUT/'conifer_smoke.ppm')),np.asarray(Image.open(OUT/'replayed.ppm'))),'standalone replay differs from capture'
gates={'source_projector_diagnostic_pass':bake['projector_vs_direct']['rmse_bytes']<.5 and bake['projector_vs_direct']['p99_bytes']<=1 and bake['projector_vs_direct']['max_bytes']<=8,'diffuse_mean_probe_pass_10_percent':bake['diffuse_grid_probe_relative_l2']<=.1,'standalone_replay_identical':True,'zero_transport_replay':replay['geometry_queries']==0 and replay['density_evaluations']==0 and not replay['scene_constructed'],'scope':'Empirical gates for this fixed conifer, medium and cameras. The indirect probe is mean radiance, not a complete angular or GI certificate. Camera quadrature residual is reported separately.'};assert all(v for v in gates.values() if isinstance(v,bool));gates['volume_surface_refinement_relative_l2']=bake['volume_surface_probe_relative_l2'];gates['volume_surface_refinement_pass_10_percent']=bake['volume_surface_probe_relative_l2']<=.1;gates['production_acceleration_validated']=False;(OUT/'validation.json').write_text(json.dumps(gates,indent=2))
labels=[('conifer_smoke','ONE TREE · ONE LIGHT · ONE SMOKE FIELD'),('surface_only','REGISTERED SURFACES WITHOUT THE CLOUD VIEW'),('scattered_light','SCATTERED LIGHT S'),('transmission','TRANSMISSION T'),('volume_light_on_surfaces','VOLUME LIGHT REGISTERED ON ACTUAL SURFACES'),('additive_only_wrong','ADDITION ALONE · EXTINCTION OMITTED')]
canvas=Image.new('RGB',(1600,1515),'#e9e5dc');draw=ImageDraw.Draw(canvas)
try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
except OSError:font=ImageFont.load_default()
for i,(name,label) in enumerate(labels):
 im=Image.open(OUT/f'{name}.png');im.thumbnail((784,441));x=8+(i%2)*800;y=8+(i//2)*505;canvas.paste(im,(x,y));draw.text((x+4,y+452),label,fill='#243431',font=font)
canvas.save(OUT/'contact_sheet.png')
review='''<!doctype html><meta charset="utf-8"><title>Conifer · registered smoke volume</title><style>body{background:#172023;color:#eee9df;font:17px system-ui;max-width:1400px;margin:40px auto;padding:0 25px}h1{font-size:36px}p{max-width:950px;line-height:1.6}img{width:100%;border-radius:6px}figure{margin:35px 0}figcaption{padding-top:10px}a{color:#d9b975}nav{display:flex;gap:24px;flex-wrap:wrap}</style><h1>A conifer in a registered smoke field</h1><p>One physical white point light. Diffuse hits end image-bearing reflection geometry and register their own outgoing energy. The smoke also deposits light on the tree and ground, which register their own outgoing energy. The cloud stores transmission and scattered radiance: L = T × background + S. These are saved native BFFT CPU renders.</p><nav><a href="../README.md">Method and limits</a><a href="bake.json">Capture measurements</a><a href="replay.json">Independent replay measurements</a><a href="validation.json">Validation scope</a></nav>'''
for name,label in labels:review+=f'<figure><img src="{name}.png" alt="{label}"><figcaption>{label}</figcaption></figure>'
review+='<p>A new pinhole needs another valid angular registration. Below: a planar reprojection of the first sheet, followed by the newly captured camera view. Purple marks unsupported parts of the original capture.</p>'
for name in ['planar_reprojection','moved_camera']:review+=f'<figure><img src="{name}.png" alt="{name}"></figure>'
review+='<p>The replay executable loads only the saved sheet. It constructs no scene and performs no density or geometry queries. The source and diffuse registrations were computed once during capture; the demo includes point → tree → medium → camera and point → medium → surface → camera paths, not full multiple scattering. The new surface refinement difference is 10.90%; first-still acceleration remains unvalidated.</p>'
(OUT/'review.html').write_text(review)
print(json.dumps({'gates':gates,'planar_reprojection':warp_stats,'replay':replay},indent=2))
