"""Inspection plate and local browser report for the fixed paper image screen."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('output/support_geometry/conv_bounded_sharpening/paper-images')
LABELS={'truth':'Analytic truth','paper-convstar':'Paper CONV*','joint-base':'Joint baseline','cap1':'Bounded · cap 1','cap2':'Bounded · cap 2','cap4':'Bounded · cap 4','cap2-strong':'Bounded · strength 4','lanczos3':'Lanczos-3'}

def run():
 packet=json.loads((ROOT/'results.json').read_text());images=np.load(ROOT/'images.npz')
 fields=list(packet['sourceParameters']);methods=['truth','paper-convstar','joint-base','cap2','cap2-strong','lanczos3']
 fig,axes=plt.subplots(4,6,figsize=(15,10.5),layout='constrained')
 for row,k in enumerate(fields):
  for col,m in enumerate(methods):
   ax=axes[row,col];ax.imshow(images[f'{k}/{m}'],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   if row==0:ax.set_title(LABELS[m],fontsize=11)
   if col==0:ax.set_ylabel(k.capitalize(),fontsize=12)
 fig.suptitle('Original paper fields · same 17 × 17 source · point synthesis',fontsize=16)
 fig.savefig(ROOT/'comparison.png',dpi=150);fig.savefig(ROOT/'comparison.pdf');plt.close(fig)
 fig,axes=plt.subplots(4,4,figsize=(10,10),layout='constrained')
 for row,k in enumerate(fields):
  for col,m in enumerate(['joint-base','cap2','cap2-strong','lanczos3']):
   ax=axes[row,col];im=ax.imshow(images[f'{k}/{m}']-images[f'{k}/truth'],cmap='RdBu_r',vmin=-.1,vmax=.1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   if row==0:ax.set_title(LABELS[m],fontsize=10)
   if col==0:ax.set_ylabel(k.capitalize())
 fig.colorbar(im,ax=axes,shrink=.5,label='Value error: common ±0.10 scale');fig.suptitle('Error against the analytic field',fontsize=15)
 fig.savefig(ROOT/'errors.png',dpi=150);plt.close(fig)
 resolution_metrics=json.loads((ROOT/'resolution-metrics.json').read_text())
 options=''.join(f'<option value="{k}">{v}</option>' for k,v in LABELS.items())
 html=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Bounded CONV — paper image study</title>
<style>body{margin:24px auto;max-width:1160px;padding:0 18px;background:#171b20;color:#e4e9ee;font:15px system-ui}h1{font-size:28px}p{line-height:1.6;max-width:950px}label{display:inline-flex;gap:8px;align-items:center;margin:8px 18px 8px 0}select{background:#28313b;color:inherit;border:1px solid #52606e;padding:8px}main{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0;min-width:0}figcaption{margin:10px 0;font-weight:600}.viewport{height:510px;overflow:auto;background:#101215;border:1px solid #384553}img{display:block;image-rendering:pixelated;width:100%;max-width:none}.metrics{font:13px ui-monospace;white-space:pre-wrap;line-height:1.7;min-height:100px}a{color:#8bdade}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:10px;border-bottom:1px solid #3e4a55}aside{color:#b8c4cf}@media(max-width:750px){main{grid-template-columns:1fr}.viewport{height:350px}}</style>
<h1>Bounded CONV on the paper’s images</h1><p>The same four analytic fields and 17 × 17 nodal source rasters used in the paper. Compare reconstructed values, integrated target pixels, and signed error. Point and area use the same selected output size. Fit and zoom use nearest pixels; a coarse 65 × 65 raster will visibly expose its pixel grid. Images share a fixed [0,1] display range; metric calculations retain unclipped floating-point values.</p>
<label>Field <select id="field"><option>edge</option><option>carrier</option><option>curved</option><option>crossing</option></select></label>
<label>View <select id="view"><option value="point">Point field</option><option value="area">Pixel-area integral</option></select></label>
<label>Output size <select id="resolution"><option value="65">65 × 65</option><option value="513" selected>513 × 513</option></select></label>
<label><input id="error" type="checkbox"> Signed error · ±0.10</label>
<label>Zoom <input id="zoom" type="range" min="100" max="600" step="25" value="100"><span id="zoomvalue">100%</span></label>
<main><figure><label>Left <select id="left">OPTIONS</select></label><div class="viewport" id="lv"><img id="li"></div><figcaption id="lc"></figcaption><div class="metrics" id="lm"></div></figure><figure><label>Right <select id="right">OPTIONS</select></label><div class="viewport" id="rv"><img id="ri"></div><figcaption id="rc"></figcaption><div class="metrics" id="rm"></div></figure></main>
<aside id="notice"></aside><p>The derivative caps are multiples of the base atlas’s largest whole-cell first-derivative Bernstein coefficient. The proposed field is certified on half-cells. Caps 2 and 4 give the same unit-strength result in all four cases. Cap 1 changes only the carrier slightly. Thus the cap-2 derivative limit is not the active restriction for these proposals. Existing range/current constraints and the proposed direction determine the result. Strength 4 tests a larger proposal under cap 2.</p>
<p>Paper CONV* retains the original factor construction. Its area result is not substituted with a different averaging algorithm; the area view offers the joint atlas and Lanczos comparisons. Error views use the reference functional corresponding to the selected point or area view. Source samples are fixed by those variants, while cell-integral preservation is not enabled.</p>
<p><a href="comparison.png">Full comparison plate</a> · <a href="errors.png">Common-scale error plate</a> · <a href="results.json">All numerical results and source hashes</a></p>
<script>const data=DATA,labels=LABELS,resolutionMetrics=RESMETRICS;const $=id=>document.getElementById(id);$('left').value='joint-base';$('right').value='cap2';
function render(){const k=$('field').value,view=$('view').value,size=Number($('resolution').value),error=$('error').checked;let notes=[];for(const side of ['l','r']){const m=$(side==='l'?'left':'right').value;const r=resolutionMetrics.find(r=>r.field===k&&r.method===m&&r.view===view&&r.side===size);const img=$(side+'i');if(!r){img.hidden=true;$(side+'c').textContent=labels[m]+' · area not evaluated';$(side+'m').textContent='Select a joint atlas, analytic truth, or Lanczos for an area comparison.';continue;}img.hidden=false;img.src=k+'-'+m+'-'+view+(error?'-error':'')+'-'+size+'.png?v=2';img.alt=k+' '+labels[m]+' '+view+' '+size;$(side+'c').textContent=labels[m]+' · '+view+(error?' error':'')+' · '+size+' × '+size;$(side+'m').textContent=size+' × '+size+' full-domain MSE '+r.mse.toExponential(4)+'\nSource-range excursion '+r.sourceRangeExcursion.toExponential(3)+'\nMaximum absolute error '+r.maxError.toExponential(3);}if(size===65)notes.push('Both views use a 65 × 65 raster. The enlarged pixel grid is present even in analytic truth.');$('notice').textContent=notes.join(' ');}
for(const id of ['field','view','left','right','resolution','error'])$(id).onchange=render;$('zoom').oninput=()=>{let z=$('zoom').value;$('zoomvalue').textContent=z+'%';for(const id of ['li','ri'])$(id).style.width=z+'%';};let syncing=false;for(const [a,b] of [['lv','rv'],['rv','lv']])$(a).onscroll=()=>{if(syncing)return;syncing=true;$(b).scrollTop=$(a).scrollTop;$(b).scrollLeft=$(a).scrollLeft;requestAnimationFrame(()=>syncing=false)};render();</script></html>'''
 html=html.replace('OPTIONS',options).replace('DATA',json.dumps(packet)).replace('LABELS',json.dumps(LABELS)).replace('RESMETRICS',json.dumps(resolution_metrics))
 (ROOT/'index.html').write_text(html)

if __name__=='__main__':run()
