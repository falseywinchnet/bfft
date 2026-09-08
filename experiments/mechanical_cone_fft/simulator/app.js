(function(){
'use strict';
const M=window.MechanicalFFT,model=M.build(),$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const cyan='#62decf',gold='#e7b970',muted='#93a6b5',metal='#c3d1dc';
const presets={wave:Array.from({length:8},(_,i)=>.65*Math.cos(2*Math.PI*i/8)+.25*Math.sin(4*Math.PI*i/8)),impulse:[1,0,0,0,0,0,0,0],alternating:[1,-1,1,-1,1,-1,1,-1],flat:Array(8).fill(.6),zero:Array(8).fill(0)};
let target=presets.wave.slice(),current=target.slice(),selected=model.bars.find(b=>Math.abs(b.beta-.5)>.01).id,focused=false,playing=false,waveTime=0,lastTime=0,lastReadout=0,raf=0;
const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
const inputs=[],sampleOutputs=[],binRefs=[],barRefs=[],routes=[],portRefs=[];
function svg(tag,attrs={},parent){const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(parent)parent.appendChild(e);return e;}
function text(parent,x,y,value,attrs={}){const t=svg('text',{x,y,...attrs},parent);t.textContent=value;return t;}
function attrs(e,a){for(const[k,v]of Object.entries(a))e.setAttribute(k,v);}
function fmt(x,d=3){return (Math.abs(x)<.5*10**(-d)?0:x).toFixed(d);}
function signed(x){return (x>=0?'+':'')+fmt(x,2);}
const assembly=window.MechanicalAssembly.build(model),Y=window.MechanicalAssembly,yokeRefs=[],inputRefs=[],rulerRefs=[];
function makeInputs(){} // The actual rocker handles are the input controls.
function syncInputLabels(){for(let i=0;i<inputRefs.length;i++){const r=inputRefs[i];r.label.textContent=`x${i}  ${signed(target[i])}`;attrs(r.handle,{'aria-valuenow':fmt(target[i],2),'aria-valuetext':`Sample ${i}: ${fmt(target[i],2)}`});}}
function changeSample(i,value){stopPlay();target[i]=Math.max(-1,Math.min(1,value));document.querySelectorAll('[data-preset]').forEach(b=>b.classList.remove('active'));syncInputLabels();wake();}
function makeDiagram(){const s=$('machine');
 const frame=svg('g',{class:'fixed-frame'},s);
 svg('path',{d:'M24 1300V115H1096V1300M24 1300H1096',fill:'none',stroke:'#405363','stroke-width':12},frame);
 for(const y of [150,410,650,760,1010])svg('line',{x1:24,y1:y,x2:1096,y2:y,stroke:'#2e414f','stroke-width':6},frame);
 text(s,40,38,'INPUT · EIGHT FIXED-PIVOT ROCKERS',{'class':'stage-label'});
 text(s,40,60,'Drag the white endpoint ↑ ↓. Both rail shuttles are attached to the same lever.',{'class':'tiny-label'});
 for(const [y,title,sub] of [[340,'01 ↓ SPLIT','16 midpoint bars'],[570,'02 ↓ ROTATE & COMBINE','24 bars · two levels for three-input averages'],[940,'03 ↓ RESOLVE','4 bars · remaining rails continue on the same rigid yokes'],[1160,'FOURIER OUTPUT ↓','Moving rulers: gold scale + cyan pointer = signed Fourier coefficient']]){text(s,40,y,title,{'class':'stage-label stage-caption'});text(s,40,y+20,sub,{'class':'tiny-label stage-caption'});}
 const transfer=svg('g',{},s);
 for(const y of assembly.yokes){const group=svg('g',{'class':'yoke','data-node':y.node},transfer),color=model.nodes[y.node].row<8?cyan:gold;
  const title=svg('title',{},group);title.textContent=`Rigid yoke ${y.layer} · ${model.nodes[y.node].label} · depth ${y.z}. All branches and slots form one translating body.`;
  const paths=y.segments.map(()=>svg('path',{fill:'none',stroke:color,'stroke-width':2.8,'stroke-linejoin':'round','class':'rigid-member'},group));
  const slots=[y.source,...y.targets].map(p=>{const slot=svg('rect',{x:p.x-Y.SLOT/2,y:p.y-3,width:Y.SLOT,height:6,rx:2,fill:'#101922',stroke:color,'stroke-width':1.8},group);return slot;});
  // A bearing around the source stem supplies the yoke's vertical prismatic joint.
  svg('path',{d:`M${y.source.x-8} ${y.source.y}V${y.source.y+40}H${y.source.x+8}V${y.source.y}`,fill:'none',stroke:'#334b5b','stroke-width':2},frame);
  svg('rect',{x:y.source.x-5,y:y.source.y+26,width:10,height:14,rx:2,fill:'#263946',stroke:'#67818f'},frame);
  yokeRefs.push({group,paths,slots,yoke:y});
 }
 const front=svg('g',{},s);
 for(let i=0;i<8;i++){const cx=112+i*128;
  const group=svg('g',{},front);const label=text(s,cx,97,'',{'text-anchor':'middle',fill:'#edf3f6','font-size':13});
  svg('path',{d:`M${cx-8} 160L${cx} 150L${cx+8} 160Z`,fill:'#8aa0ad'},group);
  const rod=svg('line',{'class':'input-rod',stroke:metal,'stroke-width':7,'stroke-linecap':'round'},group);
  svg('circle',{cx,cy:150,r:5,fill:'#12202a',stroke:'#d9e5ea','stroke-width':2},group);
  const pin=svg('circle',{r:4,fill:gold,stroke:'#0b1017','stroke-width':1.5},group);
  const handle=svg('circle',{r:9,fill:'#f1f6fa',stroke:cyan,'stroke-width':2,role:'slider',tabindex:0,'aria-label':`Sample ${i}`,'aria-valuemin':-1,'aria-valuemax':1,'aria-orientation':'vertical','class':'input-handle'},group);
  text(s,cx-45,202,'+', {fill:cyan,'font-size':14});text(s,cx+43,202,'−',{fill:gold,'font-size':14});
  let dragging=false;
  const move=e=>{const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(s.getScreenCTM().inverse());changeSample(i,2*(150-p.y)/Y.T);};
  handle.addEventListener('pointerdown',e=>{e.preventDefault();dragging=true;handle.setPointerCapture(e.pointerId);move(e);});
  handle.addEventListener('pointermove',e=>{if(dragging)move(e);});
  const release=()=>{dragging=false;};handle.addEventListener('pointerup',release);handle.addEventListener('pointercancel',release);handle.addEventListener('lostpointercapture',release);
  handle.addEventListener('keydown',e=>{let v=target[i];if(['ArrowUp','ArrowRight'].includes(e.key))v+=.01;else if(['ArrowDown','ArrowLeft'].includes(e.key))v-=.01;else if(e.key==='Home')v=-1;else if(e.key==='End')v=1;else if(e.key==='PageUp')v+=.1;else if(e.key==='PageDown')v-=.1;else return;e.preventDefault();changeSample(i,v);});
  inputRefs.push({rod,pin,handle,label});
 }
 for(const b of assembly.bars){const group=svg('g',{'class':'bar','data-bar':b.id},front);
  svg('rect',{x:b.x-3,y:b.y-Y.T/2-5,width:6,height:Y.T+10,rx:2,fill:'#101a22',stroke:'#627684'},group);
  const rod=svg('line',{'class':'rod'},group),A=svg('circle',{r:3.2,'class':'pin'},group),B=svg('circle',{r:3.2,'class':'pin'},group),centre=svg('circle',{r:2.3,fill:'#f3f8fa'},group),tap=svg('circle',{r:3.4,'class':'tap'},group);
  const hit=svg('rect',{x:b.x-Y.L/2-5,y:b.y-Y.T/2-5,width:Y.L+10,height:Y.T+10,'class':'bar-hit'},group);hit.addEventListener('click',()=>chooseBar(b.id));
  svg('title',{},group).textContent=`Bar ${b.id+1}: fixed-length bar, endpoint axle pins in the incoming yoke slots, output axle at the fixed tap.`;
  barRefs[b.id]={group,rod,A,B,tap,centre};
 }
 // Keep captions legible above the projected transfer members.
 for(const label of [...s.querySelectorAll('.stage-caption')])s.appendChild(label);
 const names=['DC','Re X₁','Im X₁','Re X₂','Im X₂','Re X₃','Im X₃','Nyquist'];
 for(const o of assembly.outputs){const group=svg('g',{},front),scale=svg('g',{},group),pointer=svg('g',{},group);
  // Both bodies extend rigidly from their receiving slots; the ruler rides n.
  svg('path',{d:`M${o.x+17} ${o.y}H${o.x+28}V${o.y-38}M${o.x+28} ${o.y-38}V${o.y+38}`,stroke:gold,'stroke-width':3,fill:'none'},scale);
  for(let j=-4;j<=4;j++)svg('line',{x1:o.x+28,y1:o.y+j*Y.T/4,x2:o.x+(j===0?41:35),y2:o.y+j*Y.T/4,stroke:gold,'stroke-width':1.5},scale);
  svg('path',{d:`M${o.x-17} ${o.y}H${o.x+24}M${o.x+18} ${o.y-4}L${o.x+24} ${o.y}L${o.x+18} ${o.y+4}`,stroke:cyan,'stroke-width':3,fill:'none'},pointer);
  const calibration=(o.k===0||o.k===7?Math.sqrt(8):2)/model.scales[3][o.row];
  for(const j of [-1,0,1])text(scale,o.x+43,o.y-j*Y.T+3,j===0?'0':(j>0?'+':'−')+fmt(calibration,1),{fill:gold,'font-size':8});
  text(s,o.x,1200,names[o.k],{'text-anchor':'middle','class':'stage-label'});
  const value=text(s,o.x,1325,'0.000',{'text-anchor':'middle',fill:cyan,'font-size':15,'class':'physical-output','data-coordinate':o.k});
  rulerRefs.push({scale,pointer,value,o});
 }
 text(s,40,1356,'FRONT PROJECTION · Rigid yokes occupy separate depth planes; ringed axle pins join each slot to its lever.',{'class':'tiny-label'});
 syncInputLabels();
}
function updateFocus(){const path=M.ancestors(model,model.bars[selected].out);for(const b of model.bars){barRefs[b.id].group.classList.toggle('selected',b.id===selected);barRefs[b.id].group.classList.toggle('faded',focused&&!path.bars.has(b.id));}for(const r of yokeRefs){r.group.classList.toggle('faded',focused&&!path.nodes.has(r.yoke.node));}}
function renderAssembly(result){const g=Y.geometry(assembly,result,current);
 for(const b of g.bars){const r=barRefs[b.id];attrs(r.rod,{x1:b.A.x,y1:b.A.y,x2:b.B.x,y2:b.B.y});for(const k of ['A','B','tap','centre'])attrs(r[k],{cx:b[k].x,cy:b[k].y});}
 g.yokes.forEach((y,i)=>{const r=yokeRefs[i];y.segments.forEach((points,j)=>r.paths[j].setAttribute('d',points.map((p,k)=>`${k?'L':'M'}${p.x},${p.y}`).join(' ')));[y.source,...y.targets].forEach((p,j)=>attrs(r.slots[j],{x:p.x-Y.SLOT/2,y:p.y-3}));});
 g.inputs.forEach((v,i)=>{const r=inputRefs[i];attrs(r.rod,{x1:v.A.x,y1:v.A.y,x2:v.B.x,y2:v.B.y});attrs(r.handle,{cx:v.A.x,cy:v.A.y});attrs(r.pin,{cx:v.B.x,cy:v.B.y});});
 g.outputs.forEach((v,i)=>{const r=rulerRefs[i];r.scale.setAttribute('transform',`translate(0 ${v.negativeY-v.y})`);r.pointer.setAttribute('transform',`translate(0 ${v.positiveY-v.y})`);const factor=(i===0||i===7?Math.sqrt(8):2)/model.scales[3][v.row];r.value.textContent=fmt((v.negativeY-v.positiveY)/Y.T*factor);});
}

function point(p,l){return {x:l.x+p.x*l.length,y:l.y+.275*l.length-p.y*l.length};}
const leverRefs={};
function makeInspector(){for(const b of model.bars){const option=document.createElement('option');option.value=b.id;option.textContent=`Bar ${String(b.id+1).padStart(2,'0')} · stage ${b.stage+1} · ${Math.abs(b.beta-.5)<.001?'midpoint':'silver tap'}`;$('bar-select').appendChild(option);} $('bar-select').value=selected;$('bar-select').addEventListener('change',e=>chooseBar(Number(e.target.value)));
 const s=$('lever');text(s,150,20,'RIGID LENGTH = 1',{'text-anchor':'middle',fill:muted,'font-size':10});
 for(const x of [43,150,257])svg('line',{x1:x,y1:34,x2:x,y2:188,class:'guide'},s);
 leverRefs.tapGuide=svg('line',{y1:34,y2:188,stroke:'#376e6b','stroke-dasharray':'2 3'},s);
 leverRefs.slots=[0,1,2].map(()=>svg('rect',{width:26,height:10,rx:3,class:'slot'},s));
 leverRefs.rod=svg('line',{stroke:metal,'stroke-width':9,'stroke-linecap':'round'},s);
 leverRefs.highlight=svg('line',{stroke:'#f6fbff','stroke-opacity':.22,'stroke-width':2},s);
 leverRefs.pins=[0,1].map(()=>svg('circle',{r:5,class:'pin'},s));leverRefs.tap=svg('circle',{r:6,class:'tap'},s);
 leverRefs.drop=svg('line',{stroke:cyan,'stroke-width':1.3},s);
 text(s,43,211,'LEFT',{'text-anchor':'middle',fill:muted,'font-size':10});text(s,257,211,'RIGHT',{'text-anchor':'middle',fill:muted,'font-size':10});leverRefs.tapLabel=text(s,150,228,'FIXED TAP',{'text-anchor':'middle',fill:cyan,'font-size':10});
}
function makeBins(){for(let k=0;k<5;k++){const card=document.createElement('article');card.className='bin-card';card.setAttribute('aria-label',`FFT bin ${k}`);card.innerHTML=`<div class="bin-top"><span>k = ${k}</span><small>${k===0?'DC':k===4?'Nyquist':`${k} / 8 cycles`}</small></div><svg class="phasor" viewBox="0 0 180 110" aria-hidden="true"><circle cx="90" cy="54" r="38" fill="none" stroke="#2e4351"/><path d="M42 54H138M90 8V100" stroke="#243745"/><text x="142" y="58" fill="#738b9d" font-size="10">Re</text><text x="95" y="12" fill="#738b9d" font-size="10">Im</text><line class="needle" x1="90" y1="54" x2="90" y2="54"/><circle class="tip" cx="90" cy="54" r="3" fill="${cyan}"/></svg><div class="bin-number" role="status" aria-live="off">0.000 + 0.000i</div><div class="magnitude-track"><div class="magnitude-fill"></div></div><div class="magnitude-label"><span>|X${k}|</span><span class="magnitude-value">0.000</span></div>`;$('bin-grid').appendChild(card);binRefs.push({needle:card.querySelector('.needle'),tip:card.querySelector('.tip'),number:card.querySelector('.bin-number'),fill:card.querySelector('.magnitude-fill'),magnitude:card.querySelector('.magnitude-value')});}}
function chooseBar(id){selected=id;focused=true;$('bar-select').value=id;updateFocus();wake();}
function render(result,readout=true){
 renderAssembly(result);
 const b=model.bars[selected],g=result.geometries[selected],l={x:150,y:111,length:214},A=point(g.A,l),B=point(g.B,l),T=point(g.tap,l),tapX=150+(b.beta-.5)*214;
 attrs(leverRefs.rod,{x1:A.x,y1:A.y,x2:B.x,y2:B.y});attrs(leverRefs.highlight,{x1:A.x,y1:A.y-2,x2:B.x,y2:B.y-2});attrs(leverRefs.pins[0],{cx:A.x,cy:A.y});attrs(leverRefs.pins[1],{cx:B.x,cy:B.y});attrs(leverRefs.tap,{cx:T.x,cy:T.y});attrs(leverRefs.tapGuide,{x1:tapX,x2:tapX});attrs(leverRefs.slots[0],{x:21,y:A.y-5,width:44});attrs(leverRefs.slots[1],{x:235,y:B.y-5,width:44});attrs(leverRefs.slots[2],{x:tapX-13,y:T.y-5});attrs(leverRefs.drop,{x1:T.x,y1:T.y+7,x2:tapX,y2:190});leverRefs.tapLabel.setAttribute('x',tapX);
 if(readout){const a=result.positions[b.a],right=result.positions[b.b];$('weight-a').textContent=fmt((1-b.beta)*a);$('weight-b').textContent=fmt(b.beta*right);$('tap-value').textContent=fmt(g.q);$('weight-a').nextElementSibling.textContent=`${fmt(1-b.beta)} × ${fmt(a)}`;$('weight-b').nextElementSibling.textContent=`${fmt(b.beta)} × ${fmt(right)}`;$('tap-fraction').textContent=Math.abs(b.beta-.5)<.001?'½ = 0.500000':'√2 − 1 = 0.414214';$('bar-length').textContent=Math.hypot(g.B.x-g.A.x,g.B.y-g.A.y).toFixed(6);$('bar-angle').textContent=fmt(g.angle*180/Math.PI,2)+'°';$('lever-copy').textContent=`${model.nodes[b.a].label} and ${model.nodes[b.b].label} position the ends. ${Math.abs(b.beta-.5)<.001?'The midpoint takes their average.':'The silver tap takes 58.58% of the left travel and 41.42% of the right.'}`;
  const ref=M.directDFT(current),err=Math.max(...ref.map((v,k)=>Math.hypot(v.re-result.spectrum[k].re,v.im-result.spectrum[k].im)));$('fft-error').textContent=err<1e-14?'< 10⁻¹⁴':err.toExponential(2);$('fft-error').style.color=err<1e-10?cyan:gold;
  const max=Math.max(1,...result.spectrum.map(z=>Math.hypot(z.re,z.im)));result.spectrum.forEach((z,k)=>{const e=binRefs[k],x=90+38*z.re/max,y=54-38*z.im/max,mag=Math.hypot(z.re,z.im);attrs(e.needle,{x2:x,y2:y});attrs(e.tip,{cx:x,cy:y});e.number.textContent=`${fmt(z.re)} ${z.im<-.0005?'−':'+'} ${fmt(Math.abs(z.im))}i`;e.fill.style.width=`${100*mag/8}%`;e.magnitude.textContent=fmt(mag);});
 }
}
function stopPlay(){playing=false;$('play').setAttribute('aria-pressed','false');$('play').innerHTML='<span aria-hidden="true">▷</span> Travel the wave';}
function tick(time){raf=0;const dt=lastTime?Math.min(.05,(time-lastTime)/1000):.016;lastTime=time;
 if(playing){waveTime+=dt;target=Array.from({length:8},(_,i)=>.65*Math.cos(2*Math.PI*(i/8-waveTime*.12))+.25*Math.sin(4*Math.PI*(i/8-waveTime*.12)));syncInputLabels();}
 const alpha=reduced.matches?1:1-Math.exp(-dt/($('slow').checked?.7:.11));let moving=false;
 current=current.map((x,i)=>{const next=x+alpha*(target[i]-x);if(Math.abs(next-target[i])>1e-7){moving=true;return next;}return target[i];});
 const result=M.evaluate(model,current);const readout=!moving||time-lastReadout>45||playing;if(readout)lastReadout=time;render(result,readout);
 $('motion-status').innerHTML=`<span class="status-dot"></span>${playing?'Wave in motion':moving?'Inputs moving · constraints exact':'At equilibrium'}`;
 if(moving||playing)raf=requestAnimationFrame(tick);else lastTime=0;
}
function wake(){if(!raf)raf=requestAnimationFrame(tick);}
makeInputs();makeDiagram();makeInspector();makeBins();updateFocus();render(M.evaluate(model,current));
$('zoom-machine').addEventListener('click',()=>{const expanded=$('machine').classList.toggle('expanded');$('zoom-machine').setAttribute('aria-pressed',String(expanded));$('zoom-machine').textContent=expanded?'Fit whole machine':'Enlarge joints';});
$('unfocus').addEventListener('click',()=>{focused=false;updateFocus();});
$('play').addEventListener('click',()=>{if(playing)stopPlay();else{playing=true;waveTime=0;$('play').setAttribute('aria-pressed','true');$('play').innerHTML='<span aria-hidden="true">Ⅱ</span> Pause wave';document.querySelectorAll('[data-preset]').forEach(b=>b.classList.remove('active'));}wake();});
$('slow').addEventListener('change',wake);
reduced.addEventListener('change',()=>{if(reduced.matches)stopPlay();wake();});
document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>{stopPlay();target=presets[button.dataset.preset].slice();syncInputLabels();document.querySelectorAll('[data-preset]').forEach(b=>b.classList.toggle('active',b===button));wake();}));
document.addEventListener('visibilitychange',()=>{if(document.hidden){if(raf)cancelAnimationFrame(raf);raf=0;lastTime=0;}else wake();});
/* Read-only state for embedding diagnostics and integration tests. */
window.fourierMachine=Object.freeze({getState:()=>({samples:current.slice(),targets:target.slice(),selectedBar:selected,playing,...M.evaluate(model,current)}),model});
})();
