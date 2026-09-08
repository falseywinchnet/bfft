'use strict';
const assert=require('node:assert/strict'),M=require('./mechanics.js'),A=require('./assembly.js');
const model=M.build(),assembly=A.build(model);
let seed=8675309,cases=0,maxJointError=0,maxRigidError=0,maxReadoutError=0;
const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32*2-1;};
function near(a,b){assert.ok(Math.abs(a-b)<2e-11,`${a} != ${b}`);}
function check(samples){
 const result=M.evaluate(model,samples),g=A.geometry(assembly,result,samples),oracle=M.directDFT(samples);
 const physicalPacked=[oracle[0].re,oracle[1].re,oracle[1].im,oracle[2].re,oracle[2].im,oracle[3].re,oracle[3].im,oracle[4].re];
 function slot(pin,slot){const error=Math.abs(pin.y-slot.y);maxJointError=Math.max(maxJointError,error);near(pin.y,slot.y);assert.ok(Math.abs(pin.x-slot.x)<A.SLOT/2-1,'Pin must remain within its material slot, with clearance');}
 g.inputs.forEach((v,i)=>{near(Math.hypot(v.B.x-v.A.x,v.B.y-v.A.y),A.INPUT_L);near((v.A.x+v.B.x)/2,v.pivot.x);near((v.A.y+v.B.y)/2,v.pivot.y);near((150-v.A.y)/A.T+.5,result.positions[i]);near((150-v.B.y)/A.T+.5,result.positions[i+8]);});
 g.yokes.forEach((y,i)=>{const ref=assembly.yokes[i];assert.ok(y.targets.length>0,'No floating output member');
  const owner=model.bars.find(b=>b.out===y.node);
  const pin=owner?g.bars[owner.id].tap:y.node<8?g.inputs[y.node].A:g.inputs[y.node-8].B;
  slot(pin,y.source);
  y.targets.forEach(p=>{if('bar'in p)slot(g.bars[p.bar][p.pin],p);else {const out=g.outputs[p.output];near(p.y,p.rail==='positive'?out.positiveY:out.negativeY);}});
  y.segments.forEach((points,j)=>{near(points[0].x,y.source.x);near(points[0].y,y.source.y);near(points.at(-1).x,y.targets[j].x);near(points.at(-1).y,y.targets[j].y);
   points.forEach((p,k)=>{near(p.x,ref.segments[j][k].x);near(p.y-ref.segments[j][k].y,y.dy);if(k){const length=Math.hypot(p.x-points[k-1].x,p.y-points[k-1].y),r=ref.segments[j],expected=Math.hypot(r[k].x-r[k-1].x,r[k].y-r[k-1].y);maxRigidError=Math.max(maxRigidError,Math.abs(length-expected));near(length,expected);}});
  });
 });
 g.bars.forEach(b=>{near(Math.hypot(b.B.x-b.A.x,b.B.y-b.A.y),A.L);near(b.centre.x,b.x);});
 g.outputs.forEach((o,i)=>{const factor=(i===0||i===7?Math.sqrt(8):2)/model.scales[3][o.row];const v=(o.negativeY-o.positiveY)/A.T*factor;maxReadoutError=Math.max(maxReadoutError,Math.abs(v-physicalPacked[i]));near(v,physicalPacked[i]);});
 cases++;
}
for(let mask=0;mask<256;mask++)check(Array.from({length:8},(_,i)=>(mask>>i&1)?1:-1));
for(let j=0;j<1000;j++)check(Array.from({length:8},random));
check(Array(8).fill(0));
assert.equal(assembly.yokes.length,60);assert.equal(assembly.outputs.length,8);
console.log(JSON.stringify({cases,input_rockers:8,interpolation_bars:44,connected_yokes:60,copy_operations_using_existing_yokes:12,slot_vertical_max_error:maxJointError,rigid_yoke_length_max_error:maxRigidError,physical_ruler_fft_max_error:maxReadoutError},null,2));
