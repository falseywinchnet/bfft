const assert=require('node:assert/strict');const M=require('./mechanics.js');
const model=M.build();
assert.equal(model.bars.length,44);assert.deepEqual([0,1,2].map(s=>model.bars.filter(b=>b.stage===s).length),[16,24,4]);
assert.equal(model.bars.filter(b=>Math.abs(b.beta-.5)<1e-12).length,36);
assert.equal(model.bars.filter(b=>Math.abs(b.beta-(Math.SQRT2-1))<1e-12).length,8);
assert.equal(model.ops.filter(o=>o.type==='copy').length,12);
let seed=83752;function rand(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;}
let error=0,lengthError=0,tapError=0,energyError=0,rowError=0,tests=0;
for(const stage of model.physical)for(const row of stage){rowError=Math.max(rowError,Math.abs(row.reduce((a,b)=>a+b,0)-1));assert(row.every(v=>v>=0));}
function trial(x){const r=M.evaluate(model,x),ref=M.directDFT(x);for(let k=0;k<5;k++)error=Math.max(error,Math.hypot(r.spectrum[k].re-ref[k].re,r.spectrum[k].im-ref[k].im));
 const inputEnergy=x.reduce((s,v)=>s+v*v,0),outputEnergy=r.packed.reduce((s,v)=>s+v*v,0);energyError=Math.max(energyError,Math.abs(inputEnergy-outputEnergy));
 for(const b of model.bars){const g=r.geometries[b.id];lengthError=Math.max(lengthError,Math.abs(Math.hypot(g.B.x-g.A.x,g.B.y-g.A.y)-1));tapError=Math.max(tapError,Math.abs(Math.hypot(g.tap.x-g.A.x,g.tap.y-g.A.y)-b.beta));assert(r.positions[b.out]>=-1e-14&&r.positions[b.out]<=1+1e-14);}
 // Independently execute the original signed Bruun stages and compare each
 // physical rail difference after its fixed kinematic calibration.
 let signed=x;for(let s=0;s<3;s++){signed=M.mv(model.stages[s],signed);const ids=model.stageNodes[s],d=model.scales[s+1];for(let j=0;j<8;j++)assert(Math.abs((r.positions[ids[j]]-r.positions[ids[j+8]])/d[j]-signed[j])<2e-14);}
 tests++;
}
for(let mask=0;mask<256;mask++)trial(Array.from({length:8},(_,i)=>(mask>>i&1)?1:-1));
for(let j=0;j<8;j++)trial(Array.from({length:8},(_,i)=>i===j?1:0));
for(let j=0;j<1000;j++)trial(Array.from({length:8},()=>2*rand()-1));
trial(Array(8).fill(0));trial(Array(8).fill(1e-10));
assert(error<2e-14);assert(lengthError<1e-14);assert(tapError<1e-14);assert(energyError<2e-14);assert(rowError<1e-14);
assert.throws(()=>M.evaluate(model,[NaN,...Array(7).fill(0)]));assert.throws(()=>M.evaluate(model,Array(8).fill(2)));assert.throws(()=>M.barGeometry(0,2,.5));
console.log(JSON.stringify({tests,bars:model.bars.length,fft_max_absolute_error:error,rigid_length_max_error:lengthError,tap_fraction_max_error:tapError,parseval_max_error:energyError,row_stochastic_max_error:rowError},null,2));
