import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {recoverTwoPixelLine} from './coverage_profile.mjs';
let worstMean=0,worstMass=0;
for(const a of [.1,.25,.415,.5,.75,.9])for(const b of [.1,.25,.585,.5,.75,.9]){
 const cap=Math.min(a+b,2*a,2*b,2*(1-a),2*(1-b));
 for(const fraction of [.1,.5,1]){
  const p=recoverTwoPixelLine(32,a,b,cap*fraction);
  worstMean=Math.max(worstMean,Math.abs(p.integral(31.5,32.5)-a),Math.abs(p.integral(32.5,33.5)-b));
  worstMass=Math.max(worstMass,Math.abs(p.integral(20,45)-a-b));
  let last=0,lastSign=0,changes=0;
  for(let k=0;k<=2000;k++){const v=p.value(31+k/500);assert.ok(v>=-1e-14&&v<=1+1e-14);const sign=Math.abs(v-last)>1e-12?Math.sign(v-last):0;if(sign){if(lastSign&&sign!==lastSign)changes++;lastSign=sign;}last=v;}
  assert.equal(changes,1);
 }
}
assert.ok(worstMean<2e-14&&worstMass<2e-14);
const p=recoverTwoPixelLine(32,.415,.585,.25),r=JSON.parse(await readFile('output/support_geometry/conv_line_coverage/results.json'));
const c=r.cases.find(c=>c.width===1&&c.phase===.585);
const views=c.views.map(v=>{const delta=v.sourceIntervalsPerOutputPixel;return {delta,phase:v.outputPhase,current:v.areaPeak,geometric:v.exactPeak,coverageProfile:Math.max(...v.samples.map(s=>p.integral(s.x-delta/2,s.x+delta/2)/delta))};});
const result={scope:'Known-background, known-unit-contrast two-cell line. Edge width 0.25 is one explicit witness from the admissible family, not a universally inferred width. C2 quintic edges, exact source cell means, range [0,1], one rise and one fall; no optimization. Geometry is inferred from the two coverage values. Tests sweep 108 coverage/edge-width combinations.',worstMean,worstMass,profile:{L:p.L,R:p.R,edgeWidth:p.edgeWidth},views};
await writeFile('output/support_geometry/conv_line_coverage/profile.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
