import {readFile,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {CompactMeasurement} from './compact.mjs';
import {rectangleReference,encode} from './rectangle-plan.mjs';
const reference=await WebAssembly.compile(await readFile(new URL('./reference.wasm',import.meta.url)));
const compact=await WebAssembly.compile(await readFile(new URL('./compact.wasm',import.meta.url)));
import {pixels} from './synthetic.mjs';
const cases=process.argv.includes('--large')?[[257,171,193,129,'waves'],[513,321,385,241,'waves']]:[[5,5,1,1,'constant'],[7,6,3,2,'edge'],[13,11,9,7,'noise'],[13,11,9,7,'alpha'],[33,29,5,7,'waves'],[17,13,17,13,'edge'],[31,27,1,5,'alpha']];
const results=[];
for(const [w,h,ow,oh,kind] of cases){
  const rgba=pixels(w,h,kind),a=(await WebAssembly.instantiate(reference)).exports;
  const rt=performance.now();assert.equal(a.prepare_native(w,h),0);
  const source=new Float32Array(a.memory.buffer,a.native_source_pointer(),w*h*4);
  for(let k=0;k<rgba.length;k+=4){const alpha=rgba[k+3]/255;for(let c=0;c<3;c++)source[k+c]=rgba[k+c]/255*alpha;source[k+3]=alpha;}
  let code;do{code=a.build_native_step(32);assert(code>=0);}while(!code);
  const referencePreparationMs=performance.now()-rt;
  const controls=new Float32Array(a.memory.buffer,a.native_control_pointer(),(5*(w-1)+1)*(5*(h-1)+1)*4);
  const expected=rectangleReference({w,h,lw:5*(w-1)+1,controls},ow,oh);
  const s=performance.now(),engine=new CompactMeasurement(await WebAssembly.instantiate(compact),w,h,rgba,ow,oh,{exactStorage:process.argv.includes('--exact')}),actual=engine.run(),compactMs=performance.now()-s;
  let maxError=0,maxByteError=0,differentBytes=0;const ab=encode(actual.values),eb=encode(expected);
  for(let k=0;k<expected.length;k++){maxError=Math.max(maxError,Math.abs(expected[k]-actual.values[k]));const d=Math.abs(ab[k]-eb[k]);maxByteError=Math.max(maxByteError,d);differentBytes+=+(d>0);}
  const row={w,h,ow,oh,kind,maxError,maxByteError,differentBytes,referencePreparationMs,compactMs,referenceMemoryBytes:a.memory.buffer.byteLength,...actual,values:undefined};
  results.push(row);console.log(JSON.stringify(row));assert(maxError<(process.argv.includes('--exact')?2e-12:1e-7),`Float error ${maxError}`);assert(maxByteError<=1);
}
if(process.argv.includes('--write'))await writeFile(new URL('./results.json',import.meta.url),JSON.stringify({passed:true,results},null,2)+'\n');
