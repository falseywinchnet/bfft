import {readFile,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {CompactMeasurement} from './compact.mjs';
import {rectangleReference,encode} from './rectangle-plan.mjs';
const reference=await WebAssembly.compile(await readFile(new URL('./reference.wasm',import.meta.url)));
const compact=await WebAssembly.compile(await readFile(new URL('./compact.wasm',import.meta.url)));
let seed=919731;const rand=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
const report={cases:0,checkedControls:0,maxStorageError:0,maxFusedError:0,maxByteError:0,differentBytes:0,maxConstantError:0,passed:false};
for(let trial=0;trial<120;trial++){
  const w=5+Math.floor(rand()*24),h=5+Math.floor(rand()*24),ow=1+Math.floor(rand()*w),oh=1+Math.floor(rand()*h),kind=trial%8;
  const rgba=new Uint8ClampedArray(w*h*4);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    let rgb;
    if(kind===0)rgb=[71,139,223];
    if(kind===1)rgb=[255*rand(),255*rand(),255*rand()];
    if(kind===2)rgb=[x<w/2?0:255,y<h/2?0:255,(x+y)%2*255];
    if(kind===3)rgb=[x*255/(w-1),y*255/(h-1),(x+y)*255/(w+h-2)];
    if(kind===4)rgb=[127+110*Math.sin(x*.19+y*.11),127+110*Math.cos(x*.13-y*.17),127+100*Math.sin(x*.43)*Math.cos(y*.41)];
    if(kind===5)rgb=[100+Math.floor(rand()*3),200-Math.floor(rand()*3),30+Math.floor(rand()*3)];
    if(kind===6)rgb=[x===2||x===w-3?255:0,y===3?255:0,x===Math.floor(w/2)&&y===Math.floor(h/2)?255:0];
    if(kind===7)rgb=[255*rand(),0,255*rand()];
    const alpha=kind===7?[0,1,2,80,255][Math.floor(rand()*5)]:trial%3===0?Math.floor(rand()*256):255;
    rgba.set([...rgb,alpha],(y*w+x)*4);
  }
  const a=(await WebAssembly.instantiate(reference)).exports;assert.equal(a.prepare_native(w,h),0);
  const source=new Float32Array(a.memory.buffer,a.native_source_pointer(),w*h*4);
  for(let k=0;k<rgba.length;k+=4){const alpha=rgba[k+3]/255;for(let c=0;c<3;c++)source[k+c]=rgba[k+c]/255*alpha;source[k+3]=alpha;}
  let code;do{code=a.build_native_step(64);assert(code>=0);}while(!code);
  const lw=5*(w-1)+1,controls=new Float32Array(a.memory.buffer,a.native_control_pointer(),lw*(5*(h-1)+1)*4);
  const expected=rectangleReference({w,h,lw,controls},ow,oh),eb=encode(expected);
  for(const exactStorage of [true,false]){
    const engine=new CompactMeasurement(await WebAssembly.instantiate(compact),w,h,rgba,ow,oh,{exactStorage}),b=engine.api;
    const seen=new Set();
    while(b.compact_phase()!==9){
      const channel=b.compact_channel();
      if(exactStorage&&b.compact_phase()===8&&!seen.has(channel)){
        seen.add(channel);
        for(let y=0;y<h-1;y++)for(let x=0;x<w-1;x++){
          const ptr=b.compact_probe_patch(x,y);assert(ptr);
          const patch=new Float32Array(b.memory.buffer,ptr,36);
          for(let j=0;j<6;j++)for(let i=0;i<6;i++){
            assert.equal(patch[j*6+i],controls[((5*y+j)*lw+5*x+i)*4+channel],`control ${trial} ${x} ${y} ${i} ${j} ${channel}`);report.checkedControls++;
          }
        }
      }
      engine.step();
    }
    const result=engine.finish(),ab=encode(result.values);let maxError=0;
    for(let k=0;k<expected.length;k++){
      maxError=Math.max(maxError,Math.abs(expected[k]-result.values[k]));const d=Math.abs(ab[k]-eb[k]);report.maxByteError=Math.max(report.maxByteError,d);report.differentBytes+=+(d>0);
    }
    assert(maxError<(exactStorage?2e-12:1e-7),`${trial} ${exactStorage} ${maxError}`);
    if(exactStorage)report.maxStorageError=Math.max(report.maxStorageError,maxError);else report.maxFusedError=Math.max(report.maxFusedError,maxError);
  }
  report.cases++;
}
assert(report.maxByteError<=1);report.passed=true;console.log(JSON.stringify(report,null,2));
await writeFile(new URL('./validation-results.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
