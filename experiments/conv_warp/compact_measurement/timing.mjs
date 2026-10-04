import {readFile,writeFile} from 'node:fs/promises';
import {CompactMeasurement} from './compact.mjs';
import {pixels} from './synthetic.mjs';
const reference=await WebAssembly.compile(await readFile(new URL('./reference.wasm',import.meta.url)));
const compact=await WebAssembly.compile(await readFile(new URL('./compact.wasm',import.meta.url)));
const median=a=>a.slice().sort((x,y)=>x-y)[Math.floor(a.length/2)];
const results=[];
for(const [w,h,ow,oh,kind,repeats] of [[257,171,193,129,'waves',7],[513,321,385,241,'waves',7],[513,321,129,81,'noise',7],[1025,641,257,161,'waves',3]]){
  const rgba=pixels(w,h,kind),a=(await WebAssembly.instantiate(reference)).exports,b=await WebAssembly.instantiate(compact);
  function runReference(){
    const s=performance.now();if(a.prepare_native(w,h))throw Error('Reference allocation failed');
    const source=new Float32Array(a.memory.buffer,a.native_source_pointer(),w*h*4);
    for(let k=0;k<rgba.length;k+=4){const alpha=rgba[k+3]/255;for(let c=0;c<3;c++)source[k+c]=rgba[k+c]/255*alpha;source[k+3]=alpha;}
    let done;do{done=a.build_native_step(32);if(done<0)throw Error('Reference failed');}while(!done);
    return {milliseconds:performance.now()-s,memory:a.memory.buffer.byteLength};
  }
  function runCompact(){const s=performance.now(),engine=new CompactMeasurement(b,w,h,rgba,ow,oh),r=engine.run();return {milliseconds:performance.now()-s,memory:r.wasmMemoryBytes,arena:r.peakArenaBytes,profile:r.profile};}
  runReference();runCompact();const original=[],streamed=[];
  for(let i=0;i<repeats;i++){
    if(i%2){streamed.push(runCompact());original.push(runReference());}else{original.push(runReference());streamed.push(runCompact());}
  }
  const row={source:[w,h],target:[ow,oh],kind,repeats,referencePreparationMilliseconds:median(original.map(r=>r.milliseconds)),compactIncludingMeasurementMilliseconds:median(streamed.map(r=>r.milliseconds)),referenceWasmMemoryBytes:original[0].memory,compactWasmMemoryBytes:streamed[0].memory,compactArenaBytes:streamed[0].arena,referenceRuns:original.map(r=>r.milliseconds),compactRuns:streamed.map(r=>r.milliseconds),phaseMedianMilliseconds:Object.fromEntries(Object.keys(streamed[0].profile).map(k=>[k,median(streamed.map(r=>r.profile[k]||0))]))};
  results.push(row);console.log(JSON.stringify(row));
}
await writeFile(new URL('./timing-results.json',import.meta.url),JSON.stringify({runtime:process.version,platform:process.platform,architecture:process.arch,results},null,2)+'\n');
