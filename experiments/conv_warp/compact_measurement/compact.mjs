import {axisPlan} from './rectangle-plan.mjs';
export function reversePlan(sourceLength,targetLength){
  const p=axisPlan(sourceLength,targetLength),length=5*(sourceLength-1)+1;
  const offsets=new Uint32Array(length+1),indices=new Uint32Array(p.indices.length),weights=new Float64Array(p.indices.length);
  for(const i of p.indices)offsets[i+1]++;
  for(let i=1;i<=length;i++)offsets[i]+=offsets[i-1];
  const cursor=offsets.slice();
  for(let target=0;target<targetLength;target++)for(let k=p.ranges[target*2];k<p.ranges[target*2]+p.ranges[target*2+1];k++){
    const j=cursor[p.indices[k]]++;indices[j]=target;weights[j]=p.exactWeights[k];
  }
  return {offsets,indices,weights};
}
export class CompactMeasurement {
  constructor(instance,w,h,rgba,width,height,{exactStorage=false}={}){
    this.api=instance.exports;this.width=width;this.height=height;this.profile={};
    const x=reversePlan(w,width),y=reversePlan(h,height),a=this.api;
    const code=a.prepare_compact(w,h,width,height,x.indices.length,y.indices.length,+exactStorage);
    if(code)throw Error(`Compact allocation failed: ${code}`);
    this.exactStorage=exactStorage;
    for(const [which,data] of [[2,x.offsets],[3,x.indices],[4,x.weights],[5,y.offsets],[6,y.indices],[7,y.weights]])new data.constructor(a.memory.buffer,a.compact_pointer(which),data.length).set(data);
    const source=new Float32Array(a.memory.buffer,a.compact_pointer(0),w*h*4);
    for(let k=0;k<rgba.length;k+=4){const alpha=rgba[k+3]/255;for(let c=0;c<3;c++)source[k+c]=rgba[k+c]/255*alpha;source[k+3]=alpha;}
  }
  step(budget=1){
    const a=this.api,phase=a.compact_phase(),start=performance.now(),code=a.compact_step(budget);
    this.profile[phase]=(this.profile[phase]||0)+performance.now()-start;
    if(code<0)throw Error(`Compact preparation failed: ${code}`);
    return code===1;
  }
  finish(){
    const a=this.api;
    if(a.compact_phase()!==9)throw Error('Compact measurement is not complete.');
    return {values:new Float64Array(a.memory.buffer,a.compact_pointer(1),this.width*this.height*4).slice(),peakArenaBytes:a.compact_pointer(8)>>>0,wasmMemoryBytes:a.memory.buffer.byteLength,diagnostic:{constraintRows:new Float64Array(a.memory.buffer,a.compact_pointer(9),5)[0],storedMargin:this.exactStorage?new Float64Array(a.memory.buffer,a.compact_pointer(9),5)[2]:null},profile:this.profile};
  }
  run(){while(!this.step()){}return this.finish();}
  async runAsync(progress=()=>{}){
    const channel=new MessageChannel();
    try{
      for(;;){const start=performance.now();let done;do{done=this.step();}while(!done&&performance.now()-start<16);progress(this.api.compact_phase(),this.api.compact_channel());if(done)break;await new Promise(resolve=>{channel.port1.onmessage=resolve;channel.port2.postMessage(null);});}
      return this.finish();
    }finally{channel.port1.close();channel.port2.close();}
  }
}
