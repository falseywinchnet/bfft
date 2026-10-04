// Diagnostic against the shipped warp kernel. No production operator changes.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const warpRoot=resolve(process.argv[2]||'../paymenottowork/web/instruments/conv-perspective');
const out=resolve(process.argv[3]||'output/support_geometry/conv_line_coverage/results.json');
const {NativeWarpWasm}=await import(pathToFileURL(resolve(warpRoot,'warp-wasm.mjs')));
const binary=await readFile(resolve(warpRoot,'warp-kernel.wasm')),module=await WebAssembly.compile(binary);
const n=65,h=9,lw=5*(n-1)+1;
const overlap=(a,b,c,d)=>Math.max(0,Math.min(b,d)-Math.max(a,c));
const bern=(u)=>{const v=1-u;return [v**5,5*u*v**4,10*u*u*v**3,10*u**3*v*v,5*u**4*v,u**5];};
const gx=[-.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526],gw=[.3478548451374538,.6521451548625461,.6521451548625461,.3478548451374538];
function value(P,x){if(x<0||x>n-1)return 1;const c=Math.min(n-2,Math.floor(x)),b=bern(x-c);let v=0;for(let i=0;i<6;i++)v+=b[i]*P[(20*lw+5*c+i)*4];return v;}
function integral(P,a,b,power=0){let sum=0;for(let c=Math.max(0,Math.floor(a));c<=Math.min(n-2,Math.floor(b));c++){const l=Math.max(a,c),r=Math.min(b,c+1);if(r<=l)continue;for(let k=0;k<4;k++){const x=(l+r)/2+(r-l)/2*gx[k];sum+=(r-l)/2*gw[k]*(1-value(P,x))*x**power;}}return sum;}
function moments(P){const mass=integral(P,0,n-1),centroid=integral(P,0,n-1,1)/mass,variance=integral(P,0,n-1,2)/mass-centroid**2;let peak=0,min=1;for(let k=0;k<=12288;k++){const d=1-value(P,26+k/1024);peak=Math.max(peak,d);min=Math.min(min,d);}return {mass,centroid,variance,peak,minimumDeficit:min};}
const cases=[];let maxNativeIntegralError=0,maxPartitionError=0,maxSourceRecoveryError=0;
for(const width of [.5,1,2])for(const phase of [0,.25,.5,.585]){
 const center=32+phase,L=center-width/2,R=center+width/2;
 const coverage=Array.from({length:n},(_,i)=>overlap(L,R,i-.5,i+.5));
 const engine=new NativeWarpWasm(new WebAssembly.Instance(module,{}),n,h,new Uint8ClampedArray(n*h*4).fill(255)),a=engine.api;
 const src=new Float32Array(a.memory.buffer,a.native_source_pointer(),n*h*4);
 for(let y=0;y<h;y++)for(let x=0;x<n;x++)for(let ch=0;ch<3;ch++)src[(y*n+x)*4+ch]=1-coverage[x];
 const stages={};let previous=-1;
 while(a.native_phase()!==12){
  const phase=a.native_phase();if(phase!==previous&&[6,7].includes(phase))stages[phase===6?'factorProposal':'rangeClipped']=new Float32Array(a.memory.buffer,a.native_control_pointer(),lw*(5*(h-1)+1)*4).slice();
  previous=phase;if(a.build_native_step(1)<0)throw Error('Native source build failed');
 }
 stages.admitted=new Float32Array(a.memory.buffer,a.native_control_pointer(),lw*(5*(h-1)+1)*4).slice();
 const metrics=Object.fromEntries(Object.entries(stages).map(([name,P])=>[name,moments(P)])),views=[];
 for(const delta of [.5,1,2,4])for(const offset of [0,.25,.5,.75]){
  let exactPeak=0,pointPeak=0,areaPeak=0,exactMass=0,areaMass=0,absError=0;
  const samples=[];
  for(let k=-9;k<=9;k++){
   const x=32+(k+offset)*delta,lo=x-delta/2,hi=x+delta/2,truth=overlap(L,R,lo,hi)/delta,p=1-value(stages.admitted,x),v=integral(stages.admitted,lo,hi)/delta;
   exactPeak=Math.max(exactPeak,truth);pointPeak=Math.max(pointPeak,p);areaPeak=Math.max(areaPeak,v);exactMass+=truth*delta;areaMass+=v*delta;absError+=Math.abs(v-truth)*delta;
   if(Math.abs(k)<=3)samples.push({x,truth,point:p,area:v});
  }
  maxPartitionError=Math.max(maxPartitionError,Math.abs(areaMass-metrics.admitted.mass));
  views.push({sourceIntervalsPerOutputPixel:delta,outputPhase:offset,exactPeak,pointPeak,areaPeak,exactMass,areaMass,integratedAbsoluteError:absError,samples});
 }
 // Verify the independent 1-D polynomial integration against native 2-D area.
 const t={forward:[1,0,0,0,1,0,0,0,1],backward:[1,0,0,0,1,0,0,0,1],quad:[[0,0],[1,0],[1,1],[0,1]]};engine.setTransform(t);
 for(const outw of [17,33,65,129]){const delta=(n-1)/(outw-1);for(let x=Math.floor(outw/2)-2;x<=Math.floor(outw/2)+2;x++){
  const v=new Float64Array(a.memory.buffer,a.integrate_pixel(x,4,outw,h,1e-6),10)[0];
  maxNativeIntegralError=Math.max(maxNativeIntegralError,Math.abs((1-v)-integral(stages.admitted,x*delta-delta/2,x*delta+delta/2)/delta));
 }}
 for(let x=0;x<n;x++)maxSourceRecoveryError=Math.max(maxSourceRecoveryError,Math.abs((1-value(stages.admitted,x))-coverage[x]));
 const occupied=coverage.map((v,i)=>({v,i})).filter(x=>x.v>0);let recovered;
 if(occupied.length===2){const [left,right]=occupied,boundary=left.i+.5;recovered={left:boundary-left.v,right:boundary+right.v};}
 const profile=width===1&&phase===.585?Array.from({length:641},(_,k)=>{const x=30+k/128;return {x,deficit:1-value(stages.admitted,x)};}):undefined;
 cases.push({width,phase,center,profile,sourcePeak:Math.max(...coverage),sourceMass:coverage.reduce((a,b)=>a+b,0),stages:metrics,views,recovered});
}
if(maxNativeIntegralError>2e-6||maxPartitionError>2e-6||maxSourceRecoveryError>1e-6)throw Error('Verification failed');
const result={runtime:process.version,architecture:process.arch,kernelSHA256:createHash('sha256').update(binary).digest('hex'),sourceHash:createHash('sha256').update(await readFile(resolve(warpRoot,'source-kernel.c'))).digest('hex'),scope:'Exact box-coverage input stored in Float32, constant along y. Source samples are pixel-area means of known unit-contrast vertical lines. Source coordinates at pixel centres; endpoints kept far from line. Current unchanged native CONV field, stages read before/after range and current admission. Profiles use independent polynomial-exact four-point Gauss quadrature, checked against native 2-D integration. Direct geometric reference integrates the known line, not the CONV interpolant. No production change.',checks:{maxNativeIntegralError,maxPartitionError,maxSourceRecoveryError},cases};
await mkdir(dirname(out),{recursive:true});await writeFile(out,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({out,checks:result.checks,cases:cases.map(c=>({width:c.width,phase:c.phase,sourcePeak:c.sourcePeak,stages:c.stages,unitView:c.views.find(v=>v.sourceIntervalsPerOutputPixel===1&&v.outputPhase===0)}))},null,2));
