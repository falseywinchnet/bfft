/* Exact finite-displacement, two-rail N=8 Bruun linkage model. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.MechanicalFFT=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const N=8, SQRT2=Math.SQRT2, EPS=1e-13;
const matrix=(r,c)=>Array.from({length:r},()=>Array(c).fill(0));
function identity(n){const m=matrix(n,n);for(let i=0;i<n;i++)m[i][i]=1;return m;}
function mul(a,b){return a.map(row=>b[0].map((_,j)=>row.reduce((v,x,k)=>v+x*b[k][j],0)));}
function mv(a,x){return a.map(row=>row.reduce((s,v,i)=>s+v*x[i],0));}
function block(a,b){const m=matrix(a.length+b.length,a.length+b.length);a.forEach((r,i)=>r.forEach((v,j)=>m[i][j]=v));b.forEach((r,i)=>r.forEach((v,j)=>m[i+a.length][j+a.length]=v));return m;}
function half(n){const h=n/2,m=matrix(n,n);for(let j=0;j<h;j++){m[j][j]=m[j][j+h]=m[j+h][j]=1/SQRT2;m[j+h][j+h]=-1/SQRT2;}return m;}
function factorization(){
 const c=1/SQRT2,s=c;
 const r=[[1,0,c,-s],[0,1,s,c],[1,0,-c,s],[0,-1,s,c]].map(row=>row.map(v=>v/SQRT2));
 const p=identity(4);const odd=mul(r,[p[0],p[2],p[1],p[3]]).map((row,i)=>row.map(v=>(i%2?-1:1)*v));
 const e=[[1/SQRT2,1/SQRT2,0,0],[0,0,1,0],[0,0,0,-1],[1/SQRT2,-1/SQRT2,0,0]];
 return [half(8),block(half(4),odd),block(e,identity(4))];
}
function lift(m){const n=m.length,out=matrix(2*n,2*n);for(let i=0;i<n;i++)for(let j=0;j<n;j++){const p=Math.max(m[i][j],0),v=Math.max(-m[i][j],0);out[i][j]=out[i+n][j+n]=p;out[i][j+n]=out[i+n][j]=v;}return out;}
function gauge(stages){const scales=[Array(16).fill(1)],physical=[];for(const stage of stages){const c=lift(stage),d=scales.at(-1),next=c.map(row=>1/row.reduce((s,v,i)=>s+v/d[i],0));physical.push(c.map((row,j)=>row.map((v,i)=>next[j]*v/d[i])));scales.push(next);}return {scales,physical};}
/* A rigid bar of unit length, centre constrained laterally, with its endpoint
   and tap pins in horizontal slots on vertical shuttles. q is travel fraction.
   The branch dx>0 is nonsingular because travel<length and q is in [0,1]. */
function barGeometry(a,b,beta,length=1,travel=.55){
 if(![a,b,beta,length,travel].every(Number.isFinite)||length<=0||travel<=0||beta<0||beta>1)throw Error('Invalid bar geometry');
 const ya=travel*a,yb=travel*b,dy=yb-ya;
 if(Math.abs(dy)>=length)throw Error('Bar travel exceeds its nonsingular range');
 const dx=Math.sqrt(length*length-dy*dy);
 const A={x:-dx/2,y:ya},B={x:dx/2,y:yb};
 const tap={x:A.x+beta*(B.x-A.x),y:A.y+beta*(B.y-A.y)};
 return {A,B,tap,q:tap.y/travel,angle:Math.atan2(dy,dx),length,travel,beta};
}
function build(){
 const stages=factorization(),{scales,physical}=gauge(stages),nodes=[],bars=[],ops=[],stageNodes=[];
 for(let i=0;i<16;i++)nodes.push({id:i,stage:-1,row:i,label:`x${i%8}${i<8?'+':'−'}`});
 let prev=nodes.map(v=>v.id);
 function node(stage,row,label){const id=nodes.length;nodes.push({id,stage,row,label});return id;}
 function combine(a,b,beta,stage,row,part){
  if(beta>.5+EPS){[a,b]=[b,a];beta=1-beta;}
  const out=node(stage,row,`s${stage+1}.${row%8}${row<8?'+':'−'}${part?'·tap':''}`);
  const bar={id:bars.length,a,b,beta,out,stage,row,part};bars.push(bar);ops.push({type:'bar',...bar});return out;
 }
 physical.forEach((m,stage)=>{
  const next=[];
  m.forEach((row,j)=>{
   const support=row.map((w,i)=>({w,id:prev[i]})).filter(v=>v.w>EPS);
   if(support.length===1){const out=node(stage,j,`s${stage+1}.${j%8}${j<8?'+':'−'}`);ops.push({type:'copy',a:support[0].id,out,stage,row:j});next.push(out);}
   else if(support.length===2)next.push(combine(support[0].id,support[1].id,support[1].w/(support[0].w+support[1].w),stage,j,0));
   else if(support.length===3){
    let best=null;
    for(const [a,b] of [[0,1],[0,2],[1,2]]){const k=[0,1,2].find(i=>i!==a&&i!==b),sum=support[a].w+support[b].w,beta=support[b].w/sum,last=support[k].w/(sum+support[k].w),score=Math.min(beta,1-beta,last,1-last);if(!best||score>best.score)best={a,b,k,beta,last,score};}
    const inner=combine(support[best.a].id,support[best.b].id,best.beta,stage,j,1);
    next.push(combine(inner,support[best.k].id,best.last,stage,j,2));
   }else throw Error('Unsupported row');
  });stageNodes.push(next);prev=next;
 });
 const permutation=[0,4,5,1,2,6,7,3];
 return {N,stages,physical,scales,nodes,bars,ops,stageNodes,permutation};
}
function evaluate(model,samples){
 if(samples.length!==8||samples.some(x=>!Number.isFinite(x)||Math.abs(x)>1+EPS))throw Error('Eight finite samples in [-1,1] required');
 const positions=Array(model.nodes.length).fill(0),geometries=[];
 const inputGeometries=samples.map(x=>{const g=barGeometry((1+x)/2,(1-x)/2,.5,1,.4);return {...g,pivot:{x:0,y:.2}};});
 inputGeometries.forEach((g,i)=>{positions[i]=g.A.y/g.travel;positions[i+8]=g.B.y/g.travel;});
 for(const op of model.ops){if(op.type==='copy')positions[op.out]=positions[op.a];else{const g=barGeometry(positions[op.a],positions[op.b],op.beta);positions[op.out]=g.q;geometries[op.id]=g;}}
 const last=model.stageNodes.at(-1),d=model.scales.at(-1);
 const packed=model.permutation.map(i=>(positions[last[i]]-positions[last[i+8]])/d[i]);
 const spectrum=[{re:Math.sqrt(8)*packed[0],im:0},...Array.from({length:3},(_,i)=>({re:2*packed[1+2*i],im:2*packed[2+2*i]})),{re:Math.sqrt(8)*packed[7],im:0}];
 return {positions,geometries,inputGeometries,packed,spectrum};
}
/* Independent oracle only; evaluate() does not call this or any Fourier sum. */
function directDFT(samples){return Array.from({length:5},(_,k)=>samples.reduce((z,x,j)=>{const p=-2*Math.PI*k*j/8;return {re:z.re+x*Math.cos(p),im:z.im+x*Math.sin(p)};},{re:0,im:0}));}
function ancestors(model,id){const nodes=new Set([id]),bars=new Set();for(let i=model.ops.length-1;i>=0;i--){const op=model.ops[i];if(nodes.has(op.out)){nodes.add(op.a);if(op.type==='bar'){nodes.add(op.b);bars.add(op.id);}}}return {nodes,bars};}
return {N,build,evaluate,barGeometry,directDFT,ancestors,factorization,lift,gauge,mul,mv};
});
