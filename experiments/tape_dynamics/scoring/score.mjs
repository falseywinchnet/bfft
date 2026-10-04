import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
function rotate(p,q){const v=q.slice(1),t=cross(v,p).map(x=>2*x),u=cross(v,t);return p.map((x,i)=>x+q[0]*t[i]+u[i]);}
function bounds(vertices){const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const p of vertices)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k]);}return{lo,hi};}
const intersect=(a,b)=>a.lo.every((v,i)=>v<b.hi[i]&&a.hi[i]>b.lo[i]);
export function transform(part,pose){const vertices=part.vertices.map(v=>rotate(v,pose.slice(3)).map((x,i)=>x+pose[i]));return{vertices,normals:part.normals.map(n=>rotate(n,pose.slice(3))),edges:part.edges.map(e=>sub(vertices[e[1]],vertices[e[0]])),...bounds(vertices)};}
export function depth(a,b){
 if(!intersect(a,b))return 0;let best=Infinity;
 function axis(n){const norm=Math.hypot(...n);if(norm<1e-12)return true;let amin=Infinity,amax=-Infinity,bmin=Infinity,bmax=-Infinity;for(const p of a.vertices){const x=dot(p,n);amin=Math.min(amin,x);amax=Math.max(amax,x);}for(const p of b.vertices){const x=dot(p,n);bmin=Math.min(bmin,x);bmax=Math.max(bmax,x);}const overlap=Math.min(amax-bmin,bmax-amin)/norm;best=Math.min(best,overlap);return overlap>0;}
 for(const n of a.normals)if(!axis(n))return 0;for(const n of b.normals)if(!axis(n))return 0;for(const ea of a.edges)for(const eb of b.edges)if(!axis(cross(ea,eb)))return 0;return Math.max(0,best);
}
function wallPart(w){const vertices=[];for(let i=0;i<8;i++)vertices.push(w.p.map((p,k)=>p+((i&(1<<k))?.5:-.5)*w.size[k]));const edges=[];for(let i=0;i<8;i++)for(let k=0;k<3;k++)if(!(i&(1<<k)))edges.push([i,i|(1<<k)]);return transform({vertices,normals:[[1,0,0],[0,1,0],[0,0,1]],edges},[0,0,0,1,0,0,0]);}
export function score(scene,result){
 const wallHulls=[...scene.walls.map(wallPart),...(scene.fixedBodies??[]).flatMap(b=>scene.shapes[b.shape].hulls.map(h=>transform(h,[...b.p,...b.q])))],frames=[];let finite=true,worst=0,worstFloor=0,maxEscaped=0;
 for(const sample of result.samples){if(!sample.poses)continue;let maxDepth=0,floor=0,pairs=0,escaped=0;const bodies=sample.poses.map((p,i)=>{if(!p)return null;if(!p.every(Number.isFinite)){finite=false;return null;}const parts=scene.shapes[scene.bodies[i].shape].hulls.map(h=>transform(h,p));const bb=bounds(parts.flatMap(h=>h.vertices));floor=Math.max(floor,-bb.lo[2]);if(Math.abs(p[0])>scene.inner/2||Math.abs(p[1])>scene.inner/2||p[2]<0)escaped++;return{parts,...bb};});
  for(let i=0;i<bodies.length;i++){const a=bodies[i];if(!a)continue;for(const w of wallHulls)if(intersect(a,w))for(const h of a.parts)maxDepth=Math.max(maxDepth,depth(h,w));for(let j=i+1;j<bodies.length;j++){const b=bodies[j];if(!b||!intersect(a,b))continue;let pair=0;for(const x of a.parts)for(const y of b.parts)pair=Math.max(pair,depth(x,y));maxDepth=Math.max(maxDepth,pair);if(pair>1e-5)pairs++;}}
  maxDepth=Math.max(maxDepth,floor);worst=Math.max(worst,maxDepth);worstFloor=Math.max(worstFloor,floor);maxEscaped=Math.max(maxEscaped,escaped);frames.push({t:sample.t,overlapMm:maxDepth*1000,floorMm:floor*1000,pairs,escaped});
 }
 const end=result.samples.at(-1).t;let settledAt=null;for(let i=result.samples.length-1;i>=0;i--){const r=result.samples[i];if(r.t<scene.releaseEnd||r.maxSpeed>=.004)break;settledAt=r.t;}if(settledAt!==null&&end-settledAt<1-1e-8)settledAt=null;
 const final=frames.at(-1),late=result.samples.filter(r=>r.t>end-1),expected=scene.shapes.map(s=>s.mass),massErrors=result.masses.map((m,i)=>m===null?0:Math.abs(m/expected[i]-1));
 return {engine:result.engine,hz:result.hz,count:scene.count,seed:scene.seed,parts:scene.bodies.reduce((n,b)=>n+scene.shapes[b.shape].hulls.length,0),wallSeconds:result.wallSeconds,activeSeconds:result.activeSeconds,activeMeanMs:1000*result.activeSeconds/result.activeSteps,setupSeconds:result.setupSeconds,activationSeconds:result.activationSeconds,timing:result.timing,finite,maxMassRelativeError:Math.max(...massErrors),guards:result.guards??null,warnings:result.warnings??null,maxSampledOverlapMm:worst*1000,maxFloorMm:worstFloor*1000,maxEscaped,final,settledAt,lastSecondMaxSpeed:Math.max(...late.map(r=>r.maxSpeed)),frames};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){const [s,r,out]=process.argv.slice(2);const result=score(JSON.parse(readFileSync(s)),JSON.parse(readFileSync(r)));writeFileSync(out,JSON.stringify(result));console.log(JSON.stringify({...result,frames:undefined}));}
