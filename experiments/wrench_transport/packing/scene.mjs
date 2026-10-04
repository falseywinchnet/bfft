import {writeFileSync} from 'node:fs';
import {cook} from '../hull.mjs';
import {boxPoints,cylinderPoints,spherePoints,makeRandom,randomRange} from '../shapes.mjs';
import {v3,normalized,quatFromAxisAngle} from '../math3.mjs';
const args=Object.fromEntries(process.argv.slice(2).reduce((a,x,i,v)=>i%2?a:[...a,[x.replace('--',''),v[i+1]]],[]));
const count=Number(args.count??32),seed=Number(args.seed??1),out=args.out;
const sectors=12,density=1000;
function ring(r0,r1,z0,z1,thickness) {
 const parts=[];
 for(let i=0;i<sectors;i++) {
  const p=[];
  for(const [r,z] of [[r0,z0],[r1,z1]]) for(const radius of [r-thickness,r]) for(const angle of [2*Math.PI*i/sectors,2*Math.PI*(i+1)/sectors]) p.push(v3(radius*Math.cos(angle),radius*Math.sin(angle),z));
  parts.push(p);
 }
 return parts;
}
function disc(r,h,z) {return cylinderPoints(r,h,sectors).map(p=>v3(p.x,p.y,p.z+z));}
function shape(name,hulls) {
 const s=cook({hulls,density,friction:.5,restitution:0,rollingResistance:0});
 return {name,mass:s.mass,inertia:s.inertia,radius:s.radius,comOffset:s.comOffset,hulls:s.hulls.map(h=>({vertices:h.vertices.map(p=>[p.x,p.y,p.z]),faces:h.faces.map(f=>f.loop),normals:h.faces.map(f=>[f.normal.x,f.normal.y,f.normal.z]),edges:h.edges.map(e=>[e.v0,e.v1])}))};
}
const shapes=[
 shape('open cup',[disc(.038,.006,-.037),...ring(.038,.043,-.034,.040,.006)]),
 shape('open bowl',[disc(.034,.006,-.015),...ring(.034,.065,-.012,.025,.006)]),
 null,
 shape('thin plate',[disc(.063,.005,0)]),
 shape('rod',[cylinderPoints(.011,.110,8)]),
 shape('small convex piece',[spherePoints(.021,16)])
];
// Bottle shoulder is a solid frustum, not a radial shell.
const shoulder=[];for(const [r,z] of [[.029,.028],[.014,.046]])for(let i=0;i<sectors;i++)shoulder.push(v3(r*Math.cos(2*Math.PI*i/sectors),r*Math.sin(2*Math.PI*i/sectors),z));
shapes[2]=shape('bottle',[disc(.029,.078,-.011),shoulder,disc(.014,.028,.060)]);
const demo=args.mode==='demo';
const scale=Math.cbrt(count/32),inner=(demo?.24:.42)*scale,height=(demo?.60:.68)*scale,wall=.025,half=(inner+wall)/2,span=inner+2*wall;
const walls=[{p:[half,0,height/2],size:[wall,span,height]},{p:[-half,0,height/2],size:[wall,span,height]},{p:[0,half,height/2],size:[span,wall,height]},{p:[0,-half,height/2],size:[span,wall,height]}];
const cols=Math.max(1,Math.floor(inner/.15)),slots=cols*cols,batches=Math.ceil(count/slots),interval=demo?.20:.40;
const random=makeRandom(seed*977+5),bodies=[];
for(let i=0;i<count;i++) {
 const batch=Math.floor(i/slots),slot=i%slots,axis=normalized(v3(randomRange(random,-1,1),randomRange(random,-1,1),randomRange(random,-1,1))),q=quatFromAxisAngle(axis,randomRange(random,0,2*Math.PI));
 bodies.push({shape:i%shapes.length,release:batch*interval,p:[.15*(slot%cols-(cols-1)/2),.15*(Math.floor(slot/cols)-(cols-1)/2),height+.16],q:[q.w,q.x,q.y,q.z],v:[0,0,0],w:[0,0,0]});
}
const scene={schema:'compound-packing-v1',mode:demo?'demo':'scaling',seed,count,inner,height,density,friction:.5,gravity:[0,0,-9.81],releaseEnd:(batches-1)*interval,seconds:(batches-1)*interval+8,sampleRate:60,geometryRate:10,shapes,walls,fixedBodies:[],bodies};
writeFileSync(out,JSON.stringify(scene));
const lines=['PACK1',`${density} .5 0 0 -9.81 ${scene.seconds} 60 10 ${shapes.length}`];
for(const s of shapes){lines.push(String(s.hulls.length));for(const h of s.hulls){lines.push(String(h.vertices.length));lines.push(...h.vertices.map(v=>v.join(' ')));}}
lines.push(String(walls.length));lines.push(...walls.map(w=>[...w.p,...w.size].join(' ')));lines.push('0');lines.push(String(count));lines.push(...bodies.map(b=>[b.shape,b.release,...b.p,...b.q,...b.v,...b.w].join(' ')));
writeFileSync(out.replace(/\.json$/,'.txt'),lines.join('\n')+'\n');
console.log(JSON.stringify({count,parts:shapes.reduce((n,s)=>n+s.hulls.length,0),seconds:scene.seconds,inner,height}));
