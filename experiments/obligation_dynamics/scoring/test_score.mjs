import assert from 'node:assert/strict';
import {transform,depth} from './score.mjs';
const vertices=[];for(let i=0;i<8;i++)vertices.push([(i&1)?1:-1,(i&2)?1:-1,(i&4)?1:-1]);const edges=[];for(let i=0;i<8;i++)for(let k=0;k<3;k++)if(!(i&(1<<k)))edges.push([i,i|(1<<k)]);const part={vertices,edges,normals:[[1,0,0],[0,1,0],[0,0,1]]},a=transform(part,[0,0,0,1,0,0,0]);
for(const x of [0,.2,1,1.999,2,3])assert.ok(Math.abs(depth(a,transform(part,[x,0,0,1,0,0,0]))-Math.max(0,2-x))<1e-12);
const small={...part,vertices:vertices.map(v=>v.map(x=>x*.1))};assert.ok(Math.abs(depth(a,transform(small,[0,0,0,1,0,0,0]))-1.1)<1e-12);
assert.equal(depth(a,transform(part,[3,3,3,Math.cos(.2),0,0,Math.sin(.2)])),0);
console.log('SAT independent known-depth, separated, contained and rotated checks passed');
