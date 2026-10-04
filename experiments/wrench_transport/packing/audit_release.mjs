// Check whether insertion itself intersects already-present geometry.
import {readFileSync,writeFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {depth,transform} from './score.mjs';
const [sp,rp,out]=process.argv.slice(2),scene=JSON.parse(readFileSync(sp)),result=JSON.parse(gunzipSync(readFileSync(rp)));let maximum=0,events=[];
for(const time of [...new Set(scene.bodies.map(b=>b.release))]){
 const sample=time===0?{poses:scene.bodies.map(()=>null)}:result.samples.find(r=>r.poses&&Math.abs(r.t-time)<1e-8);if(!sample)throw Error('release time was not captured');
 const fresh=scene.bodies.map((b,i)=>Math.abs(b.release-time)<1e-8?i:-1).filter(i=>i>=0),poses=sample.poses.map((p,i)=>fresh.includes(i)?[...scene.bodies[i].p,...scene.bodies[i].q]:p);
 const hulls=poses.map((p,i)=>p?scene.shapes[scene.bodies[i].shape].hulls.map(h=>transform(h,p)):null);let max=0;
 for(const i of fresh)for(let j=0;j<hulls.length;j++)if(j!==i&&hulls[j])for(const a of hulls[i])for(const b of hulls[j])max=Math.max(max,depth(a,b));
 maximum=Math.max(maximum,max);if(max>1e-7)events.push({t:time,overlapMm:1000*max});
}
const audit={maximumInsertionOverlapMm:maximum*1000,events};if(out)writeFileSync(out,JSON.stringify(audit));console.log(JSON.stringify(audit));
