/* Front projection of a layered, connected rigid-yoke assembly. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.MechanicalAssembly=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const L=48,T=.55*L,INPUT_L=66,SLOT=12;
const col=r=>80+(2*(r%8)+(r>=8?1:0))*64;
function build(model){
 const owners=new Map(),aliases=new Map(),ports=[],bars=[];
 for(let i=0;i<8;i++){const cx=112+128*i;owners.set(i,{x:cx-INPUT_L/2,y:150});owners.set(i+8,{x:cx+INPUT_L/2,y:150});}
 const root=id=>{while(aliases.has(id))id=aliases.get(id);return id;};
 for(const op of model.ops){if(op.type==='copy'){aliases.set(op.out,root(op.a));continue;}
  const x=col(op.row),y=op.stage===0?410:op.stage===1?(op.part===1?650:760):1010;
  const b={...op,x,y};bars[op.id]=b;
  ports.push({node:root(op.a),x:x-L/2,y,bar:op.id,pin:'A'},{node:root(op.b),x:x+L/2,y,bar:op.id,pin:'B'});
  owners.set(op.out,{x:x+(op.beta-.5)*L,y});
 }
 const outputs=model.permutation.map((row,k)=>{const x=112+k*128,y=1240;ports.push({node:root(model.stageNodes[2][row]),x:x-17,y,output:k,rail:'positive'},{node:root(model.stageNodes[2][row+8]),x:x+17,y,output:k,rail:'negative'});return {x,y,row,k,positive:root(model.stageNodes[2][row]),negative:root(model.stageNodes[2][row+8])};});
 const yokes=[...owners].map(([node,source],layer)=>{const targets=ports.filter(p=>p.node===node);const bus=source.y+65+(layer%16)*3;
  const segments=targets.map(p=>[source,{x:source.x,y:bus},{x:p.x,y:bus},p]);
  return {node,source,targets,segments,layer:layer+1,z:-(layer+1)*4};});
 return {bars,outputs,yokes,root,L,T,INPUT_L,SLOT,width:1120,height:1370};
}
function rocker(x,i){const cx=112+128*i,dy=T*x,dx=Math.sqrt(INPUT_L**2-dy**2);return {A:{x:cx-dx/2,y:150-dy/2},B:{x:cx+dx/2,y:150+dy/2},pivot:{x:cx,y:150},positive:(1+x)/2,negative:(1-x)/2};}
function geometry(assembly,result,samples){
 const shift=id=>-T*(result.positions[id]-.5);
 const yokes=assembly.yokes.map(y=>({...y,dy:shift(y.node),source:{...y.source,y:y.source.y+shift(y.node)},targets:y.targets.map(p=>({...p,y:p.y+shift(y.node)})),segments:y.segments.map(s=>s.map(p=>({x:p.x,y:p.y+shift(y.node)})))}));
 const bars=assembly.bars.map(b=>{const g=result.geometries[b.id],p=v=>({x:b.x+v.x*L,y:b.y+T/2-v.y*L});return {...b,A:p(g.A),B:p(g.B),tap:p(g.tap),centre:{x:b.x,y:(p(g.A).y+p(g.B).y)/2}};});
 const outputs=assembly.outputs.map(o=>({...o,positiveY:o.y+shift(o.positive),negativeY:o.y+shift(o.negative)}));
 return {bars,yokes,outputs,inputs:samples.map(rocker)};
}
return {build,geometry,rocker,L,T,INPUT_L,SLOT};
});
