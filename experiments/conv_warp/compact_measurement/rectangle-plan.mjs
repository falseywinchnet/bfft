// Exact rectangular functionals of the existing shared tensor-quintic atlas.
// Gauss-3 integrates each degree-five Bernstein basis exactly in real arithmetic.
const nodes=[-Math.sqrt(3/5),0,Math.sqrt(3/5)], weights=[5/9,8/9,5/9];
function basis(u){const v=1-u;return [v**5,5*u*v**4,10*u*u*v**3,10*u**3*v*v,5*u**4*v,u**5];}
export function axisPlan(sourceLength,targetLength){
  if(!Number.isInteger(sourceLength)||sourceLength<2||!Number.isInteger(targetLength)||targetLength<1)throw Error('Invalid axis dimensions');
  const cells=sourceLength-1,ranges=new Uint32Array(targetLength*2),indices=[],values=[];
  for(let pixel=0;pixel<targetLength;pixel++){
    const left=targetLength===1?0:pixel===0?0:(pixel-.5)*cells/(targetLength-1);
    const right=targetLength===1?cells:pixel===targetLength-1?cells:(pixel+.5)*cells/(targetLength-1);
    const row=new Map();
    for(let cell=Math.floor(left);cell<Math.min(cells,Math.ceil(right));cell++){
      const a=Math.max(left,cell)-cell,b=Math.min(right,cell+1)-cell;
      if(b<=a)continue;
      for(let q=0;q<3;q++){
        const u=(a+b)/2+(b-a)/2*nodes[q],factor=(b-a)/2*weights[q]/(right-left),v=basis(u);
        for(let j=0;j<6;j++){const index=cell*5+j;row.set(index,(row.get(index)||0)+factor*v[j]);}
      }
    }
    ranges[pixel*2]=indices.length;ranges[pixel*2+1]=row.size;
    for(const [index,value] of row){indices.push(index);values.push(value);}
  }
  return {ranges,indices:Uint32Array.from(indices),weights:Float32Array.from(values),exactWeights:Float64Array.from(values)};
}
export function rectangleReference(atlas,width,height){
  const xp=axisPlan(atlas.w,width),yp=axisPlan(atlas.h,height),lh=5*(atlas.h-1)+1;
  const temp=new Float64Array(lh*width*4),out=new Float64Array(width*height*4);
  for(let y=0;y<lh;y++)for(let x=0;x<width;x++){
    const begin=xp.ranges[x*2],end=begin+xp.ranges[x*2+1];
    for(let k=begin;k<end;k++)for(let ch=0;ch<4;ch++)temp[(y*width+x)*4+ch]+=xp.exactWeights[k]*atlas.controls[(y*atlas.lw+xp.indices[k])*4+ch];
  }
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const begin=yp.ranges[y*2],end=begin+yp.ranges[y*2+1];
    for(let k=begin;k<end;k++)for(let ch=0;ch<4;ch++)out[(y*width+x)*4+ch]+=yp.exactWeights[k]*temp[(yp.indices[k]*width+x)*4+ch];
  }
  return out;
}
export function encode(values){
  const out=new Uint8ClampedArray(values.length);
  for(let k=0;k<out.length;k+=4){const a=Math.max(0,Math.min(1,values[k+3]));for(let ch=0;ch<3;ch++)out[k+ch]=a>1e-10?255*values[k+ch]/a:0;out[k+3]=255*a;}
  return out;
}
