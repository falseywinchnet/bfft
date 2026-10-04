export function pixels(w,h,kind){
  let seed=9173;const rgba=new Uint8ClampedArray(w*h*4);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    const rgb=kind==='constant'?[71,139,223]:kind==='edge'?[x<w/2?0:255,y<h/2?0:255,(x+y)%2*255]:kind==='noise'?[seed&255,seed>>>8&255,seed>>>16&255]:[127+110*Math.sin(x*.19+y*.11),127+110*Math.cos(x*.13-y*.17),127+100*Math.sin(x*.43)*Math.cos(y*.41)];
    rgba.set([...rgb,kind==='alpha'?[0,1,80,255][(x+y)%4]:255],(y*w+x)*4);
  }return rgba;
}
