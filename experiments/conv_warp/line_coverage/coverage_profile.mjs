// A bounded C2 witness for two occupied pixels, with known unit contrast.
// This is a finite coverage reconstruction, not a replacement for the nodal atlas.
export function smooth(t){return t<=0?0:t>=1?1:t*t*t*(10+t*(-15+6*t));}
function primitive(t){return t<=0?0:t>=1?t-.5:t**4*(2.5+t*(-3+t));}
export function recoverTwoPixelLine(leftIndex,leftCoverage,rightCoverage,edgeWidth){
 if(!(leftCoverage>0&&leftCoverage<1&&rightCoverage>0&&rightCoverage<1&&edgeWidth>0))throw Error('Two fractional neighboring cells and positive edge width required');
 const boundary=leftIndex+.5,L=boundary-leftCoverage,R=boundary+rightCoverage;
 const cap=Math.min(leftCoverage+rightCoverage,2*(1-leftCoverage),2*leftCoverage,2*rightCoverage,2*(1-rightCoverage));
 if(edgeWidth>cap)throw Error('The smooth transitions must lie within their source cells and not overlap');
 const stepIntegral=(x,edge)=>edgeWidth*primitive((x-edge)/edgeWidth+.5);
 return {L,R,edgeWidth,value:x=>smooth((x-L)/edgeWidth+.5)-smooth((x-R)/edgeWidth+.5),integral:(a,b)=>stepIntegral(b,L)-stepIntegral(a,L)-stepIntegral(b,R)+stepIntegral(a,R)};
}
