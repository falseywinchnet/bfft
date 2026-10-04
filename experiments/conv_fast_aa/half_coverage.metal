// Only the single-edge CDF uses half. Coordinates and clipping remain float.
float edgeCoverageHalf(float distance,float2 normal){
    half d=half(distance),a=half(max(abs(normal.x),abs(normal.y))),b=half(min(abs(normal.x),abs(normal.y)));
    if(b==half(0))return float(saturate(half(.5)+d/a));
    half z=abs(d),tail;
    if(z<=(a-b)*half(.5))tail=half(.5)-z/a;
    else{half h=max((a+b)*half(.5)-z,half(0));tail=h*h/(half(2)*a*b);}
    return float(d>=half(0)?half(1)-tail:tail);
}
