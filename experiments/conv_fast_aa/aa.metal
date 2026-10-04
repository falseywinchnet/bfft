#include <metal_stdlib>
using namespace metal;
constexpr sampler linearSampler(coord::normalized,address::clamp_to_edge,filter::linear);
#define FXAA_PC 1
#define FXAA_QUALITY__PRESET 12
#define FXAA_GREEN_AS_LUMA 0
#define FXAA_GATHER4_ALPHA 0
#define FxaaBool bool
#define FxaaFloat float
#define FxaaFloat2 float2
#define FxaaFloat3 float3
#define FxaaFloat4 float4
#define FxaaInt2 int2
#define FxaaSat(x) saturate(x)
#define FxaaTex texture2d<float,access::sample>
#define FxaaTexTop(t,p) t.sample(linearSampler,p,level(0.0f))
#define FxaaTexOff(t,p,o,r) t.sample(linearSampler,p+float2(o)*r,level(0.0f))
#include "vendor/Fxaa3_11.h"

float4 readClamp(texture2d<float,access::sample> src,int2 p) {
    return src.read(uint2(clamp(p,int2(0),int2(src.get_width()-1,src.get_height()-1))));
}
kernel void copyImage(texture2d<float,access::sample> src [[texture(0)]],
                      texture2d<float,access::write> dst [[texture(1)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=dst.get_width() || p.y>=dst.get_height()) return;
    dst.write(src.read(p),p);
}
kernel void fxaa12(texture2d<float,access::sample> src [[texture(0)]],
                    texture2d<float,access::write> dst [[texture(1)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=dst.get_width() || p.y>=dst.get_height()) return;
    float2 r=1.0f/float2(src.get_width(),src.get_height());
    float4 v=FxaaPixelShader((float2(p)+.5f)*r,float4(0),src,src,src,src,r,
        float4(0),float4(0),float4(0),.75f,.166f,.0833f,8.0f,.125f,.05f,float4(0));
    dst.write(v,p);
}

// The only data-dependent geometry is the existing CONV first-jet stencil.
float4 tangent(texture2d<float,access::sample> src,int2 p,float m1,float m2) {
    float4 c=readClamp(src,p),e=readClamp(src,p+int2(1,0)),w=readClamp(src,p-int2(1,0));
    float4 n=readClamp(src,p-int2(0,1)),s=readClamp(src,p+int2(0,1));
    float gx=(8*(e.w-w.w)-(readClamp(src,p+int2(2,0)).w-readClamp(src,p-int2(2,0)).w))/12;
    float gy=(8*(s.w-n.w)-(readClamp(src,p+int2(0,2)).w-readClamp(src,p-int2(0,2)).w))/12;
    float den=max(abs(gx),abs(gy));
    float variation=max(max(abs(e.w-c.w),abs(w.w-c.w)),max(abs(n.w-c.w),abs(s.w-c.w)));
    if(den<=0x1p-20f*variation) return c;
    float ax=abs(gy)/den,ay=abs(gx)/den;
    float wd=ax*ay*m2*.5f,wx=ax*m1*.5f-wd,wy=ay*m1*.5f-wd;
    int sy=(-gx*gy>=0) ? 1 : -1;
    float4 diagonal=readClamp(src,p+int2(1,sy))+readClamp(src,p-int2(1,sy));
    return c+wx*(e+w-2*c)+wy*(n+s-2*c)+wd*(diagonal-2*c);
}
kernel void tangentTensor(texture2d<float,access::sample> src [[texture(0)]],
                    texture2d<float,access::write> dst [[texture(1)]],uint2 pos [[thread_position_in_grid]]) {
    if(pos.x>=dst.get_width() || pos.y>=dst.get_height()) return;
    int2 p=int2(pos);
    float4 c=readClamp(src,p),e=readClamp(src,p+int2(1,0)),w=readClamp(src,p-int2(1,0));
    float4 n=readClamp(src,p-int2(0,1)),s=readClamp(src,p+int2(0,1));
    float3 gx=(8*(e.xyz-w.xyz)-(readClamp(src,p+int2(2,0)).xyz-readClamp(src,p-int2(2,0)).xyz))/12;
    float3 gy=(8*(s.xyz-n.xyz)-(readClamp(src,p+int2(0,2)).xyz-readClamp(src,p-int2(0,2)).xyz))/12;
    float xx=dot(gx,gx),xy=dot(gx,gy),yy=dot(gy,gy);
    float gap=length(float2(xx-yy,2*xy)),trace=xx+yy;
    float3 variation=max(max(abs(e.xyz-c.xyz),abs(w.xyz-c.xyz)),max(abs(n.xyz-c.xyz),abs(s.xyz-c.xyz)));
    float tolerance=0x1p-20f*max(variation.x,max(variation.y,variation.z));
    if(gap==0 || trace<=tolerance*tolerance) {dst.write(c,pos);return;}
    float co=clamp((xx-yy)/gap,-1.0f,1.0f);
    float ax=sqrt((1-co)*.5f),ay=sqrt((1+co)*.5f),den=max(ax,ay);
    ax/=den;ay/=den;
    float coherence=gap/trace;
    float wd=ax*ay/6*coherence,wx=ax*.25f*coherence-wd,wy=ay*.25f*coherence-wd;
    int sy=xy<=0 ? 1 : -1;
    float4 diagonal=readClamp(src,p+int2(1,sy))+readClamp(src,p-int2(1,sy));
    dst.write(c+wx*(e+w-2*c)+wy*(n+s-2*c)+wd*(diagonal-2*c),pos);
}
kernel void tangentBox(texture2d<float,access::sample> src [[texture(0)]],
                    texture2d<float,access::write> dst [[texture(1)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=dst.get_width() || p.y>=dst.get_height()) return;
    dst.write(tangent(src,int2(p),.5f,1.0f/3),p);
}
kernel void tangentQuintic(texture2d<float,access::sample> src [[texture(0)]],
                    texture2d<float,access::write> dst [[texture(1)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=dst.get_width() || p.y>=dst.get_height()) return;
    dst.write(tangent(src,int2(p),5.0f/16,1.0f/7),p);
}
