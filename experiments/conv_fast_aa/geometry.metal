// Retained-geometry diagnostic. Exact pixel measure, not image reconstruction.
// Triangle edge equations are supplied by the renderer. Primitives in the
// benchmark are disjoint; arbitrary overlap/visibility is not solved here.
struct Triangle {float2 a,b,c;};
struct VertexOut {float4 position [[position]]; uint instance [[flat]];
    float2 origin [[flat]];float2 e1 [[flat]];float2 e2 [[flat]];
    float3 radii [[flat]];float twiceArea [[flat]];};
vertex VertexOut triangleVertex(uint vertexIndex [[vertex_id]],uint instance [[instance_id]],
    const device Triangle* triangles [[buffer(0)]],constant float2& size [[buffer(1)]]) {
    Triangle t=triangles[instance];float2 p=vertexIndex==0?t.a:vertexIndex==1?t.b:t.c;
    VertexOut o;o.position=float4(p.x/size.x*2-1,1-p.y/size.y*2,0,1);o.instance=instance;
    o.origin=0;o.e1=0;o.e2=0;o.radii=0;o.twiceArea=0;return o;
}
vertex VertexOut boundsVertex(uint vertexIndex [[vertex_id]],uint instance [[instance_id]],
    const device Triangle* triangles [[buffer(0)]],constant float2& size [[buffer(1)]]) {
    Triangle t=triangles[instance];float2 lo=floor(min(t.a,min(t.b,t.c))-.5f),hi=ceil(max(t.a,max(t.b,t.c))+.5f);
    constexpr uint2 corners[6]={uint2(0,0),uint2(1,0),uint2(0,1),uint2(0,1),uint2(1,0),uint2(1,1)};
    float2 p=mix(lo,hi,float2(corners[vertexIndex]));
    VertexOut o;o.instance=instance;
    o.origin=t.a;o.e1=t.b-t.a;o.e2=t.c-t.a;float2 e3=o.e2-o.e1;
    o.radii=.5f*float3(abs(o.e1.x)+abs(o.e1.y),abs(e3.x)+abs(e3.y),abs(o.e2.x)+abs(o.e2.y));
    o.twiceArea=o.e1.x*o.e2.y-o.e1.y*o.e2.x;
    // A centroid dilation moves every supporting edge out by at least its
    // unit-pixel support radius. This is a conservative draw support only;
    // pixel integration still uses the original triangle. No source analysis.
    float scale=1+3*max(o.radii.x,max(o.radii.y,o.radii.z))/o.twiceArea;
    if(.5f*o.twiceArea*scale*scale < (hi.x-lo.x)*(hi.y-lo.y)) {
        float2 centre=(t.a+t.b+t.c)/3;
        float2 v=vertexIndex==0?t.a:vertexIndex==1?t.b:t.c;
        p=vertexIndex<3?centre+scale*(v-centre):centre;
    }
    o.position=float4(p.x/size.x*2-1,1-p.y/size.y*2,0,1);return o;
}
fragment float4 opaqueFragment(VertexOut in [[stage_in]]) {return float4(1);}
float cross2(float2 a,float2 b) {return a.x*b.y-a.y*b.x;}
bool clipPlane(float f,float slope,thread float& lo,thread float& hi) {
    if(slope>0)lo=max(lo,-f/slope);
    else if(slope<0)hi=min(hi,-f/slope);
    else if(f<0)return false;
    return hi>lo;
}
float segmentBox(float2 a,float2 b) {
    // The square owns coincident edges, preventing a doubled boundary term.
    if((a.x==b.x && abs(a.x)==.5f)||(a.y==b.y && abs(a.y)==.5f))return 0;
    float2 d=b-a;float lo=0,hi=1;
    if(!clipPlane(a.x+.5f,d.x,lo,hi)||!clipPlane(.5f-a.x,-d.x,lo,hi)||
       !clipPlane(a.y+.5f,d.y,lo,hi)||!clipPlane(.5f-a.y,-d.y,lo,hi))return 0;
    return cross2(a+lo*d,a+hi*d)*.5f;
}
float segmentTriangle(float2 a,float2 b,thread const float2* v) {
    float2 d=b-a;float lo=0,hi=1;
    for(int i=0;i<3;i++) {
        float2 edge=v[(i+1)%3]-v[i];
        if(!clipPlane(cross2(edge,a-v[i]),cross2(edge,d),lo,hi))return 0;
    }
    return cross2(a+lo*d,a+hi*d)*.5f;
}
float edgeCoverage(float d,float2 n) {
    float a=max(abs(n.x),abs(n.y)),b=min(abs(n.x),abs(n.y));
    if(b==0)return saturate(.5f+d/a);
    float z=abs(d),tail;
    if(z<=(a-b)*.5f)tail=.5f-z/a;
    else {float h=max((a+b)*.5f-z,0.0f);tail=h*h/(2*a*b);}
    return d>=0?1-tail:tail;
}
fragment float4 areaFragment(VertexOut in [[stage_in]]) {
    float2 q=in.position.xy-in.origin;
    float2 edges[3]={in.e1,in.e2-in.e1,-in.e2};
    float3 centres=float3(cross2(edges[0],q),cross2(edges[1],q)+in.twiceArea,cross2(edges[2],q));
    if(any(centres<=-in.radii))return float4(0);
    int active=0;float d=0;float2 normal=0;
    for(int i=0;i<3;i++) {
        if(centres[i]<in.radii[i]) {active++;d=centres[i];normal=float2(-edges[i].y,edges[i].x);}
    }
    float area=1;
    if(active==1)area=edgeCoverage(d,normal);
    if(active>1) {
        float2 v[3]={-q,in.e1-q,in.e2-q};
        area=0;
        for(int i=0;i<3;i++)area+=segmentBox(v[i],v[(i+1)%3]);
        float2 q[4]={float2(-.5f,-.5f),float2(.5f,-.5f),float2(.5f,.5f),float2(-.5f,.5f)};
        for(int i=0;i<4;i++)area+=segmentTriangle(q[i],q[(i+1)%4],v);
    }
    // Fixed color + coverage alpha; standard alpha blending into black.
    return float4(1,1,1,saturate(area));
}
