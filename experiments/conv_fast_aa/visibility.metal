#include <metal_stdlib>
using namespace metal;
struct Primitive {float4 ab,cz,paint,dx,dy;};
// At most 15 planes; even retaining coincident duplicate endpoints needs <=34.
struct Polygon {float2 p[40];int n;};
float crossv(float2 a,float2 b){return a.x*b.y-a.y*b.x;}
void clipv(thread Polygon& p,float3 plane){
    if(p.n==0)return;
    Polygon q;q.n=0;
    for(int i=0;i<p.n;i++){
        float2 a=p.p[i],b=p.p[(i+1)%p.n];float da=dot(plane.xy,a)+plane.z,db=dot(plane.xy,b)+plane.z;
        if(da>=0)q.p[q.n++]=a;
        if((da<0)!=(db<0))q.p[q.n++]=a+(b-a)*(da/(da-db));
    }
    p.n=q.n;for(int i=0;i<q.n;i++)p.p[i]=q.p[i];
}
void triangleClip(thread Polygon& p,Primitive t,float2 centre){
    float2 v[3]={t.ab.xy-centre,t.ab.zw-centre,t.cz.xy-centre};
    for(int i=0;i<3;i++) {float2 e=v[(i+1)%3]-v[i],n=float2(-e.y,e.x);clipv(p,float3(n,-dot(n,v[i])));}
}
float3 momentsv(thread Polygon& p){
    float3 m=0;if(p.n<3)return m;
    for(int i=0;i<p.n;i++){float2 a=p.p[i],b=p.p[(i+1)%p.n];float d=crossv(a,b);m+=float3(d,(a+b)*d);}
    return m*float3(.5f,1.0f/6,1.0f/6);
}
uint supportv(Primitive t,float2 centre){
    float2 v[3]={t.ab.xy-centre,t.ab.zw-centre,t.cz.xy-centre};bool full=true;
    for(int i=0;i<3;i++){float2 e=v[(i+1)%3]-v[i],n=float2(-e.y,e.x);float d=-dot(n,v[i]),r=.5f*(abs(n.x)+abs(n.y));if(d+r<=0)return 0;full=full&&(d-r>=0);}
    return full?2:1;
}
float4 evaluate_visibility(const device Primitive* primitives,const device uint* counts,uint4 info,uint2 pos){
    uint tile=pos.x/32+(pos.y/32)*info.z,count=counts[tile];
    if(count>4){return float4(1,0,1,1);} // Host rejects overflow.
    float2 centre=float2(pos%32)+.5f;float3 result=0;
    uint candidate=0,full=0;
    for(uint i=0;i<count;i++){uint s=supportv(primitives[tile*4+i],centre);if(s)candidate|=1u<<i;if(s==2)full|=1u<<i;}
    if(candidate==0){return float4(0,0,0,1);}
    // Exact whole-pixel certificate from the supplied affine geometry/depth.
    for(uint i=0;i<count;i++)if(full&(1u<<i)){
        Primitive a=primitives[tile*4+i];bool front=true;
        for(uint j=0;j<count;j++)if(j!=i&&(candidate&(1u<<j))){
            Primitive b=primitives[tile*4+j];float3 d=float3(a.cz.zw-b.cz.zw,a.paint.x-b.paint.x);
            float upper=dot(d.xy,centre)+d.z+.5f*(abs(d.x)+abs(d.y));
            if(upper>0||(all(d==0)&&j<i))front=false;
        }
        if(front){return float4(a.paint.yzw+a.dx.xyz*centre.x+a.dy.xyz*centre.y,1);}
    }
    for(uint i=0;i<count;i++){
        if(!(candidate&(1u<<i)))continue;
        Primitive receiver=primitives[tile*4+i];
        Polygon base;base.n=4;base.p[0]=float2(-.5f,-.5f);base.p[1]=float2(.5f,-.5f);base.p[2]=float2(.5f,.5f);base.p[3]=float2(-.5f,.5f);
        triangleClip(base,receiver,centre);if(base.n<3)continue;
        uint occluders=0;
        for(uint j=0;j<count;j++)if(j!=i&&(candidate&(1u<<j))){
            Primitive b=primitives[tile*4+j];float3 d=float3(receiver.cz.zw-b.cz.zw,receiver.paint.x-b.paint.x);
            float upper=dot(d.xy,centre)+d.z+.5f*(abs(d.x)+abs(d.y));
            if(full&(1u<<j)){
                // A whole-pixel occluder leaves one visible depth halfplane.
                // Eliminate it before inclusion-exclusion over partial supports.
                if(all(d==0)){if(j<i){base.n=0;break;}}
                else clipv(base,float3(-d.xy,-d.z-dot(d.xy,centre)));
                if(base.n<3)break;
            } else if(upper>0||(upper==0&&!(all(d==0)&&j>i)))occluders|=1u<<j;
        }
        if(base.n<3)continue;
        float3 measure=0;
        uint mask=occluders;
        while(true){
            Polygon p=base;int parity=0;
            for(uint j=0;j<count;j++)if(mask&(1u<<j)){
                Primitive other=primitives[tile*4+j];float3 depth=float3(receiver.cz.zw-other.cz.zw,receiver.paint.x-other.paint.x);
                if(all(depth==0)&&j>i){p.n=0;break;}
                triangleClip(p,other,centre);clipv(p,float3(depth.xy,depth.z+dot(depth.xy,centre)));parity++;
                if(p.n<3)break;
            }
            measure+=(parity%2?-1.0f:1.0f)*momentsv(p);
            if(mask==0)break;mask=(mask-1)&occluders;
        }
        float3 c=receiver.paint.yzw+receiver.dx.xyz*centre.x+receiver.dy.xyz*centre.y;
        result+=c*measure.x+receiver.dx.xyz*measure.y+receiver.dy.xyz*measure.z;
    }
    return float4(result,1);
}

kernel void visibility(const device Primitive* p [[buffer(0)]],const device uint* c [[buffer(1)]],constant uint4& info [[buffer(2)]],texture2d<float,access::write> output [[texture(0)]],uint2 pos [[thread_position_in_grid]]){
 if(pos.x<info.x&&pos.y<info.y)output.write(evaluate_visibility(p,c,info,pos),pos);
}
kernel void classify(const device Primitive* p [[buffer(0)]],const device uint* counts [[buffer(1)]],constant uint4& info [[buffer(2)]],device atomic_uint* counter [[buffer(3)]],device uint* jobs [[buffer(4)]],texture2d<float,access::write> output [[texture(0)]],uint2 pos [[thread_position_in_grid]]){
 if(pos.x>=info.x||pos.y>=info.y)return;
 uint tile=pos.x/32+(pos.y/32)*info.z,count=counts[tile],candidate=0,full=0;
 float2 centre=float2(pos%32)+.5f;
 for(uint i=0;i<count;i++){uint s=supportv(p[tile*4+i],centre);if(s)candidate|=1u<<i;if(s==2)full|=1u<<i;}
 if(!candidate){output.write(float4(0,0,0,1),pos);return;}
 for(uint i=0;i<count;i++)if(full&(1u<<i)){
  Primitive a=p[tile*4+i];bool front=true;
  for(uint j=0;j<count;j++)if(j!=i&&(candidate&(1u<<j))){Primitive b=p[tile*4+j];float3 d=float3(a.cz.zw-b.cz.zw,a.paint.x-b.paint.x);float upper=dot(d.xy,centre)+d.z+.5f*(abs(d.x)+abs(d.y));if(upper>0||(all(d==0)&&j<i))front=false;}
  if(front){output.write(float4(a.paint.yzw+a.dx.xyz*centre.x+a.dy.xyz*centre.y,1),pos);return;}
 }
 uint slot=atomic_fetch_add_explicit(counter,1u,memory_order_relaxed);jobs[slot]=pos.y*info.x+pos.x;
}
kernel void dispatch_args(device atomic_uint* counter [[buffer(3)]],device uint* args [[buffer(5)]]){args[0]=(atomic_load_explicit(counter,memory_order_relaxed)+63)/64;args[1]=1;args[2]=1;}
kernel void queued_visibility(const device Primitive* p [[buffer(0)]],const device uint* c [[buffer(1)]],constant uint4& info [[buffer(2)]],device atomic_uint* counter [[buffer(3)]],const device uint* jobs [[buffer(4)]],texture2d<float,access::write> output [[texture(0)]],uint index [[thread_position_in_grid]]){
 if(index>=atomic_load_explicit(counter,memory_order_relaxed))return;
 uint pixel=jobs[index];uint2 pos=uint2(pixel%info.x,pixel/info.x);output.write(evaluate_visibility(p,c,info,pos),pos);
}
