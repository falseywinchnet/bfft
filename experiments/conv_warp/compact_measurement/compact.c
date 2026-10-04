/* Compact joint CONV measurement. No full-image control lattice is allocated.
 * Scalar channels reuse one arena. First-factor values and second-factor faces
 * reconstruct bounded 6x6 patches; admission uses the reference constraints.
 */
#include "reference_shell.h"
#include "instrumented_source.h"
static double *grid_x,*grid_y,*result,*correction,*base_ring;
static int exact_storage;
static float *bounds,*patch_ring;
static unsigned short *faces_x,*faces_y;
static int cw,chh,cp,cl,cc,unchanged;
static u32 cells,retained_end,peak_cursor,capacity_count;
static int patch_group(int cx,int cy,int i,int j){
  if(!(i%5)&&!(j%5))return -1;
  if(!(j%5))return exact_storage?(cy+j/5)*(sw-1)+cx:((cy+j/5)%3)*(sw-1)+cx;
  if(!(i%5))return exact_storage?sh*(sw-1)+(cx+i/5)*(sh-1)+cy:3*(sw-1)+(cy%2)*sw+cx+i/5;
  return exact_storage?sh*(sw-1)+sw*(sh-1)+cy*(sw-1)+cx:3*(sw-1)+2*sw+(cy%2)*(sw-1)+cx;
}
static void reset_capacity_row(int row){
  for(int x=0;x<sw-1;x++){capacity[((row+1)%3)*(sw-1)+x]=1;capacity[3*(sw-1)+2*sw+(row%2)*(sw-1)+x]=1;}
  for(int x=0;x<sw;x++)capacity[3*(sw-1)+(row%2)*sw+x]=1;
}
static double local_basis[6][6];
static v128_t inverse_pairs[3][6],tail_pairs[2][5];
/* Reverse sparse banks: source control -> destination indices and weights. */
static u32 *xr,*xi,*yr,*yi;
static double *xw,*yw;
static double jet(const double *a,int i,int n,int stride,int derivative){
  #define Q(k) a[(k)*stride]
  if(derivative==1){
    if(!i)return (-3*Q(0)+4*Q(1)-Q(2))/2;
    if(i==n-1)return (3*Q(i)-4*Q(i-1)+Q(i-2))/2;
    if(i==1||i==n-2)return (Q(i+1)-Q(i-1))/2;
    return (Q(i-2)-8*Q(i-1)+8*Q(i+1)-Q(i+2))/12;
  }
  if(!i)return 2*Q(0)-5*Q(1)+4*Q(2)-Q(3);
  if(i==n-1)return 2*Q(i)-5*Q(i-1)+4*Q(i-2)-Q(i-3);
  if(i==1||i==n-2)return Q(i-1)-2*Q(i)+Q(i+1);
  return (-Q(i+2)+16*Q(i+1)-30*Q(i)+16*Q(i-1)-Q(i-2))/12;
  #undef Q
}
static void replay(const double *a,int n,int stride,int cell,unsigned short bits,double *values){
  double f0=jet(a,cell,n,stride,1),f1=jet(a,cell+1,n,stride,1);
  double s0=jet(a,cell,n,stride,2),s1=jet(a,cell+1,n,stride,2);
  double v0=a[cell*stride],v1=a[(cell+1)*stride];
  double b[6]={v0,v0+f0/5,v0+2*f0/5+s0/20,v1-2*f1/5+s1/20,v1-f1/5,v1};
  double raw5[5],currents[5],numerator=-(v1-v0);int count=0,mask=bits&31;
  for(int k=0;k<5;k++){
    raw5[k]=b[k+1]-b[k];int sign=(bits&(1<<(5+k)))?1:-1;
    if(mask&(1<<k)){numerator+=sign*(sign*raw5[k]);count++;}
  }
  double lambda=count?numerator/count:0;
  for(int k=0;k<5;k++){int sign=(bits&(1<<(5+k)))?1:-1;currents[k]=(mask&(1<<k))?sign*max(0,sign*raw5[k]-lambda*sign):0;}
  values[0]=v0;values[5]=v1;
  for(int pair=0;pair<2;pair++){
    v128_t value=wasm_f64x2_splat(v0);
    for(int k=4;k>=0;k--)value=wasm_f64x2_add(value,wasm_f64x2_mul(tail_pairs[pair][k],wasm_f64x2_splat(currents[k])));
    wasm_v128_store(values+1+pair*2,value);
  }
}
static void bounded_patch(int cx,int cy,int channel,admission_patch *patch){
  double m[6][6],temp[6][6],values[6],mix[6][6];float proposal[36];
  for(int j=0;j<6;j++)for(int i=0;i<6;i++)mix[j][i]=blend(5*cx+i,5*cy+j);
  /* XY: preserve the reference's Float32 store between collocation axes. */
  for(int i=0;i<6;i++){
    int gx=5*cx+i;
    replay(grid_x+gx,sh,lw,cy,faces_x[(u32)gx*(sh-1)+cy],values);
    for(int j=0;j<6;j++)m[j][i]=values[j]*(1-mix[j][i]);
  }
  for(int j=0;j<6;j++)for(int i=0;i<6;i+=2){
    v128_t sum=wasm_f64x2_splat(0);
    for(int k=0;k<6;k++)sum=wasm_f64x2_add(sum,wasm_f64x2_mul(wasm_f64x2_splat(inv[j][k]),wasm_v128_load(m[k]+i)));
    wasm_v128_store(temp[j]+i,wasm_f64x2_promote_low_f32x4(wasm_f32x4_demote_f64x2_zero(sum)));
  }
  for(int j=0;j<6;j++)for(int i=0;i<6;i+=2){
    v128_t sum=wasm_f64x2_splat(0);
    for(int k=0;k<6;k++)sum=wasm_f64x2_add(sum,wasm_f64x2_mul(inverse_pairs[i/2][k],wasm_f64x2_splat(temp[j][k])));
    wasm_v128_store64_lane(proposal+j*6+i,wasm_f32x4_demote_f64x2_zero(sum),0);
  }
  /* YX: ring coefficients remain Float64 before the rounded sum with XY. */
  for(int j=0;j<6;j++){
    int gy=5*cy+j;
    replay(grid_y+(u32)gy*sw,sw,1,cx,faces_y[(u32)gy*(sw-1)+cx],values);
    for(int i=0;i<6;i++)m[j][i]=values[i]*mix[j][i];
  }
  for(int j=0;j<6;j++)for(int i=0;i<6;i+=2){
    v128_t sum=wasm_f64x2_splat(0);
    for(int k=0;k<6;k++)sum=wasm_f64x2_add(sum,wasm_f64x2_mul(inverse_pairs[i/2][k],wasm_f64x2_splat(m[j][k])));
    wasm_v128_store(temp[j]+i,sum);
  }
  for(int j=0;j<6;j++)for(int i=0;i<6;i+=2){
    v128_t sum=wasm_f64x2_splat(0);
    for(int k=0;k<6;k++)sum=wasm_f64x2_add(sum,wasm_f64x2_mul(wasm_f64x2_splat(inv[j][k]),wasm_v128_load(temp[k]+i)));
    sum=wasm_f64x2_add(wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(proposal+j*6+i)),sum);
    wasm_v128_store64_lane(proposal+j*6+i,wasm_f32x4_demote_f64x2_zero(sum),0);
  }
  for(int j=0;j<6;j++)for(int i=0;i<6;i++){
    float value=proposal[j*6+i];
    int gx=5*cx+i,gy=5*cy+j;
    /* Shared controls receive the same incident range intersections in the
     * same row-major order as the whole-image reference. */
    int x0=max(0,cx-(i==0)),x1=min(sw-2,cx+(i==5));
    int y0=max(0,cy-(j==0)),y1=min(sh-2,cy+(j==5));
    for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++){
      u32 id=(u32)y*(sw-1)+x;value=clamp(value,bounds[id*2],bounds[id*2+1]);
    }
    if(!(gx%5)&&!(gy%5))value=src(gx/5,gy/5,channel);
    int k=j*6+i;u32 id=(u32)gy*lw+gx;
    patch->candidate[k]=value;patch->base[k]=base_control(id,channel);patch->group[k]=patch_group(cx,cy,i,j);
    int g=patch->group[k];patch->limited[k]=exact_storage&&cp>=7?patch->base[k]+(g<0?0:capacity[g])*(value-patch->base[k]):patch->base[k];
  }
}
typedef struct {int count,index[5],dx[5],dy[5],group[5];} row_topology;
static row_topology topology[36];
static int entity_class(int i,int j){return (!(i%5)&&!(j%5))?-1:!j?0:!i?1:i==5?2:j==5?3:4;}
static void make_topology(void){
  for(int j=0;j<6;j++)for(int i=0;i<6;i++){
    row_topology *r=topology+j*6+i;r->count=0;
    #define TERM(xx,yy,xxc,yyc) do{int id=(yy)*6+(xx),k=0;while(k<r->count&&r->index[k]!=id)k++;if(k==r->count){r->count++;r->index[k]=id;r->dx[k]=r->dy[k]=0;r->group[k]=entity_class(xx,yy);}r->dx[k]+=(xxc);r->dy[k]+=(yyc);}while(0)
    if(!i){TERM(0,j,-5,0);TERM(1,j,5,0);}else if(i==5){TERM(4,j,-5,0);TERM(5,j,5,0);}else{TERM(i-1,j,-i,0);TERM(i,j,2*i-5,0);TERM(i+1,j,5-i,0);}
    if(!j){TERM(i,0,0,-5);TERM(i,1,0,5);}else if(j==5){TERM(i,4,0,-5);TERM(i,5,0,5);}else{TERM(i,j-1,0,-j);TERM(i,j,0,2*j-5);TERM(i,j+1,0,5-j);}
    #undef TERM
  }
}
static void fast_constraint(int row,direction normal,int phase,admission_patch *patch){
  row_topology *r=topology+row;
  double cm=0,bm=0,lm=0,motion=0,contributions[5]={0,0,0,0,0};
  int seen=0,order[5],ng=0,groups[5];
  for(int k=0;k<r->count;k++){
    int id=r->index[k];double a=r->dx[k]*normal.x+r->dy[k]*normal.y;
    if(a==0)continue;
    double candidate=patch->candidate[id];cm+=a*candidate;
    if(phase==3)continue;
    double base=patch->base[id];bm+=a*base;
    if(phase==1&&r->group[k]>=0){
      int g=r->group[k];if(!(seen&(1<<g))){seen|=1<<g;order[ng++]=g;groups[g]=patch->group[id];}
      contributions[g]+=a*(candidate-base);
    }
    if(phase==2){double limited=patch->limited[id];lm+=a*limited;motion+=a*(candidate-limited);}
  }
  if(phase==1){
    channel_minimum=min(channel_minimum,cm);native_diagnostic[0]++;
    double harm=0;for(int k=0;k<ng;k++)harm+=max(0,-contributions[order[k]]);
    if(harm>STORAGE_EPS){double ratio=min(1,max(0,bm)/harm);for(int k=0;k<ng;k++){int g=order[k];if(contributions[g]<0)capacity[groups[g]]=min(capacity[groups[g]],ratio);}}
  }else if(phase==2){if(motion<-STORAGE_EPS)completion_ray=min(completion_ray,max(0,lm)/-motion);}
  else native_diagnostic[2]=min(native_diagnostic[2],cm);
}
static void normals_cached(int x,int y,int phase,direction *ns,int *n){
  u32 id=(u32)(exact_storage?y:y%2)*(sw-1)+x;
  if(phase==1){
    *n=normals(x,y,cc,ns);cone_counts[id]=*n;native_diagnostic[3]+=*n>0;
    for(int k=0;k<min(2,*n);k++){cone_cache[id*4+2*k]=ns[k].x;cone_cache[id*4+2*k+1]=ns[k].y;}
  }else{
    *n=cone_counts[id];for(int k=0;k<min(2,*n);k++)ns[k]=(direction){cone_cache[id*4+2*k],cone_cache[id*4+2*k+1]};
    if(*n==3)ns[2]=(direction){ns[0].y,-ns[0].x};
  }
}
static void contract(int cx,int cy,int channel,admission_patch *patch,int measurement_phase){
  for(int j=cy?1:0;j<6;j++)for(int i=cx?1:0;i<6;i++){
    int gx=5*cx+i,gy=5*cy+j,k=j*6+i;
    double value=exact_storage?(unchanged?patch->candidate[k]:(float)(patch->limited[k]+completion_ray*(patch->candidate[k]-patch->limited[k]))):(measurement_phase==6?patch->candidate[k]:patch->limited[k]);
    double delta=patch->candidate[k]-patch->limited[k];
    for(u32 a=yr[gy];a<yr[gy+1];a++)for(u32 b=xr[gx];b<xr[gx+1];b++){
      u32 pixel=(u32)yi[a]*cw+xi[b];double weight=yw[a]*xw[b];result[pixel*4+channel]+=weight*value;
      if(!exact_storage&&measurement_phase==7)correction[pixel]+=weight*delta;
    }
  }
}
/* A cell row's entities are final once the following row has contributed
 * capacities. Retain two rows of patches and contract the preceding row. */
static void complete_cached_row(int row){
  for(int x=0;x<sw-1;x++){
    admission_patch patch;float *cached=patch_ring+((u32)(row%2)*(sw-1)+x)*36;
    for(int j=0;j<6;j++)for(int i=0;i<6;i++){
      int k=j*6+i;u32 id=(u32)(5*row+j)*lw+5*x+i;
      double base=base_ring[((u32)(row%2)*(sw-1)+x)*36+k];int g=patch_group(x,row,i,j);
      patch.candidate[k]=cached[k];patch.base[k]=base;patch.group[k]=g;
      patch.limited[k]=base+(g<0?0:capacity[g])*(cached[k]-base);
    }
    direction ns[3];int n;normals_cached(x,row,2,ns,&n);
    for(int k=0;k<n;k++)for(int j=0;j<6;j++)for(int i=0;i<6;i++)fast_constraint(j*6+i,ns[k],2,&patch);
    contract(x,row,cc,&patch,7);
  }
}
EXPORT("prepare_compact") int prepare_compact(int w,int h,int width,int height,int nx,int ny,int strict){
  if(w<5||h<5||width<1||height<1||width>w||height>h||w>16384||h>16384||(double)w*h>16777216)return 1;
  double estimate=(double)w*h*200+(double)width*height*32+(nx+ny)*16.0+max(w,h)*500.0+8388608;
  if(estimate>4000000000.0)return 2;
  exact_storage=strict;cursor=(u32)&__heap_base;sw=w;sh=h;lw=5*(w-1)+1;lh=5*(h-1)+1;cw=width;chh=height;cells=(u32)(w-1)*(h-1);max_axis=max(w,h);
  native_groups=h*(w-1)+w*(h-1)+(w-1)*(h-1);
  #define ALLOC(name,type,count) name=(type*)reserve((u32)(count)*sizeof(type));if(!name)return 2
  ALLOC(native_source,float,(u32)w*h*4);ALLOC(beta,double,(u32)w*h);
  ALLOC(correction,double,(u32)width*height);ALLOC(result,double,(u32)width*height*4);for(u32 k=0;k<(u32)width*height*4;k++)result[k]=0;
  ALLOC(xr,u32,lw+1);ALLOC(xi,u32,nx);ALLOC(xw,double,nx);
  ALLOC(yr,u32,lh+1);ALLOC(yi,u32,ny);ALLOC(yw,double,ny);
  retained_end=cursor;
  ALLOC(grid_x,double,(u32)lw*h);ALLOC(grid_y,double,(u32)w*lh);
  ALLOC(faces_x,unsigned short,(u32)lw*(h-1));ALLOC(faces_y,unsigned short,(u32)lh*(w-1));
  ALLOC(base_ring,double,(sw-1)*72);ALLOC(patch_ring,float,(sw-1)*72);ALLOC(bounds,float,cells*2);capacity_count=exact_storage?native_groups:7*sw-5;ALLOC(capacity,double,capacity_count);
  ALLOC(cone_cache,double,(exact_storage?cells:2*(sw-1))*4);ALLOC(cone_counts,u8,exact_storage?cells:2*(sw-1));
  ALLOC(line,double,max_axis);ALLOC(first,double,max_axis);ALLOC(second,double,max_axis);
  ALLOC(raw,double,max_axis*5);ALLOC(admitted,double,max_axis*5);ALLOC(refined,double,max_axis*5);
  ALLOC(coarse,int,max_axis);ALLOC(locations,int,max_axis);ALLOC(boundary_sign,int,max_axis);
  #undef ALLOC
  peak_cursor=cursor;inverse_collocation();make_topology();for(int r=0;r<6;r++)bernstein(r/5.0,local_basis[r]);
  for(int pair=0;pair<3;pair++)for(int k=0;k<6;k++)inverse_pairs[pair][k]=wasm_f64x2_make(inv[pair*2][k],inv[pair*2+1][k]);
  for(int pair=0;pair<2;pair++){double a=0,b=0;for(int k=4;k>=0;k--){a+=local_basis[1+pair*2][k+1];b+=local_basis[2+pair*2][k+1];tail_pairs[pair][k]=wasm_f64x2_make(a,b);}}
  for(int c=0;c<4;c++)constant_channel[c]=1;
  for(int k=0;k<5;k++)native_diagnostic[k]=0;
  cp=cl=cc=0;face_output=0;faces_only=0;return 0;
}
EXPORT("compact_pointer") u32 compact_pointer(int which){
  switch(which){case 0:return (u32)native_source;case 1:return (u32)result;case 2:return (u32)xr;case 3:return (u32)xi;case 4:return (u32)xw;case 5:return (u32)yr;case 6:return (u32)yi;case 7:return (u32)yw;case 8:return peak_cursor;case 9:return (u32)native_diagnostic;default:return 0;}
}
EXPORT("compact_phase") int compact_phase(){return cp;}
EXPORT("compact_channel") int compact_channel(){return cc;}
EXPORT("compact_step") int compact_step(int budget){
  while(budget-->0&&cp<9){
    int p=cl;
    if(cp==0){
      for(int x=0;x<sw;x++){
        double xx=0,yy=0;for(int c=0;c<4;c++){if(src(x,p,c)!=src(0,0,c))constant_channel[c]=0;double dx=source_jet(x,p,c,0),dy=source_jet(x,p,c,1);xx+=dx*dx;yy+=dy*dy;}
        beta[p*sw+x]=xx+yy>0?yy/(xx+yy):.5;
      }
      if(++cl==sh){cl=0;cp=1;}
    }else if(cp==1&&constant_channel[cc]){
      for(u32 k=0;k<(u32)cw*chh;k++)result[k*4+cc]=src(0,0,cc);
      if(++cc==4)cp=9;
    }else if(cp==1){
      face_output=0;faces_only=0;for(int x=0;x<sw;x++)line[x]=src(x,p,cc);if(refine_line(sw))return -1;
      for(int x=0;x<lw;x++)grid_x[(u32)p*lw+x]=refined[x];
      if(++cl==sh){cl=0;cp=2;}
    }else if(cp==2){
      for(int y=0;y<sh;y++)line[y]=src(p,y,cc);if(refine_line(sh))return -1;
      for(int y=0;y<lh;y++)grid_y[(u32)y*sw+p]=refined[y];
      if(++cl==sw){cl=0;cp=3;faces_only=1;}
    }else if(cp==3){
      for(int y=0;y<sh;y++)line[y]=grid_x[(u32)y*lw+p];face_output=faces_x+(u32)p*(sh-1);if(refine_line(sh))return -1;
      if(++cl==lw){cl=0;cp=4;}
    }else if(cp==4){
      for(int x=0;x<sw;x++)line[x]=grid_y[(u32)p*sw+x];face_output=faces_y+(u32)p*(sw-1);if(refine_line(sw))return -1;
      if(++cl==lh){cl=0;cp=5;face_output=0;faces_only=0;}
    }else if(cp==5){
      for(int x=0;x<sw-1;x++){
        float lo=1e30,hi=-1e30;for(int y=max(0,p-2);y<=min(sh-1,p+3);y++)for(int xx=max(0,x-2);xx<=min(sw-1,x+3);xx++){float v=src(xx,y,cc);lo=min(lo,v);hi=max(hi,v);}
        u32 id=(u32)p*(sw-1)+x;bounds[id*2]=lo;bounds[id*2+1]=hi;
        if(exact_storage){direction ns[3];int n;normals_cached(x,p,1,ns,&n);}
      }
      if(++cl==sh-1){cl=0;cp=6;channel_minimum=1e300;completion_ray=1;for(u32 k=0;k<capacity_count;k++)capacity[k]=1;
        for(u32 k=0;k<(u32)cw*chh;k++)correction[k]=0;}
    }else if(cp==6||cp==7||cp==8){
      if(!exact_storage)reset_capacity_row(p);
      for(int x=0;x<sw-1;x++){
        direction ns[3];int n;normals_cached(x,p,exact_storage?2:1,ns,&n);
        if(exact_storage&&cp!=8&&!n)continue;
        admission_patch patch;bounded_patch(x,p,cc,&patch);
        if(!exact_storage){float *cached=patch_ring+((u32)(p%2)*(sw-1)+x)*36;for(int k=0;k<36;k++){cached[k]=patch.candidate[k];base_ring[((u32)(p%2)*(sw-1)+x)*36+k]=patch.base[k];}}
        if(cp==8){
          contract(x,p,cc,&patch,cp);
          for(int k=0;k<36;k++)patch.candidate[k]=unchanged?patch.candidate[k]:(float)(patch.limited[k]+completion_ray*(patch.candidate[k]-patch.limited[k]));
        }
        for(int k=0;k<n;k++)for(int j=0;j<6;j++)for(int i=0;i<6;i++)fast_constraint(j*6+i,ns[k],cp==6?1:cp==7?2:3,&patch);
      }
      if(!exact_storage){
        if(p>0)complete_cached_row(p-1);
        if(p==sh-2)complete_cached_row(p);
      }
      if(++cl==sh-1){
        cl=0;
        if(cp==6){
          unchanged=channel_minimum>=-STORAGE_EPS;
          if(exact_storage){completion_ray=1;cp=unchanged?8:7;}
          else {
            if(unchanged)completion_ray=1;
            for(u32 k=0;k<(u32)cw*chh;k++)result[k*4+cc]+=completion_ray*correction[k];
            cp=++cc==4?9:1;
          }
        }else if(cp==7){
          if(exact_storage)cp=8;
          else {for(u32 k=0;k<(u32)cw*chh;k++)result[k*4+cc]+=completion_ray*correction[k];cp=++cc==4?9:1;}
        }else {cp=++cc==4?9:1;}
      }
    }
  }
  return cp==9?1:0;
}
/* Independent validation can inspect transient final controls in storage mode. */
static float checked_patch[36];
EXPORT("compact_probe_patch") u32 compact_probe_patch(int x,int y){
  if(!exact_storage||cp!=8||x<0||x>=sw-1||y<0||y>=sh-1)return 0;
  admission_patch p;bounded_patch(x,y,cc,&p);
  for(int k=0;k<36;k++)checked_patch[k]=unchanged?p.candidate[k]:(float)(p.limited[k]+completion_ray*(p.candidate[k]-p.limited[k]));
  return (u32)checked_patch;
}
