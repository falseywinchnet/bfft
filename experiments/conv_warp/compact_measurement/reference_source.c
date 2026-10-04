/* Native whole-image joint CONV construction. The source is never resized.
 * The two fifth-step factor orders are streamed into one shared Float32
 * control lattice. All factor, collocation and admission arithmetic is Float64.
 * Work storage is reused for the reverse order and per-channel capacities.
 */
static float *native_control, *native_source;
static double *stage, *line, *first, *second, *raw, *admitted, *refined, *beta;
static double *ring, *capacity, *cone_cache;
static u8 *cone_counts;
static int constant_channel[4];
static int *coarse, *locations, *boundary_sign;
static double inv[6][6], native_diagnostic[5];
static int lh, max_axis, build_phase, build_lane, build_channel;
static u32 native_work, native_nodes, native_groups;
static double channel_minimum, completion_ray;
#define EPS 2.22044604925031308085e-16
// Float32 lattice rounding, including collocation, must not trigger a global
// completion collapse on a mathematically flat current. The measured final
// margin is reported separately; this is a storage tolerance, not exactness.
#define STORAGE_EPS 0.000008
static double src(int x,int y,int ch){return native_source[((u32)y*sw+x)*4+ch];}
static int signum(double x){return (x>0)-(x<0);}
static double base_control(u32 id,int ch){
  int x=id%lw,y=id/lw,cx=min(sw-2,x/5),cy=min(sh-2,y/5);
  double u=(x-5*cx)/5.0,v=(y-5*cy)/5.0;
  return (1-u)*(1-v)*src(cx,cy,ch)+u*(1-v)*src(cx+1,cy,ch)+(1-u)*v*src(cx,cy+1,ch)+u*v*src(cx+1,cy+1,ch);
}
static int owner(u32 id){
  int x=id%lw,y=id/lw,rx=x%5,ry=y%5;
  if(!rx&&!ry)return -1;
  if(!ry)return (y/5)*(sw-1)+x/5;
  int horizontal=sh*(sw-1);
  if(!rx)return horizontal+(x/5)*(sh-1)+y/5;
  return horizontal+sw*(sh-1)+(y/5)*(sw-1)+x/5;
}
static void inverse_collocation(void){
  double a[6][12];
  for(int r=0;r<6;r++){bernstein(r/5.0,a[r]);for(int c=0;c<6;c++)a[r][6+c]=r==c;}
  for(int c=0;c<6;c++){
    int p=c;for(int r=c+1;r<6;r++)if(__builtin_fabs(a[r][c])>__builtin_fabs(a[p][c]))p=r;
    for(int k=0;k<12;k++){double t=a[c][k];a[c][k]=a[p][k];a[p][k]=t;}
    double d=a[c][c];for(int k=0;k<12;k++)a[c][k]/=d;
    for(int r=0;r<6;r++)if(r!=c){double f=a[r][c];for(int k=0;k<12;k++)a[r][k]-=f*a[c][k];}
  }
  for(int r=0;r<6;r++)for(int c=0;c<6;c++)inv[r][c]=a[r][c+6];
}
/* Input is the complete original line, including sign witnesses across plateaus. */
static int refine_line(int n){
  int intervals=n-1;
  first[0]=(-3*line[0]+4*line[1]-line[2])/2;
  first[1]=(line[2]-line[0])/2;
  first[n-2]=(line[n-1]-line[n-3])/2;
  first[n-1]=(3*line[n-1]-4*line[n-2]+line[n-3])/2;
  second[0]=2*line[0]-5*line[1]+4*line[2]-line[3];
  second[1]=line[0]-2*line[1]+line[2];
  second[n-2]=line[n-3]-2*line[n-2]+line[n-1];
  second[n-1]=2*line[n-1]-5*line[n-2]+4*line[n-3]-line[n-4];
  for(int i=2;i<n-2;i++){
    first[i]=(line[i-2]-8*line[i-1]+8*line[i+1]-line[i+2])/12;
    second[i]=(-line[i+2]+16*line[i+1]-30*line[i]+16*line[i-1]-line[i-2])/12;
  }
  for(int i=0;i<intervals;i++){
    double b[6]={line[i],line[i]+first[i]/5,line[i]+2*first[i]/5+second[i]/20,line[i+1]-2*first[i+1]/5+second[i+1]/20,line[i+1]-first[i+1]/5,line[i+1]};
    for(int k=0;k<5;k++){raw[5*i+k]=b[k+1]-b[k];admitted[5*i+k]=0;}
    coarse[i]=signum(line[i+1]-line[i]);
  }
  int witness=0;while(witness<intervals&&!coarse[witness])witness++;
  if(witness<intervals){
    for(int i=0;i<witness;i++)coarse[i]=coarse[witness];
    for(int i=witness+1;i<intervals;i++)if(!coarse[i])coarse[i]=coarse[i-1];
    int count=0,previous=0;
    for(int knot=1;knot<intervals;knot++)if(coarse[knot-1]!=coarse[knot]){
      int centre=5*knot,best=centre;double best_cost=1e300;
      for(int candidate=max(previous+1,centre-4);candidate<min(5*intervals,centre+5);candidate++){
        double cost=0;for(int k=max(0,centre-5);k<min(5*intervals,centre+5);k++)if((k<candidate?coarse[knot-1]:coarse[knot])*raw[k]<0)cost+=raw[k]*raw[k];
        if(cost<best_cost){best_cost=cost;best=candidate;}
      }
      locations[count]=best;boundary_sign[count++]=coarse[knot];previous=best;
    }
    int boundary=0,current_sign=coarse[witness];
    for(int cell=0;cell<intervals;cell++){
      int signs[5],order[5],masks[6],mask=0,mask_count=0;double target[5],best_values[5],best_error=1e300;
      for(int k=0;k<5;k++){
        while(boundary<count&&5*cell+k>=locations[boundary])current_sign=boundary_sign[boundary++];
        signs[k]=current_sign;target[k]=signs[k]*raw[5*cell+k];order[k]=k;
        if(signs[k]>0)mask|=1<<k;
      }
      for(int k=1;k<5;k++){int e=order[k],p=k;while(p>0&&raw[5*cell+order[p-1]]>raw[5*cell+e]){order[p]=order[p-1];p--;}order[p]=e;}
      if(mask)masks[mask_count++]=mask;
      for(int k=0;k<5;k++){mask^=1<<order[k];if(mask)masks[mask_count++]=mask;}
      double total=line[cell+1]-line[cell];int found=0;
      for(int m=0;m<mask_count;m++){
        mask=masks[m];double numerator=-total;int active=0;
        for(int k=0;k<5;k++)if(mask&(1<<k)){numerator+=signs[k]*target[k];active++;}
        double lambda=numerator/active,value[5],mass=0,error=0;int valid=1;
        for(int k=0;k<5;k++){
          value[k]=(mask&(1<<k))?target[k]-lambda*signs[k]:0;
          if(value[k]<-64*EPS){valid=0;break;}value[k]=max(0,value[k]);
          mass+=signs[k]*value[k];double residual=value[k]-target[k];error+=residual*residual;
        }
        if(valid&&__builtin_fabs(mass-total)<=256*EPS*max(1,__builtin_fabs(total))&&error<best_error){
          found=1;best_error=error;for(int k=0;k<5;k++)best_values[k]=value[k];
        }
      }
      if(!found)return 1;
      for(int k=0;k<5;k++)admitted[5*cell+k]=signs[k]*best_values[k];
    }
  }
  for(int i=0;i<intervals;i++)for(int r=0;r<5;r++){
    if(!r){refined[5*i]=line[i];continue;}
    double b[6],suffix=0,value=line[i];bernstein(r/5.0,b);
    for(int k=4;k>=0;k--){suffix+=b[k+1];value+=suffix*admitted[5*i+k];}refined[5*i+r]=value;
  }
  refined[5*intervals]=line[n-1];return 0;
}
static double source_jet(int x,int y,int ch,int vertical){
  int i=vertical?y:x,n=vertical?sh:sw;
  #define V(k) (vertical?src(x,(k),ch):src((k),y,ch))
  if(!i)return (-3*V(0)+4*V(1)-V(2))/2;
  if(i==n-1)return (3*V(i)-4*V(i-1)+V(i-2))/2;
  if(i==1||i==n-2)return (V(i+1)-V(i-1))/2;
  return (V(i-2)-8*V(i-1)+8*V(i+1)-V(i+2))/12;
  #undef V
}
static double blend(int x,int y){
  int cx=min(sw-2,x/5),cy=min(sh-2,y/5);double u=(x-5*cx)/5.0,v=(y-5*cy)/5.0;
  return (1-u)*(1-v)*beta[cy*sw+cx]+u*(1-v)*beta[cy*sw+cx+1]+(1-u)*v*beta[(cy+1)*sw+cx]+u*v*beta[(cy+1)*sw+cx+1];
}
/* Geometry of the smallest containing positive cone, without atan2/trig. */
typedef struct {double x,y;} direction;
static int half(direction a){return a.y<0||(a.y==0&&a.x<0);}
static double cross(direction a,direction b){return a.x*b.y-a.y*b.x;}
static int normals(int cx,int cy,int ch,direction *result){
  if(constant_channel[ch])return 0;
  direction dirs[100],anchor={0,0},upper={0,0},lower={0,0};int count=0,has_upper=0,has_lower=0;
  const double witness_tolerance=16384*EPS;
  int x0=max(0,cx-2),x1=min(sw-1,cx+3),y0=max(0,cy-2),y1=min(sh-1,cy+3);
  for(int y=y0;y<y1;y++)for(int x=x0;x<x1;x++){
    double a=src(x,y,ch),b=src(x+1,y,ch),d=src(x,y+1,ch),e=src(x+1,y+1,ch);
    direction v[4]={{b-a,d-a},{b-a,e-b},{e-d,d-a},{e-d,e-b}};
    for(int k=0;k<4;k++){
      double length=__builtin_sqrt(v[k].x*v[k].x+v[k].y*v[k].y);if(length<=2048*EPS)continue;
      v[k].x/=length;v[k].y/=length;
      // Three rays with strictly positive cyclic cross products already span
      // the plane. More support vectors cannot restore a directional bound.
      // Keep the original sorted construction for every restricted cone.
      if(!count)anchor=v[k];
      else {
        double side=cross(anchor,v[k]);
        if(side>witness_tolerance&&(!has_upper||cross(upper,v[k])>0)){upper=v[k];has_upper=1;}
        if(side<-witness_tolerance&&(!has_lower||cross(v[k],lower)>0)){lower=v[k];has_lower=1;}
        if(has_upper&&has_lower&&cross(upper,lower)>witness_tolerance)return 0;
      }
      int p=count;
      while(p>0&&(half(dirs[p-1])>half(v[k])||(half(dirs[p-1])==half(v[k])&&cross(dirs[p-1],v[k])<0))){dirs[p]=dirs[p-1];p--;}
      dirs[p]=v[k];count++;
    }
  }
  if(!count)return 0;
  direction start=dirs[0],last=dirs[count-1];int cut=-1,opposite=-1;double tol=4096*EPS;
  for(int i=0;i<count;i++){
    direction a=dirs[i],b=dirs[(i+1)%count];double cr=cross(a,b),dot=a.x*b.x+a.y*b.y;
    if(cr < -tol){cut=i;break;}
    if(__builtin_fabs(cr)<=tol&&dot<0)opposite=i;
  }
  if(cut>=0){start=dirs[(cut+1)%count];last=dirs[cut];}
  else if(opposite>=0){
    start=dirs[(opposite+1)%count];last=dirs[opposite];result[0]=(direction){-start.y,start.x};
    for(int i=0;i<count;i++)if(cross(start,dirs[i])>tol)return 1;
    result[1]=(direction){start.y,-start.x};return 2;
  }else{
    for(int i=1;i<count;i++)if(__builtin_fabs(cross(start,dirs[i]))>tol||start.x*dirs[i].x+start.y*dirs[i].y<0)return 0;
    result[0]=(direction){-start.y,start.x};result[1]=(direction){start.y,-start.x};result[2]=start;return 3;
  }
  result[0]=(direction){-start.y,start.x};result[1]=(direction){last.y,-last.x};return 2;
}
static void clipped_cell(int cx,int cy){
  double low[4]={1e300,1e300,1e300,1e300},high[4]={-1e300,-1e300,-1e300,-1e300};
  for(int y=max(0,cy-2);y<=min(sh-1,cy+3);y++)for(int x=max(0,cx-2);x<=min(sw-1,cx+3);x++)for(int ch=0;ch<4;ch++){
    double v=src(x,y,ch);low[ch]=min(low[ch],v);high[ch]=max(high[ch],v);
  }
  for(int j=0;j<6;j++)for(int i=0;i<6;i++){
    u32 id=(u32)(cy*5+j)*lw+cx*5+i;
    for(int ch=0;ch<4;ch++){
      if(constant_channel[ch]){native_control[id*4+ch]=src(0,0,ch);continue;}
      float v=native_control[id*4+ch],bounded=clamp(v,low[ch],high[ch]);
      native_diagnostic[1]+=v!=bounded;native_control[id*4+ch]=bounded;
      if(!(i%5)&&!(j%5))native_control[id*4+ch]=src(cx+i/5,cy+j/5,ch);
    }
  }
}
typedef struct {double candidate[36],base[36],limited[36];int group[36];} admission_patch;
static void constraint_row(int i,int j,direction normal,int phase,admission_patch *patch){
  u32 ids[5];double weights[5];int count=0;
  #define ADD(xx,yy,vv) do {u32 id=(yy)*6+(xx);double v=(vv);if(v!=0){int k=0;while(k<count&&ids[k]!=id)k++;if(k==count){ids[count]=id;weights[count++]=v;}else weights[k]+=v;}}while(0)
  double dx=normal.x,dy=normal.y;
  if(!i){ADD(0,j,-5*dx);ADD(1,j,5*dx);}else if(i==5){ADD(4,j,-5*dx);ADD(5,j,5*dx);}else{ADD(i-1,j,-i*dx);ADD(i,j,(2*i-5)*dx);ADD(i+1,j,(5-i)*dx);}
  if(!j){ADD(i,0,-5*dy);ADD(i,1,5*dy);}else if(j==5){ADD(i,4,-5*dy);ADD(i,5,5*dy);}else{ADD(i,j-1,-j*dy);ADD(i,j,(2*j-5)*dy);ADD(i,j+1,(5-j)*dy);}
  #undef ADD
  double candidate_margin=0,base_margin=0,limited_margin=0,motion=0;
  int groups[5],ng=0;double contributions[5];
  for(int k=0;k<count;k++){
    u32 id=ids[k];double a=weights[k],candidate=patch->candidate[id];candidate_margin+=a*candidate;
    if(phase==0||phase==3)continue;
    double base=patch->base[id];base_margin+=a*base;int g=patch->group[id];
    if(phase==1&&g>=0){int p=0;while(p<ng&&groups[p]!=g)p++;if(p==ng){groups[ng]=g;contributions[ng++]=0;}contributions[p]+=a*(candidate-base);}
    if(phase==2){double limited=patch->limited[id];limited_margin+=a*limited;motion+=a*(candidate-limited);}
  }
  if(phase==0){native_diagnostic[0]++;channel_minimum=min(channel_minimum,candidate_margin);}
  if(phase==1){double harm=0;for(int p=0;p<ng;p++)harm+=max(0,-contributions[p]);if(harm>STORAGE_EPS){double ratio=min(1,max(0,base_margin)/harm);for(int p=0;p<ng;p++)if(contributions[p]<0)capacity[groups[p]]=min(capacity[groups[p]],ratio);}}
  if(phase==2&&motion<-STORAGE_EPS)completion_ray=min(completion_ray,max(0,limited_margin)/-motion);
  if(phase==3)native_diagnostic[2]=min(native_diagnostic[2],candidate_margin);
}
static void admit_cells(int y,int ch,int phase){
  for(int x=0;x<sw-1;x++){
    direction ns[3];int n;u32 cell=(u32)y*(sw-1)+x;
    if(phase==0){
      n=normals(x,y,ch,ns);cone_counts[cell]=n;native_diagnostic[3]+=n>0;
      for(int k=0;k<min(2,n);k++){cone_cache[cell*4+2*k]=ns[k].x;cone_cache[cell*4+2*k+1]=ns[k].y;}
    }else{
      n=cone_counts[cell];
      for(int k=0;k<min(2,n);k++)ns[k]=(direction){cone_cache[cell*4+2*k],cone_cache[cell*4+2*k+1]};
      if(n==3)ns[2]=(direction){ns[0].y,-ns[0].x};
    }
    if(!n)continue;
    // Every directional row reuses these same 36 values and group owners.
    // In particular, do not reconstruct the bilinear base hundreds of times.
    admission_patch patch;
    for(int j=0;j<6;j++)for(int i=0;i<6;i++){
      int local=j*6+i;u32 id=(u32)(y*5+j)*lw+x*5+i;
      patch.candidate[local]=native_control[id*4+ch];
      if(phase==1||phase==2){
        double base=base_control(id,ch);int g=owner(id);patch.base[local]=base;patch.group[local]=g;
        if(phase==2)patch.limited[local]=base+(g<0?0:capacity[g])*(patch.candidate[local]-base);
      }
    }
    for(int k=0;k<n;k++)for(int j=0;j<6;j++)for(int i=0;i<6;i++)constraint_row(i,j,ns[k],phase,&patch);
  }
}
EXPORT("prepare_native") int prepare_native(int w,int h){
  if(w<5||h<5||w>16384||h>16384||(double)w*h>16777216)return 1;
  // Fail before allocating if the whole-image control lattice cannot fit wasm32.
  double estimate=(double)(5*(w-1)+1)*(5*(h-1)+1)*16+(double)w*h*24+max((double)(5*(w-1)+1)*h,(double)w*(5*(h-1)+1))*32+max(w,h)*400+8388608;
  if(estimate>4000000000.0)return 2;
  cursor=(u32)&__heap_base;native_mode=1;sw=w;sh=h;lw=5*(w-1)+1;lh=5*(h-1)+1;max_axis=max(w,h);
  native_nodes=(u32)lw*lh;native_groups=sh*(sw-1)+sw*(sh-1)+(sw-1)*(sh-1);
  native_control=(float*)reserve(native_nodes*16);native_source=(float*)reserve((u32)w*h*16);
  matrix=(double*)reserve(9*8);quad=(double*)reserve(8*8);metrics=(double*)reserve(4*8);probe=(double*)reserve(4*8);
  if(!native_control||!native_source||!matrix||!quad||!metrics||!probe)return 2;
  source_end=cursor;native_work=cursor;
  stage=(double*)reserve((u32)max((double)lw*h,(double)w*lh)*4*8);
  beta=(double*)reserve((u32)w*h*8);
  line=(double*)reserve(max_axis*8);first=(double*)reserve(max_axis*8);second=(double*)reserve(max_axis*8);
  raw=(double*)reserve(max_axis*5*8);admitted=(double*)reserve(max_axis*5*8);refined=(double*)reserve(max_axis*5*8);
  coarse=(int*)reserve(max_axis*4);locations=(int*)reserve(max_axis*4);boundary_sign=(int*)reserve(max_axis*4);
  ring=(double*)reserve((u32)lw*6*4*8);
  if(!stage||!beta||!line||!first||!second||!raw||!admitted||!refined||!coarse||!locations||!boundary_sign||!ring)return 2;
  maximum_pixels=16777216;output_capacity=0;output=comparison=0;
  build_phase=0;build_lane=0;build_channel=0;
  for(int c=0;c<4;c++)constant_channel[c]=1;
  for(int k=0;k<5;k++)native_diagnostic[k]=0;
  native_diagnostic[4]=(double)(w-1)*(h-1)*4;
  inverse_collocation();return 0;
}
EXPORT("native_source_pointer") u32 native_source_pointer(){return (u32)native_source;}
EXPORT("native_control_pointer") u32 native_control_pointer(){return (u32)native_control;}
EXPORT("native_diagnostic_pointer") u32 native_diagnostic_pointer(){return (u32)native_diagnostic;}
EXPORT("native_phase") int native_phase(){return build_phase;}
/* One bounded batch per call lets the worker report progress or be terminated. */
EXPORT("build_native_step") int build_native_step(int budget){
  while(budget-->0&&build_phase<12){
    int p=build_lane,ch=build_channel;
    if(build_phase==0){
      for(int x=0;x<sw;x++){double xx=0,yy=0;for(int c=0;c<4;c++){if(src(x,p,c)!=src(0,0,c))constant_channel[c]=0;double dx=source_jet(x,p,c,0),dy=source_jet(x,p,c,1);xx+=dx*dx;yy+=dy*dy;}beta[p*sw+x]=xx+yy>0?yy/(xx+yy):.5;}
      if(++build_lane==sh){build_lane=0;build_phase++;}
    }else if(build_phase==1){
      for(int c=0;c<4;c++){if(constant_channel[c])continue;for(int x=0;x<sw;x++)line[x]=src(x,p,c);if(refine_line(sw))return -1;for(int x=0;x<lw;x++)stage[((u32)p*lw+x)*4+c]=refined[x];}
      if(++build_lane==sh){build_lane=0;build_phase++;}
    }else if(build_phase==2){
      for(int c=0;c<4;c++){
        if(constant_channel[c])continue;
        for(int y=0;y<sh;y++)line[y]=stage[((u32)y*lw+p)*4+c];if(refine_line(sh))return -1;
        for(int y=0;y<lh;y++)refined[y]*=1-blend(p,y);
        for(int cy=0;cy<sh-1;cy++)for(int j=cy?1:0;j<6;j++){
          double sum=0;for(int k=0;k<6;k++)sum+=inv[j][k]*refined[5*cy+k];native_control[((u32)(5*cy+j)*lw+p)*4+c]=sum;
        }
      }
      if(++build_lane==lw){build_lane=0;build_phase++;}
    }else if(build_phase==3){
      for(int cx=0;cx<sw-1;cx++)for(int c=0;c<4;c++){
        if(constant_channel[c])continue;
        double values[6];for(int k=0;k<6;k++)values[k]=native_control[((u32)p*lw+5*cx+k)*4+c];
        for(int i=0;i<6;i++){double sum=0;for(int k=0;k<6;k++)sum+=inv[i][k]*values[k];native_control[((u32)p*lw+5*cx+i)*4+c]=sum;}
      }
      if(++build_lane==lh){build_lane=0;build_phase++;}
    }else if(build_phase==4){
      for(int c=0;c<4;c++){if(constant_channel[c])continue;for(int y=0;y<sh;y++)line[y]=src(p,y,c);if(refine_line(sh))return -1;for(int y=0;y<lh;y++)stage[((u32)y*sw+p)*4+c]=refined[y];}
      if(++build_lane==sw){build_lane=0;build_phase++;}
    }else if(build_phase==5){
      int row=p%5;if(p&&row==0)row=5;
      for(int c=0;c<4;c++){
        if(constant_channel[c])continue;
        for(int x=0;x<sw;x++)line[x]=stage[((u32)p*sw+x)*4+c];if(refine_line(sw))return -1;
        for(int x=0;x<lw;x++)refined[x]*=blend(x,p);
        for(int cx=0;cx<sw-1;cx++)for(int i=cx?1:0;i<6;i++){
          double sum=0;for(int k=0;k<6;k++)sum+=inv[i][k]*refined[5*cx+k];ring[((u32)row*lw+5*cx+i)*4+c]=sum;
        }
      }
      if(row==5){
        int cy=p/5-1;
        for(int j=cy?1:0;j<6;j++)for(int x=0;x<lw;x++)for(int c=0;c<4;c++){
          if(constant_channel[c])continue;
          double sum=0;for(int k=0;k<6;k++)sum+=inv[j][k]*ring[((u32)k*lw+x)*4+c];native_control[((u32)(5*cy+j)*lw+x)*4+c]+=sum;
        }
        for(u32 k=0;k<(u32)lw*4;k++)ring[k]=ring[(u32)5*lw*4+k];
      }
      if(++build_lane==lh){build_lane=0;build_phase++;cursor=native_work;capacity=(double*)reserve(native_groups*8);cone_cache=(double*)reserve((u32)(sw-1)*(sh-1)*32);cone_counts=(u8*)reserve((u32)(sw-1)*(sh-1));if(!capacity||!cone_cache||!cone_counts)return -2;}
    }else if(build_phase==6){
      for(int x=0;x<sw-1;x++)clipped_cell(x,p);
      if(++build_lane==sh-1){build_lane=0;build_phase++;channel_minimum=1e300;}
    }else if(build_phase==7){
      admit_cells(p,ch,0);
      if(++build_lane==sh-1){build_lane=0;if(channel_minimum>=-STORAGE_EPS){native_diagnostic[2]=min(native_diagnostic[2],channel_minimum);if(++build_channel==4)build_phase=12;channel_minimum=1e300;}else{for(u32 k=0;k<native_groups;k++)capacity[k]=1;build_phase++;}}
    }else if(build_phase==8){
      admit_cells(p,ch,1);
      if(++build_lane==sh-1){build_lane=0;build_phase++;completion_ray=1;}
    }else if(build_phase==9){
      admit_cells(p,ch,2);
      if(++build_lane==sh-1){build_lane=0;build_phase++;}
    }else if(build_phase==10){
      for(int x=0;x<lw;x++){
        u32 id=(u32)p*lw+x;int g=owner(id);double base=base_control(id,ch),candidate=native_control[id*4+ch],limited=base+(g<0?0:capacity[g])*(candidate-base);
        native_control[id*4+ch]=limited+completion_ray*(candidate-limited);
      }
      if(++build_lane==lh){build_lane=0;build_phase++;}
    }else if(build_phase==11){
      admit_cells(p,ch,3);
      if(++build_lane==sh-1){build_lane=0;channel_minimum=1e300;build_phase=++build_channel==4?12:7;}
    }
  }
  if(build_phase==12){cursor=source_end;return 1;}return 0;
}
// Monotone stage progress across all four channel-admission passes.
EXPORT("native_progress") double native_progress(){
  if(build_phase==12)return 1;
  int lengths[12]={sh,sh,lw,lh,sw,lh,sh-1,sh-1,sh-1,sh-1,lh,sh-1};
  double fraction=(double)build_lane/lengths[build_phase];
  return build_phase<7?.35*(build_phase+fraction)/7:.35+.65*(build_channel+(build_phase-7+fraction)/5)/4;
}
EXPORT("native_channel") int native_channel(){return build_channel;}
