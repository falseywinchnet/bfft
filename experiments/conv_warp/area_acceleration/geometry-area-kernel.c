/* Knot-clipped CONV integration. Positive triangle cubature samples the exact
 * projective Jacobian. The plan depends only on denominator variation.
 * See geometry-plan.md for the fixed rule table and operator validation. */
#include "triangle-rules.c"
static const double geometry_radius[7]={0,9e-6,.00036,.0023,.0074,.0159,.0276};
typedef struct {double x,y;} Point;
typedef struct {Point p[16];int n;} Polygon;
static double forward[9],area_metrics[8],area_probe[10];
static int area_width,area_height,area_method,area_position;
static double area_det;
EXPORT("forward_pointer") u32 forward_pointer(){return (u32)forward;}
EXPORT("area_metrics_pointer") u32 area_metrics_pointer(){return (u32)area_metrics;}
static double poly_area(Polygon *p){double s=0;for(int i=0;i<p->n;i++){Point a=p->p[i],b=p->p[(i+1)%p->n];s+=a.x*b.y-a.y*b.x;}return __builtin_fabs(s)*.5;}
static Polygon cut(Polygon in,int axis,double bound,int greater){
  Polygon out={.n=0};if(!in.n)return out;Point prev=in.p[in.n-1];double pv=axis?prev.y:prev.x;int inside=greater?pv>=bound:pv<=bound;
  for(int i=0;i<in.n;i++){Point cur=in.p[i];double cv=axis?cur.y:cur.x;int next=greater?cv>=bound:cv<=bound;
    if(next!=inside){double t=(bound-pv)/(cv-pv);out.p[out.n++]=(Point){prev.x+t*(cur.x-prev.x),prev.y+t*(cur.y-prev.y)};}
    if(next)out.p[out.n++]=cur;prev=cur;pv=cv;inside=next;
  }return out;
}
static Polygon rectangle(Polygon p,double x0,double x1,double y0,double y1){return cut(cut(cut(cut(p,0,x0,1),0,x1,0),1,y0,1),1,y1,0);}
static double control_value(int cx,int cy,int i,int j,int ch){u32 k=((u32)(cy*5+j)*lw+cx*5+i)*4+ch;return native_mode?native_control[k]:control[k];}
static Point midpoint(Point a,Point b){return (Point){(a.x+b.x)/2,(a.y+b.y)/2};}
static void integrate_triangle(int cx,int cy,Point a,Point b,Point d,double patch[36][4],double *value,double *leaves){
  Point eb={b.x-a.x,b.y-a.y},ed={d.x-a.x,d.y-a.y};double det=__builtin_fabs(eb.x*ed.y-eb.y*ed.x);
  double da=forward[6]*(a.x+cx)/(sw-1)+forward[7]*(a.y+cy)/(sh-1)+forward[8];
  double db=forward[6]*(b.x+cx)/(sw-1)+forward[7]*(b.y+cy)/(sh-1)+forward[8];
  double dd=forward[6]*(d.x+cx)/(sw-1)+forward[7]*(d.y+cy)/(sh-1)+forward[8];
  double lo=min(da,min(db,dd)),hi=max(da,max(db,dd)),centre=(lo+hi)/2;
  double r=(hi-lo)/(2*__builtin_fabs(centre));
  int degree=0;while(degree<6&&r>geometry_radius[degree])degree++;
  if(r>geometry_radius[6]){
    Point ab=midpoint(a,b),bd=midpoint(b,d),da=midpoint(d,a);
    integrate_triangle(cx,cy,a,ab,da,patch,value,leaves);
    integrate_triangle(cx,cy,ab,b,bd,patch,value,leaves);
    integrate_triangle(cx,cy,da,bd,d,patch,value,leaves);
    integrate_triangle(cx,cy,ab,bd,da,patch,value,leaves);return;
  }
  (*leaves)++;
  double density=det*area_det/((sw-1)*(double)(sh-1));
  double du=db-da,dv=dd-da,constant_weight=density/__builtin_fabs(da*da*da);
  // Compile the 2-D footprint bank before touching any channel controls.
  // One geometry contraction serves every channel of the admitted patch.
  double moments[36]={0};
  for(int kq=0;kq<triangle_count[degree];kq++){
    double u=triangle_rules[degree][kq][0],v=triangle_rules[degree][kq][1];
    double x=a.x+u*eb.x+v*ed.x,y=a.y+u*eb.y+v*ed.y;
    double den=da+u*du+v*dv;
    double wt=triangle_rules[degree][kq][2]*(r==0?constant_weight:density/__builtin_fabs(den*den*den)),bx[6],by[6];
    bernstein(x,bx);bernstein(y,by);
    for(int q=0;q<6;q++){
      v128_t row_weight=wasm_f64x2_splat(wt*by[q]);
      for(int p=0;p<6;p+=2){double *m=moments+q*6+p;wasm_v128_store(m,wasm_f64x2_add(wasm_v128_load(m),wasm_f64x2_mul(row_weight,wasm_v128_load(bx+p))));}
    }
  }
  v128_t rg=wasm_f64x2_splat(0),ba=wasm_f64x2_splat(0);
  for(int k=0;k<36;k++){v128_t m=wasm_f64x2_splat(moments[k]);rg=wasm_f64x2_add(rg,wasm_f64x2_mul(m,wasm_v128_load(patch[k])));ba=wasm_f64x2_add(ba,wasm_f64x2_mul(m,wasm_v128_load(patch[k]+2)));}
  wasm_v128_store(value,wasm_f64x2_add(wasm_v128_load(value),rg));wasm_v128_store(value+2,wasm_f64x2_add(wasm_v128_load(value+2),ba));
}
// Integrated quintic Bernstein weights, compiled from a degree-six basis.
// d/dt [sum_{j=i+1}^6 B_j^6(t)/6] = B_i^5(t).
static void integrated_basis(double t,double *out){
  double b[6],six[7];bernstein(t,b);six[0]=(1-t)*b[0];six[6]=t*b[5];
  for(int j=1;j<6;j++)six[j]=(1-t)*b[j]+t*b[j-1];
  double sum=0;for(int i=5;i>=0;i--){sum+=six[i+1];out[i]=sum/6;}
}
static void integrate_rectangle(Polygon *local,double patch[36][4],double *out){
  double x0=1,x1=0,y0=1,y1=0;for(int k=0;k<local->n;k++){x0=min(x0,local->p[k].x);x1=max(x1,local->p[k].x);y0=min(y0,local->p[k].y);y1=max(y1,local->p[k].y);}
  double a[6],b[6],c[6],d[6];integrated_basis(x0,a);integrated_basis(x1,b);integrated_basis(y0,c);integrated_basis(y1,d);
  double density=area_det/(__builtin_fabs(forward[8]*forward[8]*forward[8])*(sw-1)*(sh-1));
  for(int j=0;j<6;j++)for(int i=0;i<6;i++){double weight=(b[i]-a[i])*(d[j]-c[j])*density;for(int ch=0;ch<4;ch++)out[ch]+=weight*patch[j*6+i][ch];}
}
static void area_pixel(int col,int row,int width,int height,double tolerance,double *out){
  for(int ch=0;ch<10;ch++)out[ch]=0;
  double x0=col==0?0:(col-.5)/(width-1),x1=col==width-1?1:(col+.5)/(width-1),y0=row==0?0:(row-.5)/(height-1),y1=row==height-1?1:(row+.5)/(height-1);
  double target_area=(x1-x0)*(y1-y0);Polygon target={.n=4};for(int i=0;i<4;i++)target.p[i]=(Point){quad[2*i],quad[2*i+1]};target=rectangle(target,x0,x1,y0,y1);
  double covered=poly_area(&target);if(target.n<3||covered<1e-18)return;
  Polygon polygon={.n=target.n};double xmin=sw,xmax=0,ymin=sh,ymax=0;
  for(int i=0;i<target.n;i++){Point q=target.p[i];double den=matrix[6]*q.x+matrix[7]*q.y+matrix[8];Point p={clamp((matrix[0]*q.x+matrix[1]*q.y+matrix[2])/den*(sw-1),0,sw-1),clamp((matrix[3]*q.x+matrix[4]*q.y+matrix[5])/den*(sh-1),0,sh-1)};polygon.p[i]=p;xmin=min(xmin,p.x);xmax=max(xmax,p.x);ymin=min(ymin,p.y);ymax=max(ymax,p.y);}
  double source_area=poly_area(&polygon);if(source_area<1e-18)return;
  double reference[4];sample(xmin,ymin,reference);
  for(int cy=max(0,(int)ymin);cy<=min(sh-2,(int)ymax);cy++)for(int cx=max(0,(int)xmin);cx<=min(sw-2,(int)xmax);cx++){
    Polygon local=rectangle(polygon,cx,cx+1,cy,cy+1);if(local.n<3)continue;
    for(int i=0;i<local.n;i++){local.p[i].x-=cx;local.p[i].y-=cy;}
    // Cache one patch and its constant-reference residual once per piece.
    double patch[36][4];for(int j=0;j<6;j++)for(int i=0;i<6;i++)for(int ch=0;ch<4;ch++){double v=control_value(cx,cy,i,j,ch)-reference[ch];patch[j*6+i][ch]=v;}
    if(forward[1]==0&&forward[3]==0&&forward[6]==0&&forward[7]==0){integrate_rectangle(&local,patch,out);continue;}
    for(int k=1;k<local.n-1;k++){
      Point a=local.p[0],b=local.p[k],d=local.p[k+1];double ar=__builtin_fabs((b.x-a.x)*(d.y-a.y)-(b.y-a.y)*(d.x-a.x))*.5;if(ar<1e-18)continue;
      integrate_triangle(cx,cy,a,b,d,patch,out,out+9);
    }
  }
  for(int ch=0;ch<4;ch++){out[ch]=(reference[ch]*covered+out[ch])/target_area;out[ch+4]=__builtin_nan("");}out[8]=min(1,covered/target_area);
}
static void setup_area(){area_det=__builtin_fabs(forward[0]*(forward[4]*forward[8]-forward[5]*forward[7])-forward[1]*(forward[3]*forward[8]-forward[5]*forward[6])+forward[2]*(forward[3]*forward[7]-forward[4]*forward[6]));}
EXPORT("integrate_pixel") u32 integrate_pixel(int x,int y,int width,int height,double tolerance){setup_area();area_pixel(x,y,width,height,tolerance,area_probe);return (u32)area_probe;}
EXPORT("begin_area") int begin_area(int width,int height,int method){
  if((native_mode?build_phase!=12:!control)||width<5||height<5||width>8192||height>8192||(u32)width*height>maximum_pixels)return 1;
  u32 pixels=(u32)width*height;if(pixels>output_capacity){cursor=source_end;u8 *o=(u8*)reserve(pixels*4),*b=(u8*)reserve(pixels*4);if(!o||!b)return 2;output=o;comparison=b;output_capacity=pixels;}
  area_width=width;area_height=height;area_method=method;area_position=0;setup_area();
  for(int k=0;k<8;k++)area_metrics[k]=0;area_metrics[0]=area_metrics[2]=1e300;area_metrics[1]=area_metrics[3]=-1e300;return 0;
}
EXPORT("step_area") int step_area(int budget){
  int end=min(area_width*area_height,area_position+budget);
  for(;area_position<end;area_position++){
    int x=area_position%area_width,y=area_position/area_width;double v[10];area_pixel(x,y,area_width,area_height,1e-6,v);encode(v,output+area_position*4);
    if(v[8]>0)for(int ch=0;ch<3;ch++){area_metrics[0]=min(area_metrics[0],v[ch]);area_metrics[1]=max(area_metrics[1],v[ch]);}
    area_metrics[4]=__builtin_nan("");area_metrics[5]+=v[9];
    double qx=(double)x/(area_width-1),qy=(double)y/(area_height-1);int inside=1;
    for(int e=0;e<4;e++){int n=(e+1)%4;if((quad[n*2]-quad[e*2])*(qy-quad[e*2+1])-(quad[n*2+1]-quad[e*2+1])*(qx-quad[e*2])<-1e-12){inside=0;break;}}
    ((u32*)comparison)[area_position]=0;if(inside){double den=matrix[6]*qx+matrix[7]*qy+matrix[8];compare_sample(clamp((matrix[0]*qx+matrix[1]*qy+matrix[2])/den*(sw-1),0,sw-1),clamp((matrix[3]*qx+matrix[4]*qy+matrix[5])/den*(sh-1),0,sh-1),area_method,v);encode(v,comparison+area_position*4);for(int ch=0;ch<3;ch++){area_metrics[2]=min(area_metrics[2],v[ch]);area_metrics[3]=max(area_metrics[3],v[ch]);}}
  }return area_position;
}
