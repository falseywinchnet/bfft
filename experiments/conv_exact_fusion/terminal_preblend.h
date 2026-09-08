/* Exact terminal dataflow fusion. Both original admitted profiles are input. */
typedef struct { int cell,anchor; float t[5]; } fused_phase;
typedef struct {
 const float *x,*y,*cy,*cx,*eta;float *out;const double *etax;
 int h,w,oh,ow,c;const fused_phase *px,*py;
} fused_terminal_context;
static void fused_terminal_worker(void *opaque,int begin,int end){
 fused_terminal_context *j=(fused_terminal_context *)opaque;
 const int lx=j->ow*j->c,ly=j->oh*j->c;
 for(int oy=begin;oy<end;++oy){
  const fused_phase *p=j->py+oy;
  double sy=j->oh==1?0.0:(double)oy*(j->h-1)/(j->oh-1);
  int y0=(int)floor(sy);double fy=sy-y0;
  if(y0>=j->h-1){y0=j->h-2;fy=1.0;}
  for(int ox=0;ox<j->ow;++ox){
   const fused_phase *q=j->px+ox;
   const double top=j->etax[y0*j->ow+ox];
   const double bottom=j->etax[(y0+1)*j->ow+ox];
   float beta=(float)((1.0-fy)*top+fy*bottom);
   for(int ch=0;ch<j->c;++ch){
    int a_lane=ox*j->c+ch,b_lane=oy*j->c+ch;
    float a=j->x[(size_t)(p->anchor>=0?p->anchor:p->cell)*lx+a_lane];
    float b=j->y[(size_t)(q->anchor>=0?q->anchor:q->cell)*ly+b_lane];
    if(p->anchor<0)for(int k=0;k<5;++k){
     float z=j->cy[(size_t)(p->cell*5+k)*lx+a_lane];
#if CONV_NEON
     if(a_lane<lx-lx%4)a=fmaf(p->t[k],z,a);else
#endif
     a+=p->t[k]*z;
    }
    if(q->anchor<0)for(int k=0;k<5;++k){
     float z=j->cx[(size_t)(q->cell*5+k)*ly+b_lane];
#if CONV_NEON
     if(b_lane<ly-ly%4)b=fmaf(q->t[k],z,b);else
#endif
     b+=q->t[k]*z;
    }
#if CONV_NEON
    if(ch<j->c-j->c%4)j->out[(size_t)oy*lx+a_lane]=fmaf(beta,b-a,a);else
#endif
    j->out[(size_t)oy*lx+a_lane]=a+beta*(b-a);
   }
  }
 }
}
static void fused_make_phases(fused_phase *p,int n,int m){
 for(int i=0;i<m;++i){
  double t=m==1?0.0:(double)i*(n-1)/(m-1);
  int anchor=(int)llround(t);p[i].anchor=-1;
  if(anchor>=0&&anchor<n&&fabs(t-anchor)<=4.0e-13){p[i].anchor=anchor;p[i].cell=anchor;continue;}
  int cell=(int)floor(t);if(cell>=n-1)cell=n-2;
  p[i].cell=cell;tail_weights(t-cell,p[i].t);
 }
}
API int conv_fused_terminal(const float *x,const float *y,const float *cy,const float *cx,
 const float *eta,int h,int w,int oh,int ow,int c,float *out){
 if(!x||!y||!cy||!cx||!eta||!out||h<5||w<5||oh<1||ow<1||c<1)return -2;
 fused_phase *px=malloc((size_t)ow*sizeof(*px)),*py=malloc((size_t)oh*sizeof(*py));
 if(!px||!py){free(px);free(py);return -1;}
 fused_make_phases(px,w,ow);fused_make_phases(py,h,oh);
 double *etax=malloc((size_t)h*ow*sizeof(*etax));
 if(!etax){free(px);free(py);return -1;}
 for(int ox=0;ox<ow;++ox){
  double sx=ow==1?0.0:(double)ox*(w-1)/(ow-1);
  int x0=(int)floor(sx);double fx=sx-x0;
  if(x0>=w-1){x0=w-2;fx=1.0;}
  for(int iy=0;iy<h;++iy)
   etax[(size_t)iy*ow+ox]=(1.0-fx)*eta[(size_t)iy*w+x0]+fx*eta[(size_t)iy*w+x0+1];
 }
 fused_terminal_context j={x,y,cy,cx,eta,out,etax,h,w,oh,ow,c,px,py};
 parallel_for(oh,8,fused_terminal_worker,&j);free(etax);free(px);free(py);return 0;
}
