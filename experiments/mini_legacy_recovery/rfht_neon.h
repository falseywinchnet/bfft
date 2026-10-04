/* rfht_neon.h - Apple Silicon (ARM NEON) port of the split-radix Hartley
 * magnitude+phase transform. float64x2_t (2-wide) combine.
 *
 * Same math as the AVX2 version; only the inner combine SIMD differs.
 * Build on macOS Apple Silicon:
 *   clang -O3 -mcpu=apple-m1 rfht_neon_example.c -o example -lm
 *   (use -mcpu=apple-m2/m3/m4 as appropriate; -mcpu=native also works)
 */
#ifndef RFHT_NEON_H
#define RFHT_NEON_H
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- accuracy switch ----
 * Default: fast polynomial trig (forward atan ~2e-4 rad, inverse sincos ~1e-8).
 * Define RFHT_ACCURATE (e.g. -DRFHT_ACCURATE) to use libm atan2/sin/cos:
 *   forward phase exact, inverse exact -> round-trip ~1e-15, ~25% slower.
 */
#ifdef RFHT_ACCURATE
  #define RFHT_ATAN2(y,x)        atan2((y),(x))
  #define RFHT_SINCOS(a,sp,cp)   do{ *(sp)=sin(a); *(cp)=cos(a); }while(0)
#else
  #define RFHT_ATAN2(y,x)        rfht__fast_atan2((y),(x))
  #define RFHT_SINCOS(a,sp,cp)   rfht__acc_sincos((a),(sp),(cp))
#endif


typedef struct {
    int N; double *arena,*rev;
    double **C1,**S1,**C3,**S3; int maxL;
} rfht_plan;

static rfht_plan* rfht_plan_create(int N){
    rfht_plan*p=(rfht_plan*)malloc(sizeof(rfht_plan)); p->N=N;
    /* macOS: posix_memalign is reliable; aligned_alloc needs size multiple of align */
    posix_memalign((void**)&p->arena,64,sizeof(double)*4*N);
    posix_memalign((void**)&p->rev,64,sizeof(double)*N);
    int maxL=0; while((1<<maxL)<N) maxL++; p->maxL=maxL;
    p->C1=(double**)calloc(maxL+1,sizeof(double*)); p->S1=(double**)calloc(maxL+1,sizeof(double*));
    p->C3=(double**)calloc(maxL+1,sizeof(double*)); p->S3=(double**)calloc(maxL+1,sizeof(double*));
    for(int L=3;L<=maxL;L++){
        int M=1<<L,M4=M/4,sc=N/M; size_t cnt=((M4+1)&~1); /* even for 2-wide */
        posix_memalign((void**)&p->C1[L],16,sizeof(double)*cnt);
        posix_memalign((void**)&p->S1[L],16,sizeof(double)*cnt);
        posix_memalign((void**)&p->C3[L],16,sizeof(double)*cnt);
        posix_memalign((void**)&p->S3[L],16,sizeof(double)*cnt);
        for(int t=0;t<M4;t++){
            double w1=2.0*M_PI*((long)t*sc%N)/N, w3=2.0*M_PI*((long)3*t*sc%N)/N;
            p->C1[L][t]=cos(w1); p->S1[L][t]=sin(w1);
            p->C3[L][t]=cos(w3); p->S3[L][t]=sin(w3);
        }
    }
    return p;
}
static void rfht_plan_destroy(rfht_plan*p){
    for(int L=3;L<=p->maxL;L++){ free(p->C1[L]);free(p->S1[L]);free(p->C3[L]);free(p->S3[L]); }
    free(p->C1);free(p->S1);free(p->C3);free(p->S3); free(p->arena);free(p->rev);free(p);
}

static void rfht__rec(const double *in,int stride,int M,double *out,rfht_plan*p,double *work){
    if(M==1){ out[0]=in[0]; return; }
    if(M==2){ double a=in[0],b=in[stride]; out[0]=a+b; out[1]=a-b; return; }
    if(M==4){ double x0=in[0],x1=in[stride],x2=in[2*stride],x3=in[3*stride];
        out[0]=x0+x1+x2+x3; out[1]=x0+x1-x2-x3; out[2]=x0-x1+x2-x3; out[3]=x0-x1-x2+x3; return; }
    int M2=M/2,M4=M/4,L=0; { int m=M; while(m>1){ m>>=1; L++; } }
    double *He=work,*Ho1=work+M2,*Ho3=work+M2+M4,*cw=work+M;
    rfht__rec(in,stride*2,M2,He,p,cw);
    rfht__rec(in+stride,stride*4,M4,Ho1,p,cw);
    rfht__rec(in+stride*3,stride*4,M4,Ho3,p,cw);
    double *Ho1r=p->rev,*Ho3r=p->rev+M4;
    Ho1r[0]=Ho1[0]; Ho3r[0]=Ho3[0];
    for(int t=1;t<M4;t++){ Ho1r[t]=Ho1[M4-t]; Ho3r[t]=Ho3[M4-t]; }
    const double *C1b=p->C1[L],*S1b=p->S1[L],*C3b=p->C3[L],*S3b=p->S3[L];
    for(int b=0;b<4;b++){
        double *outb=out+b*M4; const double *Heb=He+((b&1)*M4);
        int q1=b,q3=(3*b)&3;
        int t=0,t2=M4&~1;
        for(t=0;t<t2;t+=2){
            float64x2_t h1=vld1q_f64(Ho1+t),  h1m=vld1q_f64(Ho1r+t);
            float64x2_t h3=vld1q_f64(Ho3+t),  h3m=vld1q_f64(Ho3r+t);
            float64x2_t he=vld1q_f64(Heb+t);
            float64x2_t c1=vld1q_f64(C1b+t),  s1=vld1q_f64(S1b+t);
            float64x2_t c3=vld1q_f64(C3b+t),  s3=vld1q_f64(S3b+t);
            float64x2_t ce1,se1,ce3,se3;
            switch(q1){ case 0: ce1=c1; se1=s1; break;
                        case 1: ce1=vnegq_f64(s1); se1=c1; break;
                        case 2: ce1=vnegq_f64(c1); se1=vnegq_f64(s1); break;
                        default:ce1=s1; se1=vnegq_f64(c1); }
            switch(q3){ case 0: ce3=c3; se3=s3; break;
                        case 1: ce3=vnegq_f64(s3); se3=c3; break;
                        case 2: ce3=vnegq_f64(c3); se3=vnegq_f64(s3); break;
                        default:ce3=s3; se3=vnegq_f64(c3); }
            float64x2_t acc=vfmaq_f64(he,ce1,h1);
            acc=vfmaq_f64(acc,se1,h1m);
            acc=vfmaq_f64(acc,ce3,h3);
            acc=vfmaq_f64(acc,se3,h3m);
            vst1q_f64(outb+t,acc);
        }
        for(;t<M4;t++){
            int qq; double c,s,ce1,se1,ce3,se3;
            qq=q1&3; c=C1b[t]; s=S1b[t];
            ce1=(qq==0)?c:(qq==1)?-s:(qq==2)?-c:s; se1=(qq==0)?s:(qq==1)?c:(qq==2)?-s:-c;
            qq=q3&3; c=C3b[t]; s=S3b[t];
            ce3=(qq==0)?c:(qq==1)?-s:(qq==2)?-c:s; se3=(qq==0)?s:(qq==1)?c:(qq==2)?-s:-c;
            outb[t]=Heb[t]+ce1*Ho1[t]+se1*Ho1r[t]+ce3*Ho3[t]+se3*Ho3r[t];
        }
    }
}
static inline double rfht__fast_atan2(double y,double x){
    double ax=fabs(x),ay=fabs(y);
    double a=(ax>ay)? ay/(ax+1e-300):ax/(ay+1e-300);
    double s=a*a;
    double r=((-0.0464964749*s+0.15931422)*s-0.327622764)*s*a+a;
    if(ay>ax) r=M_PI_2-r; if(x<0.0) r=M_PI-r; if(y<0.0) r=-r; return r;
}
static void rfht_polar(rfht_plan*p,const double *x,double *H_scratch,double *mag,double *phase){
    int N=p->N; rfht__rec(x,1,N,H_scratch,p,p->arena);
    for(int k=0;k<=N/2;k++){
        int nk=(N-k)%N;
        double A=0.5*(H_scratch[k]+H_scratch[nk]);
        double B=0.5*(H_scratch[nk]-H_scratch[k]);
        double phi=RFHT_ATAN2(B,A); if(phi<0.0) phi+=2.0*M_PI;
        phase[k]=phi; mag[k]=sqrt(A*A+B*B);
    }
}
/* inverse */
static inline void rfht__acc_sincos(double a,double*so,double*co){
    const double PIO2=M_PI/2; double k=floor(a/PIO2+0.5); double r=a-k*PIO2; double r2=r*r;
    double sr=r*(1.0+r2*(-1.0/6+r2*(1.0/120+r2*(-1.0/5040+r2*(1.0/362880)))));
    double cr=1.0+r2*(-0.5+r2*(1.0/24+r2*(-1.0/720+r2*(1.0/40320))));
    int q=((int)k)&3;
    switch(q){ case 0:*so=sr;*co=cr;break; case 1:*so=cr;*co=-sr;break;
               case 2:*so=-sr;*co=-cr;break; default:*so=-cr;*co=sr; }
}
static void rfht_ipolar(rfht_plan*p,const double*mag,const double*phase,double*H_scratch,double*x_out){
    int N=p->N;
    { double s,c; RFHT_SINCOS(phase[0],&s,&c); H_scratch[0]=mag[0]*c; }
    { int h=N/2; double s,c; RFHT_SINCOS(phase[h],&s,&c); H_scratch[h]=mag[h]*c; }
    for(int k=1;k<N/2;k++){ int nk=N-k; double s,c; RFHT_SINCOS(phase[k],&s,&c);
        double Re=mag[k]*c,Im=mag[k]*s; H_scratch[k]=Re-Im; H_scratch[nk]=Re+Im; }
    rfht__rec(H_scratch,1,N,x_out,p,p->arena);
    double inv=1.0/N; for(int n=0;n<N;n++) x_out[n]*=inv;
}
#endif
