/* rfht_polar.h - real-input Fourier magnitude+phase via split-radix Hartley transform
 *
 * Computes, for a real input x[0..N-1] (N a power of two, N>=8):
 *   mag[k], phase[k] for k = 0..N/2   (the non-redundant half-spectrum)
 * where (mag,phase) is the polar form of the forward DFT bin
 *   X[k] = sum_n x[n] e^{-j 2 pi k n / N},  with A=Re(X[k]), B=-Im... see notes.
 *
 * Method:
 *   1. Hartley transform H (split-radix, real arithmetic, no complex type).
 *   2. Recover real projections  A[k] =  (H[k]+H[N-k])/2   (= sum x cos)
 *                                B[k] = -(H[k]-H[N-k])/2   (= -sum x sin, forward-DFT convention)
 *   3. phase[k] = maximizer of  C(phi)=A cos phi + B sin phi  (your construction).
 *      This maximizer equals atan2(B,A); evaluated here by a real polynomial arctan
 *      (multiply/add only, no transcendental call, no complex coefficient).
 *   4. mag[k]   = sqrt(A^2+B^2).   (See MAG_NO_SQRT below for the no-sqrt variant.)
 *
 * CONTRACT:
 *   PRE:  N power of two, N>=8. x length N. mag/phase length >= N/2+1.
 *         Call rfht_plan_create(N) once; reuse plan across transforms; destroy at end.
 *   POST: mag[k] machine-precision; phase[k] in [0,2*pi), abs error ~2e-4 rad
 *         (set by the cubic arctan; raise polynomial degree for tighter).
 *   GUARANTEES: no complex type; no atan2/atan call; phase = real argmax object.
 *   DOES NOT GUARANTEE: non-power-of-two N; NaN/Inf handling; thread-safety of a
 *         SHARED plan across threads (plan has per-call scratch -> use one plan per thread);
 *         phase meaning when A=B=0 (returns 0); input validation.
 */
#ifndef RFHT_POLAR_H
#define RFHT_POLAR_H
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int N;
    double *arena;   /* recursion scratch, length 4*N */
    double *rev;     /* reversal scratch, length N    */
    double **C1,**S1,**C3,**S3; /* per-level twiddles, indexed by L=log2(M) */
    int maxL;
} rfht_plan;

static rfht_plan* rfht_plan_create(int N){
    rfht_plan*p=(rfht_plan*)malloc(sizeof(rfht_plan));
    p->N=N;
    p->arena=(double*)aligned_alloc(64,sizeof(double)*4*N);
    p->rev  =(double*)aligned_alloc(64,sizeof(double)*N);
    int maxL=0; while((1<<maxL)<N) maxL++;
    p->maxL=maxL;
    p->C1=(double**)calloc(maxL+1,sizeof(double*));
    p->S1=(double**)calloc(maxL+1,sizeof(double*));
    p->C3=(double**)calloc(maxL+1,sizeof(double*));
    p->S3=(double**)calloc(maxL+1,sizeof(double*));
    for(int L=3; L<=maxL; L++){
        int M=1<<L, M4=M/4, sc=N/M;
        size_t z=sizeof(double)*((M4+3)&~3);
        p->C1[L]=(double*)aligned_alloc(32,z);
        p->S1[L]=(double*)aligned_alloc(32,z);
        p->C3[L]=(double*)aligned_alloc(32,z);
        p->S3[L]=(double*)aligned_alloc(32,z);
        for(int t=0;t<M4;t++){
            double w1=2.0*M_PI*((long)t*sc % N)/N;
            double w3=2.0*M_PI*((long)3*t*sc % N)/N;
            p->C1[L][t]=cos(w1); p->S1[L][t]=sin(w1);
            p->C3[L][t]=cos(w3); p->S3[L][t]=sin(w3);
        }
    }
    return p;
}
static void rfht_plan_destroy(rfht_plan*p){
    for(int L=3;L<=p->maxL;L++){ free(p->C1[L]);free(p->S1[L]);free(p->C3[L]);free(p->S3[L]); }
    free(p->C1);free(p->S1);free(p->C3);free(p->S3);
    free(p->arena);free(p->rev);free(p);
}

/* ---- split-radix Hartley, recursive, AVX2 combine ---- */
static void rfht__rec(const double *in,int stride,int M,double *out,rfht_plan*p,double *work){
    if(M==1){ out[0]=in[0]; return; }
    if(M==2){ double a=in[0],b=in[stride]; out[0]=a+b; out[1]=a-b; return; }
    if(M==4){
        double x0=in[0],x1=in[stride],x2=in[2*stride],x3=in[3*stride];
        out[0]=x0+x1+x2+x3; out[1]=x0+x1-x2-x3; out[2]=x0-x1+x2-x3; out[3]=x0-x1-x2+x3; return;
    }
    int M2=M/2,M4=M/4,L=0; { int m=M; while(m>1){ m>>=1; L++; } }
    double *He=work,*Ho1=work+M2,*Ho3=work+M2+M4,*cw=work+M;
    rfht__rec(in,          stride*2, M2, He, p, cw);
    rfht__rec(in+stride,   stride*4, M4, Ho1,p, cw);
    rfht__rec(in+stride*3, stride*4, M4, Ho3,p, cw);
    double *Ho1r=p->rev, *Ho3r=p->rev+M4;
    Ho1r[0]=Ho1[0]; Ho3r[0]=Ho3[0];
    for(int t=1;t<M4;t++){ Ho1r[t]=Ho1[M4-t]; Ho3r[t]=Ho3[M4-t]; }
    const double *C1b=p->C1[L],*S1b=p->S1[L],*C3b=p->C3[L],*S3b=p->S3[L];
    __m256d zero=_mm256_setzero_pd();
    for(int b=0;b<4;b++){
        double *outb=out+b*M4;
        const double *Heb=He+((b&1)*M4);
        int q1=b, q3=(3*b)&3;
        int t=0,t4=M4&~3;
        for(t=0;t<t4;t+=4){
            __m256d h1 =_mm256_load_pd(Ho1+t),  h1m=_mm256_load_pd(Ho1r+t);
            __m256d h3 =_mm256_load_pd(Ho3+t),  h3m=_mm256_load_pd(Ho3r+t);
            __m256d he =_mm256_loadu_pd(Heb+t);
            __m256d c1 =_mm256_load_pd(C1b+t),  s1 =_mm256_load_pd(S1b+t);
            __m256d c3 =_mm256_load_pd(C3b+t),  s3 =_mm256_load_pd(S3b+t);
            __m256d ce1,se1,ce3,se3;
            if(q1==0){ce1=c1;se1=s1;}
            else if(q1==1){ce1=_mm256_sub_pd(zero,s1);se1=c1;}
            else if(q1==2){ce1=_mm256_sub_pd(zero,c1);se1=_mm256_sub_pd(zero,s1);}
            else{ce1=s1;se1=_mm256_sub_pd(zero,c1);}
            if(q3==0){ce3=c3;se3=s3;}
            else if(q3==1){ce3=_mm256_sub_pd(zero,s3);se3=c3;}
            else if(q3==2){ce3=_mm256_sub_pd(zero,c3);se3=_mm256_sub_pd(zero,s3);}
            else{ce3=s3;se3=_mm256_sub_pd(zero,c3);}
            __m256d acc=_mm256_fmadd_pd(ce1,h1,he);
            acc=_mm256_fmadd_pd(se1,h1m,acc);
            acc=_mm256_fmadd_pd(ce3,h3,acc);
            acc=_mm256_fmadd_pd(se3,h3m,acc);
            _mm256_storeu_pd(outb+t,acc);
        }
        for(;t<M4;t++){
            int qq; double c,s,ce1,se1,ce3,se3;
            qq=q1&3; c=C1b[t]; s=S1b[t];
            ce1=(qq==0)?c:(qq==1)?-s:(qq==2)?-c:s;
            se1=(qq==0)?s:(qq==1)?c:(qq==2)?-s:-c;
            qq=q3&3; c=C3b[t]; s=S3b[t];
            ce3=(qq==0)?c:(qq==1)?-s:(qq==2)?-c:s;
            se3=(qq==0)?s:(qq==1)?c:(qq==2)?-s:-c;
            outb[t]=Heb[t]+ce1*Ho1[t]+se1*Ho1r[t]+ce3*Ho3[t]+se3*Ho3r[t];
        }
    }
}

/* polynomial arctan2, |abs err| ~2e-4 rad over full range, multiply/add only.
   Returns angle in (-pi,pi]. Same maximizer object as the bisection phase search. */
static inline double rfht__fast_atan2(double y,double x){
    double ax=fabs(x), ay=fabs(y);
    double a=(ax>ay)? ay/(ax+1e-300) : ax/(ay+1e-300);   /* a in [0,1] */
    double s=a*a;
    double r=((-0.0464964749*s+0.15931422)*s-0.327622764)*s*a + a;  /* atan(a) */
    if(ay>ax) r=M_PI_2-r;
    if(x<0.0) r=M_PI-r;
    if(y<0.0) r=-r;
    return r;
}

/* Main entry: x[0..N-1] -> mag[0..N/2], phase[0..N/2]. */
static void rfht_polar(rfht_plan*p,const double *x,double *H_scratch,
                       double *mag,double *phase){
    int N=p->N;
    rfht__rec(x,1,N,H_scratch,p,p->arena);
    for(int k=0;k<=N/2;k++){
        int nk=(N-k)%N;
        double A=0.5*(H_scratch[k]+H_scratch[nk]);
        double B=-0.5*(H_scratch[k]-H_scratch[nk]);
        double phi=rfht__fast_atan2(B,A);
        if(phi<0.0) phi+=2.0*M_PI;
        phase[k]=phi;
        mag[k]=sqrt(A*A+B*B);
        /* MAG_NO_SQRT variant (avoids sqrt; needs cos/sin of phi, mag error ~0.5*mag*phase_err^2):
           mag[k]=A*cos(phi)+B*sin(phi);
           -- only if your no-sqrt constraint dominates; otherwise prefer the sqrt above. */
    }
}
#endif /* RFHT_POLAR_H */

/* ============================ INVERSE ============================
 * rfht_ipolar: (mag,phase)[0..N/2] -> real x[0..N-1].
 * Reconstructs Hartley spectrum then inverse-FHT (DHT self-inverse, /N).
 *
 * ACCURACY: the inverse stage is ~1e-8 relative (acc_sincos below).
 * Round-trip accuracy is limited by the FORWARD phase precision:
 *   - forward via rfht__fast_atan2 (cubic): round-trip ~3e-4 relative.
 *   - for machine-precision round-trip, store forward phase from libm atan2
 *     (or raise the arctan polynomial degree); then round-trip ~1e-8.
 * H_scratch length N. x_out length N. mag/phase length N/2+1.
 * Re-uses the SAME rfht_plan as the forward transform.
 */
#ifndef RFHT_NO_INVERSE
static inline void rfht__acc_sincos(double a,double*so,double*co){
    const double PIO2=M_PI/2;
    double k=floor(a/PIO2+0.5);
    double r=a-k*PIO2;                 /* r in [-pi/4,pi/4] */
    double r2=r*r;
    double sr=r*(1.0+r2*(-1.0/6+r2*(1.0/120+r2*(-1.0/5040+r2*(1.0/362880)))));
    double cr=1.0+r2*(-0.5+r2*(1.0/24+r2*(-1.0/720+r2*(1.0/40320))));
    int q=((int)k)&3;
    switch(q){
        case 0: *so=sr;  *co=cr;  break;
        case 1: *so=cr;  *co=-sr; break;
        case 2: *so=-sr; *co=-cr; break;
        default:*so=-cr; *co=sr;  break;
    }
}
static void rfht_ipolar(rfht_plan*p,const double*mag,const double*phase,
                        double*H_scratch,double*x_out){
    int N=p->N;
    { double s,c; rfht__acc_sincos(phase[0],&s,&c);   H_scratch[0]=mag[0]*c; }
    { int h=N/2; double s,c; rfht__acc_sincos(phase[h],&s,&c); H_scratch[h]=mag[h]*c; }
    for(int k=1;k<N/2;k++){
        int nk=N-k; double s,c; rfht__acc_sincos(phase[k],&s,&c);
        double Re=mag[k]*c, Im=mag[k]*s;
        H_scratch[k]=Re-Im;
        H_scratch[nk]=Re+Im;
    }
    rfht__rec(H_scratch,1,N,x_out,p,p->arena);
    double inv=1.0/N;
    for(int n=0;n<N;n++) x_out[n]*=inv;
}
#endif
