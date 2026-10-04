/* rfht_neon_example.c - Apple Silicon usage
 * Build (Apple Silicon, macOS):
 *   clang -O3 -mcpu=native rfht_neon_example.c -o example -lm
 *   (or -mcpu=apple-m1 / apple-m2 / apple-m3 / apple-m4)
 * The NEON combine is correctness-validated (vs direct DFT and round-trip);
 * timing on real M-series silicon is yours to measure.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "rfht_neon.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9; }

int main(void){
    int N=16384;
    double *x,*H,*xr,*mag,*ph;
    posix_memalign((void**)&x,64,sizeof(double)*N);
    posix_memalign((void**)&H,64,sizeof(double)*N);
    posix_memalign((void**)&xr,64,sizeof(double)*N);
    mag=malloc(sizeof(double)*(N/2+1)); ph=malloc(sizeof(double)*(N/2+1));

    srand(1);
    for(int n=0;n<N;n++)
        x[n]=1.0*sin(2*M_PI*440.0*n/N)+0.5*sin(2*M_PI*1234.0*n/N+0.7)
            +0.01*(2.0*rand()/RAND_MAX-1.0);

    rfht_plan*p=rfht_plan_create(N);

    /* forward */
    rfht_polar(p,x,H,mag,ph);
    /* inverse (overwrites H scratch) */
    rfht_ipolar(p,mag,ph,H,xr);

    double e=0,rms=0; for(int n=0;n<N;n++){double d=fabs(xr[n]-x[n]);if(d>e)e=d;rms+=x[n]*x[n];}
    printf("round-trip rel err = %.3e\n", e/sqrt(rms/N));

    /* simple timing */
    int reps=500; double t0=now();
    for(int r=0;r<reps;r++) rfht_polar(p,x,H,mag,ph);
    printf("forward: %.4f ms/transform (N=%d)\n",(now()-t0)/reps*1e3,N);

    rfht_plan_destroy(p);
    free(x);free(H);free(xr);free(mag);free(ph);
    return 0;
}
