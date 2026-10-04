/* Fixed scalar RGBA evaluation cost only; no browser or complete warp timing.
 * Build with clang -O3 retention_cost.c -o /tmp/conv-retention-cost -lm.
 * Matching float control storage, double accumulation and shared lattices.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static volatile double receipt;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
#define KERNEL(D) \
static double kernel##D(const float *p,int cells){ \
 int stride=(D)*cells+1;double sum=0; \
 for(int y=0;y<512;y++)for(int x=0;x<512;x++){ \
  double sx=cells*(x+.375)/512,sy=cells*(y+.625)/512; \
  int cx=(int)sx,cy=(int)sy;double u=sx-cx,v=sy-cy; \
  double bx[(D)+1]={1},by[(D)+1]={1},out[4]={0}; \
  for(int k=1;k<=(D);k++){ \
   bx[k]=u*bx[k-1];by[k]=v*by[k-1]; \
   for(int j=k-1;j>0;j--){bx[j]=(1-u)*bx[j]+u*bx[j-1];by[j]=(1-v)*by[j]+v*by[j-1];} \
   bx[0]*=1-u;by[0]*=1-v; \
  } \
  for(int j=0;j<=(D);j++)for(int i=0;i<=(D);i++){ \
   const float *q=p+4*((cy*(D)+j)*stride+cx*(D)+i);double w=bx[i]*by[j]; \
   for(int c=0;c<4;c++)out[c]+=w*q[c]; \
  } \
  sum+=out[0]+out[1]+out[2]+out[3]; \
 }return sum;}
KERNEL(5)
KERNEL(6)
static int cmp(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(void){
 for(int cells=16;cells<=512;cells=(cells==16?128:cells*4)){
  float *p[2];double t[2][9];
  for(int m=0;m<2;m++){int d=5+m,stride=d*cells+1;size_t count=(size_t)stride*stride*4;
   p[m]=malloc(count*sizeof(float));if(!p[m])return 2;
   for(size_t i=0;i<count;i++)p[m][i]=(float)((i*17)%1024)/1024;
  }
  receipt=kernel5(p[0],cells)+kernel6(p[1],cells);
  for(int r=0;r<9;r++)for(int k=0;k<2;k++){int m=(k+r)%2;double start=now();
   receipt=m?kernel6(p[m],cells):kernel5(p[m],cells);t[m][r]=(now()-start)*1000;}
  for(int m=0;m<2;m++)qsort(t[m],9,sizeof(double),cmp);
  printf("{\"sourceSide\":%d,\"outputPixels\":262144,\"repeats\":9,\"quinticMs\":%.6f,\"degree6Ms\":%.6f,\"ratio\":%.6f,\"quinticMinMs\":%.6f,\"quinticMaxMs\":%.6f,\"degree6MinMs\":%.6f,\"degree6MaxMs\":%.6f}\n",cells+1,t[0][4],t[1][4],t[1][4]/t[0][4],t[0][0],t[0][8],t[1][0],t[1][8]);
  free(p[0]);free(p[1]);
 }return receipt<0;
}
