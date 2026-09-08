#include <cmath>
#include <algorithm>
// Arrays are branch-major, row-major. No fast-math or reduced precision.
static void advance(double* m,double* p,const double* fs,const double* qs,int K,int branch,int n){
 const double* f=fs+(n*K+branch)*16; const double* q=qs+(n*K+branch)*16;
 double mm[12]={}, tmp[16]={}, pp[16]={};
 for(int i=0;i<4;i++)for(int j=0;j<4;j++){
  for(int d=0;d<3;d++)mm[i*3+d]+=f[i*4+j]*m[j*3+d];
  for(int d=0;d<4;d++)tmp[i*4+d]+=f[i*4+j]*p[j*4+d];
 }
 for(int i=0;i<4;i++)for(int j=0;j<4;j++){
  pp[i*4+j]=q[i*4+j];for(int d=0;d<4;d++)pp[i*4+j]+=tmp[i*4+d]*f[j*4+d];
 }
 std::copy(mm,mm+12,m);std::copy(pp,pp+16,p);
}
static void transport(double*m,double*p,const double*fs,const double*qs,int K,int branch,int start,int target,int split,double h){
 if(start<split && split<=target){
  advance(m,p,fs,qs,K,branch,split-1-start);
  double b[4]={1,h,h,0},mm[3]={},v=0;
  for(int i=0;i<4;i++){
   for(int d=0;d<3;d++)mm[d]+=b[i]*m[3*i+d];
   for(int j=0;j<4;j++)v+=b[i]*p[4*i+j]*b[j];
  }
  std::fill(m,m+12,0);std::copy(mm,mm+3,m);std::fill(p,p+16,0);
  p[0]=v;p[5]=p[10]=p[15]=4;start=split;
 }
 if(target>start)advance(m,p,fs,qs,K,branch,target-start);
}
static double weights(const double*logs,double*w,int K){
 double mx=*std::max_element(logs,logs+K),z=0;
 for(int k=0;k<K;k++){w[k]=std::exp(logs[k]-mx);z+=w[k];}
 for(int k=0;k<K;k++)w[k]/=z;
 return mx+std::log(z);
}
extern "C" double markov_update(double*m,double*p,double*logs,double*w,const double*fs,const double*qs,
 int K,int start,int target,int split,double h,double tau,const double*y,double variance){
 const double b[4]={1,tau,tau,0};
 for(int k=0;k<K;k++){
  double*mk=m+12*k;double*pk=p+16*k;
  transport(mk,pk,fs,qs,K,k,start,target,split,h);
  double pb[4]={},r[3],s=variance,rr=0;
  for(int i=0;i<4;i++)for(int j=0;j<4;j++)pb[i]+=pk[i*4+j]*b[j];
  for(int i=0;i<4;i++)s+=b[i]*pb[i];
  for(int d=0;d<3;d++){r[d]=y[d];for(int i=0;i<4;i++)r[d]-=b[i]*mk[i*3+d];rr+=r[d]*r[d];}
  logs[k]-=.5*(3*std::log(6.283185307179586476925286766559*s)+rr/s);
  for(int i=0;i<4;i++){
   for(int d=0;d<3;d++)mk[i*3+d]+=pb[i]/s*r[d];
   for(int j=i;j<4;j++){double v=.5*(pk[i*4+j]+pk[j*4+i])-pb[i]*pb[j]/s;pk[i*4+j]=pk[j*4+i]=v;}
  }
 }
 return weights(logs,w,K);
}
extern "C" void markov_query(const double*m,const double*p,const double*w,const double*fs,const double*qs,
 int K,int start,int split,int cells,double h,const double*times,int T,const double*anchor,
 double*means,double*vars,double*mean,double*cov){
 std::fill(mean,mean+3*T,0);std::fill(cov,cov+9*T,0);
 for(int t=0;t<T;t++){
  int target=std::min(int(times[t]/h),cells-1);double tau=times[t]-target*h,b[4]={1,tau,tau,0};
  for(int k=0;k<K;k++){
   double mm[12],pp[16];std::copy(m+k*12,m+(k+1)*12,mm);std::copy(p+k*16,p+(k+1)*16,pp);
   transport(mm,pp,fs,qs,K,k,start,target,split,h);
   double v=0;for(int i=0;i<4;i++)for(int j=0;j<4;j++)v+=b[i]*pp[i*4+j]*b[j];vars[k*T+t]=v;
   for(int d=0;d<3;d++){
    double z=anchor[d];for(int i=0;i<4;i++)z+=b[i]*mm[i*3+d];
    means[(k*T+t)*3+d]=z;mean[t*3+d]+=w[k]*z;cov[t*9+d*3+d]+=w[k]*v;
   }
  }
  for(int k=0;k<K;k++)for(int i=0;i<3;i++)for(int j=0;j<3;j++)
   cov[t*9+i*3+j]+=w[k]*(means[(k*T+t)*3+i]-mean[t*3+i])*(means[(k*T+t)*3+j]-mean[t*3+j]);
 }
}
