#include <algorithm>
extern "C" void cv_update(double*m,double*p,double dt,const double*y,double variance){
 double a=p[0]+2*dt*p[1]+dt*dt*p[3]+dt*dt*dt/3;
 double b=p[1]+dt*p[3]+dt*dt/2,c=p[3]+dt,s=a+variance;
 for(int d=0;d<3;d++){m[d]+=dt*m[3+d];double r=y[d]-m[d];m[d]+=a/s*r;m[3+d]+=b/s*r;}
 p[0]=a-a*a/s;p[1]=p[2]=b-a*b/s;p[3]=c-b*b/s;
}
extern "C" void cv_query(const double*m,const double*p,const double*dt,int n,double*mean,double*cov){
 std::fill(cov,cov+n*9,0);
 for(int t=0;t<n;t++)for(int d=0;d<3;d++){
  double h=dt[t];mean[t*3+d]=m[d]+h*m[3+d];
  cov[t*9+d*3+d]=p[0]+2*h*p[1]+h*h*p[3]+h*h*h/3;
 }
}
