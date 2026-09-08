static void primitive_tails(double u,double p[5]){
 if(u==0){for(int k=0;k<5;++k)p[k]=0;return;}
 if(u==1){for(int k=0;k<5;++k)p[k]=(5.0-k)/6.0;return;}
 double v=1-u;
 double b[7]={v*v*v*v*v*v,6*u*v*v*v*v*v,15*u*u*v*v*v*v,20*u*u*u*v*v*v,
              15*u*u*u*u*v*v,6*u*u*u*u*u*v,u*u*u*u*u*u};
 for(int k=0;k<5;++k){double sum=0;for(int l=k+2;l<=6;++l)sum+=(l-k-1)*b[l];p[k]=sum/6.0;}
}
