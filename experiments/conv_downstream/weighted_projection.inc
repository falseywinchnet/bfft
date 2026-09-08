static const double METRIC[5]=@WEIGHTS@;
static const double INVERSE_METRIC[5]=@INVERSE@;
static void weighted_swap(double*b,int*idx,int i,int j){
 if(b[i]>b[j]){double x=b[i];b[i]=b[j];b[j]=x;int k=idx[i];idx[i]=idx[j];idx[j]=k;}
}
static void weighted_sort(double*b,int*idx){
 weighted_swap(b,idx,0,1);weighted_swap(b,idx,3,4);weighted_swap(b,idx,2,4);
 weighted_swap(b,idx,2,3);weighted_swap(b,idx,1,4);weighted_swap(b,idx,0,3);
 weighted_swap(b,idx,0,2);weighted_swap(b,idx,1,3);weighted_swap(b,idx,1,2);
}
static void project_fibre(const float a[5],const int8_t sign[5],float delta,float c[5]){
 int uniform=sign[0]!=0;for(int k=1;k<5;++k)uniform&=sign[k]==sign[0];
 if(uniform && sign[0]*delta<=0){for(int k=0;k<5;++k)c[k]=0;return;}
 double b[5];int idx[5]={0,1,2,3,4};
 for(int k=0;k<5;++k)b[k]=METRIC[k]*a[k]*(uniform?sign[0]:1);
 weighted_sort(b,idx);
 double lambda=0;int found=0;
 if(uniform){
  double sum=0,inverse=0;
  for(int rank=1;rank<=5;++rank){
   int k=idx[5-rank];sum+=sign[0]*(double)a[k];inverse+=INVERSE_METRIC[k];
   double candidate=(sum-sign[0]*(double)delta)/inverse;
   if(b[5-rank]>candidate){lambda=sign[0]*candidate;found=1;}
  }
 }else{
  for(int region=0;region<=5 && !found;++region){
   double low=region==0?-INFINITY:b[region-1],high=region==5?INFINITY:b[region];
   double probe=!isfinite(low)?high-fmax(1,fabs(high)):!isfinite(high)?low+fmax(1,fabs(low)):.5*(low+high);
   double sum=0,inverse=0;
   for(int k=0;k<5;++k)if(sign[k]*(METRIC[k]*a[k]-probe)>0){sum+=a[k];inverse+=INVERSE_METRIC[k];}
   if(inverse==0)continue;
   double value=(sum-delta)/inverse;
   double tol=64*DBL_EPSILON*fmax(1e-300,fmax(fabs(value),fmax(isfinite(low)?fabs(low):0,isfinite(high)?fabs(high):0)));
   if(value>=low-tol && value<=high+tol){lambda=value;found=1;}
  }
 }
 if(!found && delta==0){
  double lower=-INFINITY,upper=INFINITY;
  for(int k=0;k<5;++k){if(sign[k]>0)lower=fmax(lower,METRIC[k]*a[k]);else upper=fmin(upper,METRIC[k]*a[k]);}
  if(lower<=upper){for(int k=0;k<5;++k)c[k]=0;return;}
 }
 if(!found){for(int k=0;k<5;++k)c[k]=NAN;return;}
 double mass=0;int best=0;
 for(int k=0;k<5;++k){
  double value=a[k]-lambda*INVERSE_METRIC[k];
  c[k]=(float)(sign[k]>0?fmax(0,value):fmin(0,value));
  mass+=c[k];if(fabsf(c[k])>fabsf(c[best]))best=k;
 }
 c[best]+=(float)((double)delta-mass);
}
