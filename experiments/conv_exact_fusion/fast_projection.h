/* The projection onto the mass hyperplane is the signed-fibre optimum
   whenever it satisfies all sign constraints. For uniform fibres, certify
   the same binary32 rounding as the original simplex implementation.
   A non-decisive certificate uses the original implementation verbatim. */
static void project_fibre(const float a[5],const int8_t sign[5],float delta,float c[5]){
    int uniform=sign[0]!=0;
    for(int k=1;k<5;++k)uniform&=sign[k]==sign[0];
    if(uniform && sign[0]*delta>0){
        double sum=0,magnitude=fabs((double)delta);
        for(int k=0;k<5;++k){sum+=a[k];magnitude+=fabs((double)a[k]);}
        const double lambda=(sum-(double)delta)/5.0;
        const double error=64.0*DBL_EPSILON*magnitude;
        int stable=1;
        for(int k=0;k<5;++k){
            double value=(double)a[k]-lambda;
            const float low=(float)(value-error),high=(float)(value+error);
            if(sign[k]*value<=error || low!=high){stable=0;break;}
            c[k]=(float)value;
        }
        if(stable){
            double mass=0;int best=0;
            for(int k=0;k<5;++k){mass+=c[k];if(fabsf(c[k])>fabsf(c[best]))best=k;}
            c[best]+=(float)((double)delta-mass);
            return;
        }
    }
    project_fibre_original(a,sign,delta,c);
}
