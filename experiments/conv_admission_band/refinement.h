/* Injected after the production project_fibre definition in an isolated copy.
   MODE 1: uniform-sign certificate only; MODE 2: ordered sign-word certificate;
   MODE 3: MODE 2 plus maximal uniform-sign restoration along the proposal ray.
   DEPTH is a compile-time certificate depth, never image refinement. */
static int refined_word(const double a[5], const int8_t signs[5], int *slot,
                        int *previous, int depth) {
    if (depth) {
        double row[5],left[5],right[5];
        memcpy(row,a,sizeof(row));left[0]=row[0];right[4]=row[4];
        for(int level=1;level<5;++level){
            for(int k=0;k<5-level;++k)row[k]=0.5*(row[k]+row[k+1]);
            left[level]=row[0];right[4-level]=row[4-level];
        }
        return refined_word(left,signs,slot,previous,depth-1)
            && refined_word(right,signs,slot,previous,depth-1);
    }
    for(int k=0;k<5;++k){
        int s=(a[k]>0)-(a[k]<0);
        if(!s || s==*previous)continue;
        if(*previous)++*slot;
        while(*slot<5 && signs[*slot]!=s)++*slot;
        if(*slot==5)return 0;
        *previous=s;
    }
    return 1;
}

static double uniform_ray(const double b[5],const double a[5],int sign,int depth){
    if(depth){
        double br[5],ar[5],bl[5],al[5],bt[5],at[5];
        memcpy(bt,b,sizeof(bt));memcpy(at,a,sizeof(at));
        bl[0]=bt[0];br[4]=bt[4];al[0]=at[0];ar[4]=at[4];
        for(int level=1;level<5;++level){
            for(int k=0;k<5-level;++k){bt[k]=0.5*(bt[k]+bt[k+1]);at[k]=0.5*(at[k]+at[k+1]);}
            bl[level]=bt[0];br[4-level]=bt[4-level];al[level]=at[0];ar[4-level]=at[4-level];
        }
        return fmin(uniform_ray(bl,al,sign,depth-1),uniform_ray(br,ar,sign,depth-1));
    }
    double t=1;
    for(int k=0;k<5;++k){
        double margin=sign*b[k],direction=sign*(a[k]-b[k]);
        if(margin<0)return 0;
        if(direction<0)t=fmin(t,margin/-direction);
    }
    return t;
}

static void refined_fibre(const float a[5],const int8_t signs[5],float delta,float c[5]){
    int uniform=1;for(int k=1;k<5;++k)uniform&=signs[k]==signs[0];
    if(ADMISSION_MODE==1 && !uniform){project_fibre(a,signs,delta,c);return;}
    /* Repair the FIR's floating mass discrepancy before certification. The
       central coordinate is reversal invariant. It is no model parameter. */
    double proposal[5];for(int k=0;k<5;++k)proposal[k]=a[k];
    proposal[2]=(double)delta-proposal[0]-proposal[1]-proposal[3]-proposal[4];
    int slot=0,previous=0;
    if(refined_word(proposal,signs,&slot,&previous,ADMISSION_DEPTH)){
        for(int k=0;k<5;++k)c[k]=(float)proposal[k];return;
    }
    project_fibre(a,signs,delta,c);
    if(ADMISSION_MODE==3 && uniform && delta!=0){
        double base[5];for(int k=0;k<5;++k)base[k]=c[k];
        double t=uniform_ray(base,proposal,signs[0],ADMISSION_DEPTH);
        for(int k=0;k<5;++k)c[k]=(float)(base[k]+t*(proposal[k]-base[k]));
    }
}

API void conv_audit_fibre(const float*a,const int8_t*s,float delta,float*c){
#if ADMISSION_MODE > 0
    refined_fibre(a,s,delta,c);
#else
    project_fibre(a,s,delta,c);
#endif
}
