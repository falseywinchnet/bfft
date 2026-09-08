typedef struct {const float*x;int n,lanes;float*raw;} raw_context;
static void raw_currents_worker(void*opaque,int begin,int end){
 raw_context*j=opaque;int lanes=j->lanes;
 for(int cell=begin;cell<end;++cell){
  if(cell>=@R@ && cell<j->n-1-@R@){
   int lane=0;
#if CONV_NEON
   for(;lane+4<=lanes;lane+=4){
    float32x4_t acc[@D@];for(int k=0;k<@D@;++k)acc[k]=vdupq_n_f32(0);
    for(int t=0;t<@T@;++t){
     float32x4_t x=vld1q_f32(j->x+(cell-@R@+t)*lanes+lane);
     for(int k=0;k<@D@;++k)acc[k]=vfmaq_n_f32(acc[k],x,BANK[k][t]);
    }
    for(int k=0;k<@D@;++k)vst1q_f32(j->raw+(cell*@D@+k)*lanes+lane,acc[k]);
   }
#endif
   for(;lane<lanes;++lane)for(int k=0;k<@D@;++k){
    float sum=0;for(int t=0;t<@T@;++t)sum=fmaf(BANK[k][t],j->x[(cell-@R@+t)*lanes+lane],sum);
    j->raw[(cell*@D@+k)*lanes+lane]=sum;
   }
   continue;
  }
  int start=cell-@R@,reverse=0;const float(*bank)[@T@]=BANK;
  if(cell<@R@){start=0;bank=LEFT[cell];}
  else if(cell>=j->n-1-@R@){start=j->n-@T@;reverse=1;bank=LEFT[j->n-2-cell];}
  int lane=0;
#if CONV_NEON
  for(;lane+4<=lanes;lane+=4){
   float32x4_t acc[@D@];for(int k=0;k<@D@;++k)acc[k]=vdupq_n_f32(0);
   for(int t=0;t<@T@;++t){
    float32x4_t x=vld1q_f32(j->x+(start+t)*lanes+lane);
    for(int k=0;k<@D@;++k)acc[k]=vfmaq_n_f32(acc[k],x,reverse?-bank[@D@-1-k][@T@-1-t]:bank[k][t]);
   }
   for(int k=0;k<@D@;++k)vst1q_f32(j->raw+(cell*@D@+k)*lanes+lane,acc[k]);
  }
#endif
  for(;lane<lanes;++lane)for(int k=0;k<@D@;++k){
   float sum=0;for(int t=0;t<@T@;++t)sum=fmaf(reverse?-bank[@D@-1-k][@T@-1-t]:bank[k][t],j->x[(start+t)*lanes+lane],sum);
   j->raw[(cell*@D@+k)*lanes+lane]=sum;
  }
 }
}
static void raw_currents(const float*x,int n,int lanes,float*c){raw_context j={x,n,lanes,c};parallel_for(n-1,24,raw_currents_worker,&j);}
