static void basin_average_worker(void*opaque,int begin,int end){
 basin_context*job=opaque;
 const double nodes[4]={-.8611363115940525752,-.3399810435848562648,.3399810435848562648,.8611363115940525752};
 const double weights[4]={.3478548451374538574,.6521451548625461426,.6521451548625461426,.3478548451374538574};
 const double step=job->m>1?(double)job->cells/(job->m-1):job->cells;
 for(int target=begin;target<end;++target){
  double center=target*step;
  double norm=job->m==1?job->cells:((target==0||target==job->m-1)?step*.5:step);
  float*dest=job->out+(size_t)target*job->lanes;memset(dest,0,(size_t)job->lanes*sizeof(float));
  for(int side=0;side<(job->m==1?1:2);++side){
   double left=job->m==1?0:fmax(0,center+(side-1)*step);
   double right=job->m==1?job->cells:fmin(job->cells,center+side*step);
   int first=(int)floor(left),last=(int)ceil(right)-1;if(last>=job->cells)last=job->cells-1;
   for(int cell=first;cell<=last;++cell){
    double l=fmax(left,cell),r=fmin(right,cell+1);if(r<=l)continue;
    double middle=.5*(l+r),half=.5*(r-l);float base=0,current_factor[5]={0};
    for(int q=0;q<4;++q){
     double x=middle+half*nodes[q];float tail[5];tail_weights(x-cell,tail);
     float factor=(float)(half*weights[q]*(job->m==1?1:1-fabs(x-center)/step)/norm);
     base+=factor;for(int k=0;k<5;++k)current_factor[k]+=factor*tail[k];
    }
    int lane=0;
#if CONV_NEON
    for(;lane+4<=job->lanes;lane+=4){
     float32x4_t inc=vmulq_n_f32(vld1q_f32(job->x+cell*job->lanes+lane),base);
     for(int k=0;k<5;++k)inc=vfmaq_n_f32(inc,vld1q_f32(job->current+(cell*5+k)*job->lanes+lane),current_factor[k]);
     vst1q_f32(dest+lane,vaddq_f32(vld1q_f32(dest+lane),inc));
    }
#endif
    for(;lane<job->lanes;++lane){float inc=base*job->x[cell*job->lanes+lane];for(int k=0;k<5;++k)inc+=current_factor[k]*job->current[(cell*5+k)*job->lanes+lane];dest[lane]+=inc;}
   }
  }
 }
}
