            static const double gram[5][5]=@GRAM@;
            double residual[10],gr[10]={0};
            for(int index=0;index<10;++index){
                int slot=local_begin+index,cell=slot/5,k=slot%5;
                double a=current[(cell*5+k)*lanes+lane];
                int s=slot<begin?coarse[knot-1]:coarse[knot];
                residual[index]=s*a<0?a:0;
            }
            double cost=0;
            for(int index=0;index<10;++index){
                int group=index/5;
                for(int j=0;j<5;++j)gr[index]+=gram[index%5][j]*residual[group*5+j];
                cost+=residual[index]*gr[index];
            }
            double best_cost=cost,second_cost=INFINITY;int best_boundary=begin;
            for(int candidate=begin+1;candidate<=end;++candidate){
                int index=candidate-1-local_begin;
                int slot=local_begin+index,cell=slot/5,k=slot%5;
                double a=current[(cell*5+k)*lanes+lane];
                double next=coarse[knot-1]*a<0?a:0;
                double change=next-residual[index];
                cost+=2*change*gr[index]+change*change*gram[k][k];
                int group=index/5;
                for(int j=0;j<5;++j)gr[group*5+j]+=change*gram[j][k];
                residual[index]=next;
                if(cost<best_cost){second_cost=best_cost;best_cost=cost;best_boundary=candidate;}else if(cost<second_cost)second_cost=cost;
            }
            double energy=0;
            for(int j=0;j<10;++j){int slot=local_begin+j;double a=current[slot*lanes+lane];energy+=5*a*a;}
            if(second_cost-best_cost<=256*DBL_EPSILON*energy){
                best_cost=INFINITY;best_boundary=begin;
                for(int candidate=begin;candidate<=end;++candidate){
                    double r[10],exact=0;
                    for(int j=0;j<10;++j){int slot=local_begin+j;double a=current[slot*lanes+lane];int s=slot<candidate?coarse[knot-1]:coarse[knot];r[j]=s*a<0?a:0;}
                    for(int group=0;group<2;++group)for(int i=0;i<5;++i)for(int j=0;j<5;++j)exact+=r[group*5+i]*gram[i][j]*r[group*5+j];
                    if(exact<best_cost){best_cost=exact;best_boundary=candidate;}
                }
            }
