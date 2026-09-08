    /* The full active face needs no sorting. Within this binary32 exponent
       band every five-term binary64 sum is exact, so theta is identical to
       the legacy descending-prefix theta, including its final division. */
    double sum_all=0.0;
    float minimum=a[0],maximum=a[0];
    for(int k=0;k<5;++k){
        sum_all+=(double)a[k];
        minimum=fminf(minimum,a[k]);maximum=fmaxf(maximum,a[k]);
    }
    if(minimum>0.0f && (double)maximum<=(double)minimum*16777216.0){
        const double theta_all=(sum_all-(double)total)/5.0;
        if((double)minimum>theta_all){
            double mass_all=0.0;int best_all=0;
            for(int k=0;k<5;++k){
                c[k]=(float)((double)a[k]-theta_all);
                mass_all+=c[k];
                if(c[k]>c[best_all])best_all=k;
            }
            c[best_all]+=(float)((double)total-mass_all);
            return;
        }
    }
