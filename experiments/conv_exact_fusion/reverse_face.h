    /* The legacy scan keeps the LAST successful rank. Check those same
       predicates in reverse order, using identically accumulated prefixes. */
    const double p1=(double)ordered[4];
    const double p2=p1+(double)ordered[3];
    const double p3=p2+(double)ordered[2];
    const double p4=p3+(double)ordered[1];
    const double p5=p4+(double)ordered[0];
    double theta=(p5-(double)total)/5.0;
    if(!((double)ordered[0]>theta)){
        theta=(p4-(double)total)/4.0;
        if(!((double)ordered[1]>theta)){
            theta=(p3-(double)total)/3.0;
            if(!((double)ordered[2]>theta)){
                theta=(p2-(double)total)/2.0;
                if(!((double)ordered[3]>theta)){
                    theta=p1-(double)total;
                    if(!((double)ordered[4]>theta))theta=0.0;
                }
            }
        }
    }
