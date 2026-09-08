// Included inside the experimental class. A four-child phase-gauged DIP cell.
// theta=pi*d/e. Pre-rotate z1,z2,z3 by theta/2,theta,3theta/2;
// apply a fixed positive-sign DFT4; fold into [Y0,conj(Y2),conj(Y3),Y1].
// This is the identical packet factorization with 3 rather than 4 rotations.
static BRUUN_ALWAYS_INLINE void cell4_gauge(double* RESTRICT v,int q,
    double c1,double s1,double c2,double s2,double c3,double s3) {
    int i=0;
#if BRUUN_LEVEL >= 1
    const auto C1=V2_SET1(c1),S1=V2_SET1(s1),C2=V2_SET1(c2),S2=V2_SET1(s2),C3=V2_SET1(c3),S3=V2_SET1(s3);
    for(;i+1<q;i+=2) {
        const auto a=V2_LD(v+i),b=V2_LD(v+4*q+i);
        const auto a1=V2_LD(v+q+i),b1=V2_LD(v+5*q+i);
        const auto a2=V2_LD(v+2*q+i),b2=V2_LD(v+6*q+i);
        const auto a3=V2_LD(v+3*q+i),b3=V2_LD(v+7*q+i);
        const auto r1=V2_MSUB(V2_MUL(C1,a1),S1,b1),t1=V2_MADD(V2_MUL(S1,a1),C1,b1);
        const auto r2=V2_MSUB(V2_MUL(C2,a2),S2,b2),t2=V2_MADD(V2_MUL(S2,a2),C2,b2);
        const auto r3=V2_MSUB(V2_MUL(C3,a3),S3,b3),t3=V2_MADD(V2_MUL(S3,a3),C3,b3);
        const auto ar=V2_ADD(a,r2),ai=V2_ADD(b,t2),br=V2_SUB(a,r2),bi=V2_SUB(b,t2);
        const auto cr=V2_ADD(r1,r3),ci=V2_ADD(t1,t3),dr=V2_SUB(r1,r3),di=V2_SUB(t1,t3);
        V2_ST(v+i,V2_ADD(ar,cr)); V2_ST(v+q+i,V2_ADD(ai,ci));
        V2_ST(v+2*q+i,V2_SUB(ar,cr)); V2_ST(v+3*q+i,V2_SUB(ci,ai));
        V2_ST(v+4*q+i,V2_ADD(br,di)); V2_ST(v+5*q+i,V2_SUB(dr,bi));
        V2_ST(v+6*q+i,V2_SUB(br,di)); V2_ST(v+7*q+i,V2_ADD(bi,dr));
    }
#endif
    for(;i<q;++i) {
        double a=v[i],b=v[4*q+i],a1=v[q+i],b1=v[5*q+i],a2=v[2*q+i],b2=v[6*q+i],a3=v[3*q+i],b3=v[7*q+i];
        double r1=c1*a1-s1*b1,t1=s1*a1+c1*b1,r2=c2*a2-s2*b2,t2=s2*a2+c2*b2,r3=c3*a3-s3*b3,t3=s3*a3+c3*b3;
        double ar=a+r2,ai=b+t2,br=a-r2,bi=b-t2,cr=r1+r3,ci=t1+t3,dr=r1-r3,di=t1-t3;
        v[i]=ar+cr;v[q+i]=ai+ci;v[2*q+i]=ar-cr;v[3*q+i]=ci-ai;
        v[4*q+i]=br+di;v[5*q+i]=dr-bi;v[6*q+i]=br-di;v[7*q+i]=bi+dr;
    }
}
BRUUN_ALWAYS_INLINE void cell4_dispatch(double* RESTRICT v,int q,double c0,double s0,double cl,double sl,double ch,double sh) const {
    if constexpr (Mode == 5 || Mode == 6 || Mode == 8) {
        if constexpr (Mode == 8) { if(q<8) { cell4_fwd_ip(v,q,c0,s0,cl,sl,ch,sh); return; } }
        // cos(3phi),sin(3phi) from theta=2phi and phi. Exact trig identities;
        // coefficients are computed once per span, outside the column loop.
        cell4_gauge(v,q,cl,sl,c0,s0,c0*cl-s0*sl,s0*cl+c0*sl);
    } else cell4_fwd_ip(v,q,c0,s0,cl,sl,ch,sh);
}
