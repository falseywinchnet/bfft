// Move all packet-specific phase to its coefficient boundary. The interior
// is a positive complex DIF FFT in the SAME two real rows. Its bit reversal
// and negative-frequency fold are absorbed in the standard comb emission.
void unfolded_packet(double* RESTRICT v,int w,int d,int e,complex_t* RESTRICT out) const {
    double* RESTRICT a=v;double* RESTRICT b=v+w;
    for(int j=1;j<w;++j) {
        const int k=j*d;const double c=cos_[k],s=sin_[k],ar=a[j],bi=b[j];
        a[j]=c*ar-s*bi;b[j]=s*ar+c*bi;
    }
    for(int len=w;len>=2;len/=2) {
        const int h=len/2,step=n_/len;
        for(int off=0;off<w;off+=len) {
            for(int j=0;j<h;++j) {
                const double ar=a[off+j],ai=b[off+j],br=a[off+j+h],bi=b[off+j+h];
                const double dr=ar-br,di=ai-bi,c=cos_[j*step],s=sin_[j*step];
                a[off+j]=ar+br;b[off+j]=ai+bi;
                a[off+j+h]=c*dr-s*di;b[off+j+h]=s*dr+c*di;
            }
        }
    }
    // One output move per bin. No intermediate bit-reversal or folding pass.
    const int bits=bfft_ctz_u32(static_cast<unsigned>(w));
    for(int r=0;r<w;++r) {
        const int m=(r+1)/2;
        unsigned k=(r&1)?w-m:m,p=0;
        for(int j=0;j<bits;++j){p=(p<<1)|(k&1);k>>=1;}
        const int bin=rank_bin(r,d,e);
        out[bin].re=a[p];out[bin].im=(r&1)?b[p]:-b[p];
    }
}
