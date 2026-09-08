"""Verify native dynamic counters against independent closed-form schedules."""
import json
import sys


def fft_counts(n):
    l=n.bit_length()-1
    return {"butterflies":n*l//2,
            "bit_reversal_swaps":(n-2**((l+1)//2))//2,
            "complex_multiplies":n*l//2-3*n//2+2 if n>=2 else 0,
            "quarter_turns":n//2-1 if n>=2 else 0}


def expected(r,method):
    n=4**r
    a=2**(r//2);b=2**((r+1)//2)
    if method in ("dit","dif","stockham"):
        c=fft_counts(n)
        if method=="stockham":c["bit_reversal_swaps"]=0
        passes=0 if method=="stockham" else 1
    else:
        ca,cb=fft_counts(a),fft_counts(b)
        bfactor=6 if method=="packet" else 2
        c={k:2*(n//a)*ca[k]+bfactor*(n//b)*cb[k] for k in ca}
        c["complex_multiplies"]+=3*n
        if method=="fused":c["bit_reversal_swaps"]=0
        passes=4 if method=="fused" else 5
    c["reads"]=c["writes"]=2*c["butterflies"]+2*c["bit_reversal_swaps"]+passes*n
    c["real_scalings"]=2*n if method=="packet" else 0
    return c


if __name__=="__main__":
    report=json.load(open(sys.argv[1]))
    for row in report["results"]:
        target=expected(row["r"],row["method"])
        for k,v in target.items():
            assert row[k]==v,(row["N"],row["method"],k,row[k],v)
        if report.get("backend")=="neon_complex_pair":
            assert row["intrinsic_lane_swaps"]==target["complex_multiplies"]+target["quarter_turns"]
            assert row["coefficient_vector_reads"]==target["complex_multiplies"]
            assert row["coefficient_lane_operands"]==2*target["complex_multiplies"]
            assert row["boundary_layout_conversions"]==0
    print("Verified all seven dynamic counters for",len(report["results"]),"size/method cases")
