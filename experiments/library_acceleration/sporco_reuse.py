"""Small independent upstream candidate: use the Xf already computed by xstep.

This preserves the ADMM map and all default control flow. It is exact
representation reuse, not a Krylov jump; benchmarks must label it separately.
"""
from sporco.admm.tvl2 import TVL2Deconv


class TVL2DeconvReuse(TVL2Deconv):
    def relax_AX(self):
        self.AXnr=self.cnst_A(None,self.Xf)
        if self.rlx==1.:
            self.AX=self.AXnr
        else:
            if not hasattr(self,'_cnst_c'):self._cnst_c=self.cnst_c()
            self.AX=self.rlx*self.AXnr-(1-self.rlx)*(self.cnst_B(self.Y)-self._cnst_c)
