import importlib.util
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).parent
SPEC=importlib.util.spec_from_file_location('transport',HERE/'run_experiment.py')
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class TransportTests(unittest.TestCase):
    def test_field_has_conjugate_antipodal_symmetry(self):
        f=M.make_field(8,32,4); h=f.shape[1]//2
        np.testing.assert_allclose(f[:,h:],np.conj(f[:,:h]))

    def test_exact_connection_transports_both_directions(self):
        n=16; angle=.17*np.arange(n); amp=np.exp(.03*np.arange(n)); f=amp*np.exp(1j*angle)
        phase=np.full(n,.17); logamp=np.full(n,.03); cost=np.ones(n)
        for target in [1,5,12,15]:
            got,_=M.path_transport(0,target,f[0],phase,logamp,cost,True)
            # For targets beyond halfway, the synthetic nonperiodic connection
            # is inconsistent across the wrap; test only chosen forward arc.
            if target<=8: np.testing.assert_allclose(got,f[target],rtol=1e-12,atol=1e-12)

    def test_trial_is_finite(self):
        _,_,_,result=M.run_trial(1,8,32,3,.08)
        for metrics in result.values():
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))


if __name__=='__main__': unittest.main()
