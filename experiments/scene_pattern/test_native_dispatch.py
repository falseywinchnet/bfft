"""Subprocess timeout catches the native nested-pool deadlock without hanging tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

class DispatchTests(unittest.TestCase):
    def test_parallel_profiles_finish_and_equal_serial(self):
        code='''import json,numpy as np
from standalone_conv_resize_demo.backend import conv_evaluate_profile_2d
rng=np.random.default_rng(62)
field=rng.normal(size=(129,161,3)).astype(np.float32)
x=rng.uniform(4,156,96).astype(np.float32);y=rng.uniform(4,124,96).astype(np.float32)
print(json.dumps(conv_evaluate_profile_2d(field,x,y).tolist()))
'''
        arrays=[]
        for count in (1,4):
            env=dict(os.environ,CONV_NATIVE_THREADS=str(count))
            p=subprocess.run([sys.executable,'-c',code],env=env,cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=45,check=True)
            arrays.append(np.array(json.loads(p.stdout)))
        np.testing.assert_array_equal(*arrays)

if __name__=='__main__':unittest.main()
