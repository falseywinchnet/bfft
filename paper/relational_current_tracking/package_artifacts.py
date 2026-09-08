"""Build the TeX-only arXiv bundle and separate evidence/source archive."""
import json,hashlib,shutil,tarfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];OUT=REPO/'output/pdf'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'build/main.pdf',OUT/'relational_current_tracking.pdf')
    members=[ROOT/'main.tex']+sorted((ROOT/'tables').glob('*.tex'))+sorted((ROOT/'figures').glob('*.pdf'))
    path=OUT/'relational_current_tracking_arxiv.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        for p in members:tar.add(p,arcname=p.relative_to(ROOT),recursive=False)
    src=REPO/'experiments/civilian_transport'
    names=['__init__.py','relational_kalman.py','relational_reference.py','relational_markov.py','relational_native.py',
           'adaptive_acquisition.py','acquisition_study.py','optimization_study.py','cost_controls.py','data.py','geometry.py',
           'baselines.py','study.py','geometric_filter.py','relational_study.py','test_relational_markov.py',
           'test_relational_kalman.py','test_adaptive_acquisition.py','native/markov.cpp','native/cv_control.cpp','native/Makefile',
           'results/development.json','certificates.py','test_certificates.py']
    hashes={n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in names}
    evidence_members=members+[ROOT/'README.md',ROOT/'generate_assets.py']+sorted((ROOT/'evidence').glob('*.json'))
    path=OUT/'relational_current_tracking_evidence.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in evidence_members:z.write(p,arcname='manuscript/'+str(p.relative_to(ROOT)))
        z.writestr('code/experiments/__init__.py','')
        for n in names:z.write(src/n,arcname='code/experiments/civilian_transport/'+n)
        z.writestr('code/source_manifest.json',json.dumps(hashes,indent=2)+'\n')
        z.writestr('code/README.md','''# Inspection snapshot and portable reproduction\n\nPython requires NumPy and SciPy; figure generation also requires Matplotlib.\nC++17 compiler and make are needed for the optional native runtime.\n\nRun from this code directory:\n\n```sh\nmake -C experiments/civilian_transport/native libmarkov.so libcv_control.so\npython3 -m unittest experiments.civilian_transport.test_relational_markov experiments.civilian_transport.test_relational_kalman experiments.civilian_transport.test_adaptive_acquisition -v\npython3 -m experiments.civilian_transport.optimization_study --out /tmp/relational_optimization --seeds 8 --repeats 31\npython3 -m experiments.civilian_transport.cost_controls /tmp/relational_cv_control.json\n```\n\nUse OPENBLAS_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1 for the declared single-BLAS-thread timing scope. Timings vary by host. Original evidence remains under ../manuscript/evidence/. The original dense implementation is relational_reference.py. Rerunning other studies through relational_kalman.py uses the promoted exact Markov implementation. No software is installed by these source files.\n''')
    clean=ROOT/'build/clean_source';clean.mkdir(exist_ok=True)
    with tarfile.open(OUT/'relational_current_tracking_arxiv.tar.gz') as tar:
        for member in tar.getmembers():
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            tar.extract(member,clean)
    report={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.glob('relational_current_tracking*') if p.is_file()}
    (ROOT/'build/deliverables.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
