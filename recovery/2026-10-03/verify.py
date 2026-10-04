"""Focused recovery checks; no study sweeps, engine benchmarks, or installation.

Run through m4build from this recovery checkout. Build/output directories are
isolated from every active research task. Timing below is test bookkeeping,
never a performance result. Existing optional dependencies are read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path('/tmp/bfft-recovery-20261003')
CM = '/opt/homebrew/bin/cmake'
NODE = '/opt/homebrew/bin/node'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('native', 'python', 'javascript'), required=True)
    parser.add_argument('--python', default=sys.executable,
                        help='Existing interpreter; miniforge3 includes optional Matplotlib on the Mini')
    args = parser.parse_args()
    os.chdir(ROOT)
    BUILD.mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               OMP_NUM_THREADS='1', CONV_NATIVE_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    dependency_root = Path('/Users/joshuahkuttenkuler/Developer/CodexBuilds/bfft-daa90db3ea8e/tmp/library_sources')
    env['PYTHONPATH'] = ':'.join(map(str, [ROOT, ROOT/'standalone_conv_resize_demo',
                                       dependency_root/'deps', dependency_root/'sporco-0.2.2.post1']))
    env['BFFT_LIBRARY'] = str(BUILD/'bfft/libbfft.dylib')
    # Restore only compact public receipt inputs needed by the historical checks.
    # Their original output/ paths are ignored by Git; authoritative stored bytes
    # remain under recovery/evidence, and no existing runtime file is replaced.
    for row in json.loads(Path(__file__).with_name('manifest.json').read_text()):
        if row['source_path'] == 'output/support_geometry/conv_line_coverage/results.json':
            source = ROOT/row['destination']
            target = ROOT/row['source_path']
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Invalid retained line-coverage input')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    tasks = []
    if args.phase == 'native':
        tasks += [('bfft-configure', [CM, '-S', '.', '-B', str(BUILD/'bfft'),
                  '-DCMAKE_BUILD_TYPE=Release', '-DBFFT_BUILD_EXAMPLES=OFF',
                  '-DBFFT_BUILD_TESTS=OFF', '-DBFFT_BUILD_PROBES=OFF', '-DBFFT_BUILD_HIGH_VISION=OFF']),
                  ('bfft-build', [CM, '--build', str(BUILD/'bfft'), '--target', 'bfft_shared', '-j1'])]
        for label, path, flags in [
                ('tape', 'experiments/tape_dynamics', []),
                ('obligation', 'experiments/obligation_dynamics', []),
                ('rvfx', 'realtime_vector_fx', ['-DRVFX_BUILD_OBS=OFF']),
                ('mosaic', 'mosaic_fx', ['-DMOSAIC_WITH_BFFT=OFF', '-DMOSAIC_BUILD_OBS=OFF'])]:
            tasks += [(label+'-configure', [CM, '-S', path, '-B', str(BUILD/label),
                       '-DCMAKE_BUILD_TYPE=Release'] + flags),
                      (label+'-build', [CM, '--build', str(BUILD/label), '-j1']),
                      (label+'-tests', ['/opt/homebrew/bin/ctest', '--test-dir', str(BUILD/label), '--output-on-failure'])]
    elif args.phase == 'python':
        roots = ['experiments/conv_dual_pair', 'experiments/conv_fast_aa', 'experiments/conv_warp',
                 'experiments/entropic_transport_closure', 'experiments/krylov_bregman',
                 'experiments/krylov_em', 'experiments/tropical_transport',
                 'experiments/library_acceleration', 'experiments/scene_pattern']
        for root in roots:
            modules = sorted(str(p.with_suffix('')).replace('/', '.') for p in Path(root).rglob('test_*.py'))
            if root == 'experiments/conv_warp':
                optional = [m for m in modules if m.endswith(('.test_paper_images', '.test_range'))]
                modules = [m for m in modules if m not in optional]
                plotting_python = '/Users/joshuahkuttenkuler/miniforge3/bin/python'
                tasks.append(('conv-warp-optional', [plotting_python, '-m', 'unittest', *optional, '-v']))
            if modules:
                tasks.append((Path(root).name, [args.python, '-m', 'unittest', *modules, '-v']))
        tasks.append(('conv-backend', [args.python, '-m', 'unittest',
                                     'standalone_conv_resize_demo.test_backend', '-v']))
    else:
        paths = ['experiments/wrench_transport/test_collide.mjs',
                 'experiments/wrench_transport/test_basic.mjs',
                 'experiments/wrench_transport/test_features.mjs',
                 'experiments/wrench_transport/packing/test_score.mjs',
                 'experiments/obligation_dynamics/scoring/test_score.mjs',
                 'experiments/tape_dynamics/scoring/test_score.mjs',
                 'experiments/conv_warp/line_coverage/check_profile.mjs']
        tasks = [(str(Path(p).with_suffix('')).replace('/', '-'), [NODE, p]) for p in paths if Path(p).exists()]
    receipt = BUILD/(args.phase+'.json')
    records = [r for r in json.loads(receipt.read_text()) if r.get('exit_code') == 0] if receipt.exists() else []
    completed = {tuple(r['command']) for r in records}
    for label, command in tasks:
        if tuple(command) in completed:
            continue
        # Do not compete with the parent's timed rigid-body runs.
        processes = subprocess.check_output(['ps', '-axo', 'comm'], text=True)
        busy = any(Path(x.strip()).name in ('tape_bench', 'phys_bench', 'obligation_bench')
                   for x in processes.splitlines())
        if busy:
            print('DEFERRED: active physics benchmark; rerun this phase later.', flush=True)
            records.append({'label': label, 'status': 'deferred', 'reason': 'active physics benchmark'})
            break
        start = time.time()
        log = BUILD/(args.phase+'-'+label+'.txt')
        try:
            with log.open('w') as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
            record = {'label': label, 'command': command, 'exit_code': result.returncode,
                      'seconds': time.time()-start, 'log': str(log)}
        except subprocess.TimeoutExpired:
            record = {'label': label, 'command': command, 'status': 'timeout', 'log': str(log)}
        records.append(record)
        print(json.dumps(record), flush=True)
        (BUILD/(args.phase+'.json')).write_text(json.dumps(records, indent=2)+'\n')
    (BUILD/(args.phase+'.json')).write_text(json.dumps(records, indent=2)+'\n')
    return int(any(r.get('exit_code', 1) != 0 for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
