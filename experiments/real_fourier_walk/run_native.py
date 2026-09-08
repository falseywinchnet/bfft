"""Run the actual examples/benchmark.cpp binary, retaining every output line."""
import argparse
import hashlib
import json
import pathlib
import platform
import subprocess


def command(*args):
    return subprocess.check_output(args,text=True).strip()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',default='/tmp/quartic_bfft_benchmark')
    parser.add_argument('--out',default='/tmp/quartic_native_final')
    parser.add_argument('--runs',type=int,default=3)
    args=parser.parse_args();out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    root=pathlib.Path(__file__).resolve().parents[2]
    sources=['examples/benchmark.cpp','src/bfft.cpp','src/detail/bruun_dif_kernel.hpp',
             'src/detail/bruun_simd_backend.hpp','experiments/real_fourier_walk/normalized_quartic.hpp',
             'experiments/real_fourier_walk/test_native.cpp','experiments/real_fourier_walk/test_factorization.py',
             'experiments/real_fourier_walk/run_native.py','experiments/real_fourier_walk/run_native.sh']
    metadata={'host':platform.platform(),'cpu':command('sysctl','-n','machdep.cpu.brand_string'),
              'compiler':command('clang++','--version'),'uptime_start':command('uptime'),
              'flags':'-O3 -DNDEBUG -std=c++17 -ffp-contract=fast -DBFFT_BENCH_QUARTIC_WALK -Iinclude',
              'benchmark':'examples/benchmark.cpp + src/bfft.cpp; default DIF dispatch',
              'sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in sources}}
    rows=[]
    with (out/'benchmark.txt').open('w') as raw:
        for run in range(args.runs):
            sizes=[2**p for p in range(6,21)]
            if run&1:sizes.reverse()
            for n in sizes:
                # About several milliseconds per small/medium-size chunk;
                # large sizes retain at least eight full transforms per chunk.
                iters=max(8,min(100000,8000000//n))
                result=subprocess.run([args.binary,str(n),str(iters)],capture_output=True,text=True,check=True)
                raw.write(f'RUN {run} N {n}\n'+result.stdout+result.stderr);raw.flush()
                records=[json.loads(line[8:]) for line in result.stdout.splitlines() if line.startswith('QUARTIC ')]
                assert len(records)==4
                rows.extend(dict(row,run=run) for row in records)
                (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(f'run {run} N {n}: separated / DIF {records[1]["walk_over_dif"]:.3f}',flush=True)
    metadata['uptime_end']=command('uptime')
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
