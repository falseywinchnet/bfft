"""Read-only source audit. Writes only a receipt beside this script."""
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
MIRROR = Path('/Users/joshuahkuttenkuler/Developer/CodexBuilds/bfft-6b3e7ffa7539')
ARCHIVE = Path('/Users/joshuahkuttenkuler/CodexEmergencyBackups/20260916T015732Z-MacBook-Neo/home-bfft.tar.gz')
PREFIX = 'Users/ultimussecundai/bfft/'
FILES = [
    'output/pdf/conv_paper_composed.tex',
    'output/pdf/convstar_warp_addendum.tex',
    'output/pdf/convstar_warp_measurements.tex',
    'experiments/conv_warp/compact_measurement/reference_source.c',
    'experiments/conv_warp/joint_reference.py',
    'experiments/conv_warp/PRELIMINARY_THEORY.md',
]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    remaining = set(FILES)
    results = {}
    with tarfile.open(ARCHIVE, mode='r|gz') as archive:
        for member in archive:
            relative = member.name.removeprefix(PREFIX)
            if relative not in remaining:
                continue
            with archive.extractfile(member) as source:
                archived = source.read()
            mirrored = (MIRROR / relative).read_bytes()
            results[relative] = {
                'bytes': len(archived),
                'archive_sha256': digest(archived),
                'mirror_sha256': digest(mirrored),
                'byte_identical': archived == mirrored,
                'archive_mtime_utc': datetime.datetime.fromtimestamp(member.mtime, datetime.timezone.utc).isoformat(),
            }
            remaining.remove(relative)
            if not remaining:
                break
    published = (ROOT / 'published_main.tex').read_bytes()
    mirrored_paper = (MIRROR / FILES[0]).read_bytes()
    receipt = {
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'archive': str(ARCHIVE),
        'archive_member_prefix': PREFIX,
        'mirror': str(MIRROR),
        'files': results,
        'missing': sorted(remaining),
        'published_paper': {
            'url': 'https://github.com/falseywinchnet/papers_please/blob/main/conv_paper_composed.tex',
            'bytes': len(published),
            'sha256': digest(published),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(published)).encode() + b'\0' + published).hexdigest(),
            'byte_identical_to_mirror': published == mirrored_paper,
        },
        'published_head': json.loads((ROOT / 'published_head.json').read_text()),
    }
    assert not remaining, 'Missing named archive members'
    assert all(row['byte_identical'] for row in results.values()), 'Archive/mirror mismatch'
    assert published == mirrored_paper, 'Published paper/mirror mismatch'
    assert receipt['published_paper']['git_blob_sha1'] == '7e2b7d3b1821ba5f5a60e92fcb4d49d0e2926232'
    (ROOT / 'source_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
