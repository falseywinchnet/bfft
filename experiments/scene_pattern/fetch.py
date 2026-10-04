"""Fetch Oxford's six real Graffiti photographs and evaluation homographies."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
URL = 'https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz'

def fetch(destination=HERE / 'data' / 'graf'):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = [f'img{i}.ppm' for i in range(1, 7)] + [f'H1to{i}p' for i in range(2, 7)]
    if all((destination / name).exists() for name in names):
        return destination
    raw = urllib.request.urlopen(URL, timeout=60).read()
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name in names and member.isfile():
                data = archive.extractfile(member).read()
                (destination / name).write_bytes(data)
                files[name] = hashlib.sha256(data).hexdigest()
    if set(files) != set(names):
        raise RuntimeError('Oxford archive missing expected photographs or evaluation maps')
    (destination / 'provenance.json').write_text(json.dumps({
        'source': URL, 'source_page': 'https://www.robots.ox.ac.uk/~vgg/research/affine/',
        'archive_sha256': hashlib.sha256(raw).hexdigest(), 'files_sha256': files,
        'purpose': 'Local research evaluation; homographies are held out from registration.',
    }, indent=2) + '\n')
    return destination

if __name__ == '__main__':
    print(fetch())
