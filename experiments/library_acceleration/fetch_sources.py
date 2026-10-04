"""Fetch pinned, unmodified upstream sources into ignored tmp/; no install.

SPORCO's filetype dependency is unpacked beside it solely for import. All
execution uses PYTHONPATH; the system/site-packages environment is untouched.
"""
import hashlib,io,json,tarfile,urllib.request,zipfile
from pathlib import Path


def checked(url,expected):
    blob=urllib.request.urlopen(url).read()
    if hashlib.sha256(blob).hexdigest()!=expected:raise ValueError('source digest mismatch')
    return blob


def main():
    dest=Path('tmp/library_sources');dest.mkdir(parents=True,exist_ok=True)
    blob=checked('https://github.com/bwohlberg/sporco/archive/refs/tags/v0.2.2.post1.tar.gz',
                 '9b60ea0d9bfb463323f7072ee4dc0e9f9f5ccd9bc0fd98110a6ab2134f00072f')
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for item in tar.getmembers():
            path=Path(item.name)
            if path.is_absolute() or '..' in path.parts or item.issym() or item.islnk():
                raise ValueError('unexpected archive member')
        tar.extractall(dest)
    metadata=json.load(urllib.request.urlopen('https://pypi.org/pypi/filetype/1.2.0/json'))
    url=next(x['url'] for x in metadata['urls'] if x['filename']=='filetype-1.2.0-py2.py3-none-any.whl')
    blob=checked(url,'7ce71b6880181241cf7ac8697a2f1eb6a8bd9b429f7ad6d27b8db9ba5f1c2d25')
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for item in archive.namelist():
            if Path(item).is_absolute() or '..' in Path(item).parts:raise ValueError('unexpected wheel member')
        archive.extractall(dest/'deps')
    print('Pinned source checkouts are ready in',dest)


if __name__=='__main__':main()
