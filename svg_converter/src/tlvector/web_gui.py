"""Local browser GUI with a true SVG preview."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np
from PIL import Image

from .core import VectorizerConfig, vectorize_array


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Transport-Locked Vectorizer</title>
<style>
:root{color-scheme:dark;--ink:#eef4f1;--muted:#98aaa4;--panel:#16211f;--line:#30433e;--accent:#6ee7bd}
*{box-sizing:border-box}body{margin:0;background:#0b1110;color:var(--ink);font:14px system-ui,sans-serif}
header{height:64px;display:flex;align-items:center;gap:14px;padding:0 22px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0}header span{color:var(--muted)}main{height:calc(100vh - 64px);display:grid;grid-template-columns:260px 1fr;gap:16px;padding:16px}
aside,.stage{background:var(--panel);border:1px solid var(--line);border-radius:14px}aside{padding:16px;overflow:auto}
label{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:11px 0;color:var(--muted)}
input[type=number]{width:82px;background:#0d1513;color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px}
input[type=file]{width:100%;margin:4px 0 12px}.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:16px}
button,.download{border:0;border-radius:8px;padding:10px;background:var(--accent);color:#092018;font-weight:700;text-align:center;text-decoration:none;cursor:pointer}
button:disabled,.download.disabled{opacity:.4;pointer-events:none}.secondary{background:#263934;color:var(--ink)}
.stage{display:grid;grid-template-columns:1fr 1fr;min-width:0;overflow:hidden}.pane{display:grid;grid-template-rows:42px 1fr;border-right:1px solid var(--line);min-width:0}.pane:last-child{border:0}
.pane h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0;padding:14px 16px;border-bottom:1px solid var(--line)}
.canvas{min-height:0;display:flex;align-items:center;justify-content:center;padding:18px;background-color:#d8d8d8;background-image:linear-gradient(45deg,#bbb 25%,transparent 25%),linear-gradient(-45deg,#bbb 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#bbb 75%),linear-gradient(-45deg,transparent 75%,#bbb 75%);background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0}
.canvas img,.canvas object{max-width:100%;max-height:100%;background:transparent}.empty{color:#43544f;text-align:center;font-size:16px}
#status{color:var(--muted);margin-top:14px;line-height:1.45}.check{justify-content:flex-start}.note{font-size:12px;color:var(--muted);line-height:1.45;margin-top:14px}
@media(max-width:850px){main{grid-template-columns:1fr;height:auto}.stage{grid-template-columns:1fr}.pane{min-height:440px;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body>
<header><h1>Transport-Locked Vectorizer</h1><span>PNG → topology-first SVG</span></header>
<main><aside>
<input id="file" type="file" accept="image/png">
<label>Structural colors <input id="colors" type="number" min="1" max="256" value="12"></label>
<label>Residual colors <input id="details" type="number" min="0" max="32" value="6"></label>
<label>Coarse side <input id="coarse" type="number" min="16" max="1024" value="160"></label>
<label>Minimum island <input id="minimum" type="number" min="1" value="10"></label>
<label>Simplification <input id="simplify" type="number" min="0" step="0.05" value="0.85"></label>
<label>Curve tolerance <input id="curve" type="number" min="0" step="0.05" value="0.65"></label>
<label>Subpixel passes <input id="smoothing" type="number" min="0" max="20" value="4"></label>
<label>Seam overlap <input id="seam" type="number" min="0" step="0.05" value="0.65"></label>
<label>Transparency <select id="alpha"><option value="auto">Auto</option><option value="cutout">Cutout</option><option value="preserve">Preserve</option></select></label>
<label class="check"><input id="trim" type="checkbox" checked> Trim transparent border</label>
<div class="actions"><button id="convert" disabled>Convert</button><a id="download" class="download disabled">Export SVG</a></div>
<button id="clear" class="secondary" style="width:100%;margin-top:8px">Clear</button>
<div id="status">Choose a PNG. All processing stays on this machine.</div>
<div class="note">Seam overlap closes white antialiasing pinholes. Subpixel passes smooth raster stairs while persistent corners stay pinned.</div>
</aside><section class="stage">
<div class="pane"><h2>Source PNG</h2><div class="canvas" id="source"><div class="empty">No image loaded</div></div></div>
<div class="pane"><h2>Rendered SVG</h2><div class="canvas" id="result"><div class="empty">Conversion preview</div></div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);let file=null,sourceURL=null,svgURL=null;
$('file').onchange=()=>{file=$('file').files[0]||null;$('convert').disabled=!file;if(sourceURL)URL.revokeObjectURL(sourceURL);if(file){sourceURL=URL.createObjectURL(file);$('source').innerHTML='';let i=new Image;i.src=sourceURL;$('source').append(i);$('status').textContent=`Loaded ${file.name} — ready to convert.`}};
$('convert').onclick=async()=>{if(!file)return;$('convert').disabled=true;$('status').textContent='Converting…';let q=new URLSearchParams({colors:$('colors').value,details:$('details').value,coarse:$('coarse').value,minimum:$('minimum').value,simplify:$('simplify').value,curve:$('curve').value,smoothing:$('smoothing').value,seam:$('seam').value,trim:$('trim').checked?'1':'0',name:file.name});
q.set('alpha',$('alpha').value);try{let r=await fetch('/convert?'+q,{method:'POST',headers:{'Content-Type':'image/png'},body:file});let data=await r.json();if(!r.ok)throw Error(data.error||'Conversion failed');if(svgURL)URL.revokeObjectURL(svgURL);svgURL=URL.createObjectURL(new Blob([data.svg],{type:'image/svg+xml'}));$('result').innerHTML='';let o=document.createElement('object');o.type='image/svg+xml';o.data=svgURL;$('result').append(o);$('download').href=svgURL;$('download').download=file.name.replace(/\.png$/i,'')+'.svg';$('download').classList.remove('disabled');let d=data.diagnostics;$('status').textContent=`Ready — ${d.paths} paths, ${d.loops} loops, ${d.alpha_mode} alpha, ${(d.svg_bytes/1024).toFixed(1)} KiB, ${(d.total_ms/1000).toFixed(2)} s.`}catch(e){$('status').textContent=e.message}finally{$('convert').disabled=false}};
$('clear').onclick=()=>{location.reload()};
</script></body></html>"""


def convert_request(data: bytes, query: str) -> dict:
    values = parse_qs(query)
    get = lambda key, default: values.get(key, [str(default)])[0]
    config = VectorizerConfig(
        colors=int(get("colors", 12)), detail_colors=int(get("details", 6)),
        coarse_side=int(get("coarse", 160)), minimum_region=int(get("minimum", 10)),
        simplify=float(get("simplify", 0.85)), curve_tolerance=float(get("curve", 0.65)),
        subpixel_smoothing=int(get("smoothing", 4)), seam_overlap=float(get("seam", 0.65)),
        alpha_mode=get("alpha", "auto"),
        trim_transparent=get("trim", "1") != "0",
    )
    with Image.open(BytesIO(data)) as image:
        rgba = np.asarray(image.convert("RGBA"))
    result = vectorize_array(rgba, config, title=get("name", "vectorized image"))
    return {"svg": result.svg, "diagnostics": result.diagnostics}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/":
            self._send(404, "text/plain", b"not found")
            return
        self._send(200, "text/html; charset=utf-8", PAGE.encode())

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/convert":
            self._send(404, "application/json", b'{"error":"not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 50 * 1024 * 1024:
                raise ValueError("PNG must be between 1 byte and 50 MiB")
            payload = convert_request(self.rfile.read(length), parsed.query)
            data = json.dumps(payload).encode()
            self._send(200, "application/json", data)
        except Exception as error:
            data = json.dumps({"error": str(error)}).encode()
            self._send(400, "application/json", data)

    def log_message(self, format: str, *args) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the local vectorizer GUI")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Transport-Locked Vectorizer: {url}")
    print("Press Ctrl-C to stop.")
    if not args.no_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
