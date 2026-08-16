"""Local browser interface for the compact converter-v2 method."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np
from PIL import Image

from .core import V2Config, vectorize_array_v2


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Converter V2</title>
<style>
:root{color-scheme:dark;--bg:#090e0d;--panel:#121b19;--line:#2a3b36;--ink:#edf6f2;--muted:#93a8a0;--accent:#72e6b8;--blue:#7fc7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif}
header{height:66px;display:flex;align-items:baseline;gap:13px;padding:20px 24px;border-bottom:1px solid var(--line)}
h1{font-size:19px;letter-spacing:-.02em;margin:0}header span{color:var(--muted)}main{height:calc(100vh - 66px);display:grid;grid-template-columns:290px 1fr;gap:16px;padding:16px}
aside,.stage{background:var(--panel);border:1px solid var(--line);border-radius:14px}aside{padding:17px;overflow:auto}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);margin-bottom:8px}
label{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:11px 0;color:var(--muted)}
input[type=number],select{width:88px;background:#0c1412;color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px}
input[type=file]{width:100%;margin:3px 0 12px}.check{justify-content:flex-start}.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:16px}
button,.download{border:0;border-radius:8px;padding:10px;background:var(--accent);color:#072019;font-weight:750;text-align:center;text-decoration:none;cursor:pointer}
.download.svgz{background:var(--blue);color:#071724}button:disabled,.download.disabled{opacity:.35;pointer-events:none}.secondary{background:#253632;color:var(--ink)}
.stage{display:grid;grid-template-columns:1fr 1fr;min-width:0;overflow:hidden}.pane{display:grid;grid-template-rows:43px 1fr;border-right:1px solid var(--line);min-width:0}.pane:last-child{border:0}
.pane h2{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0;padding:14px 16px;border-bottom:1px solid var(--line)}
.canvas{min-height:0;display:flex;align-items:center;justify-content:center;padding:18px;background-color:#d8d8d8;background-image:linear-gradient(45deg,#bbb 25%,transparent 25%),linear-gradient(-45deg,#bbb 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#bbb 75%),linear-gradient(-45deg,transparent 75%,#bbb 75%);background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0}
.canvas img,.canvas object{max-width:100%;max-height:100%;background:transparent}.empty{color:#465750;text-align:center;font-size:16px}
#status{color:var(--muted);margin-top:14px;line-height:1.5}.metric{color:var(--ink)}.smaller{color:var(--accent);font-weight:700}.note{font-size:12px;color:var(--muted);line-height:1.48;margin-top:14px}
@media(max-width:880px){main{grid-template-columns:1fr;height:auto}.stage{grid-template-columns:1fr}.pane{min-height:440px;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body>
<header><h1>Converter V2</h1><span>compact exact-lattice PNG → SVG / SVGZ</span></header>
<main><aside>
<div class="eyebrow">Rate–distortion controls</div>
<input id="file" type="file" accept="image/png">
<label>Structural colors <input id="colors" type="number" min="1" max="256" value="128"></label>
<label>Split budget <input id="budget" type="number" min="0" max="2048" value="128"></label>
<label>Headroom MSE <input id="splitTarget" type="number" min="0" step="0.1" value="20"></label>
<label>Final MSE <input id="finalTarget" type="number" min="0" step="0.1" value="30"></label>
<label>Maximum merge area <input id="area" type="number" min="1" value="32"></label>
<label>Merge rounds <input id="rounds" type="number" min="0" max="8" value="2"></label>
<label>Coarse side <input id="coarse" type="number" min="16" max="1024" value="160"></label>
<label>Minimum region <input id="minimum" type="number" min="1" value="10"></label>
<label>Transparency <select id="alpha"><option value="auto">Auto</option><option value="cutout">Cutout</option><option value="preserve">Preserve</option></select></label>
<label class="check"><input id="trim" type="checkbox" checked> Trim transparent border</label>
<div class="actions"><button id="convert" disabled>Convert</button><button id="clear" class="secondary">Clear</button></div>
<div class="actions"><a id="svg" class="download disabled">Export SVG</a><a id="svgz" class="download svgz disabled">Export SVGZ</a></div>
<div id="status">Choose a PNG. Processing remains on this machine.</div>
<div class="note">V2 first creates quality headroom, then spends it merging small components by added error per estimated byte saved. The final MSE is a hard ceiling. SVGZ is deterministic gzip-compressed SVG.</div>
</aside><section class="stage">
<div class="pane"><h2>Source PNG</h2><div class="canvas" id="source"><div class="empty">No image loaded</div></div></div>
<div class="pane"><h2>Rendered compact SVG</h2><div class="canvas" id="result"><div class="empty">Conversion preview</div></div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);let file=null,sourceURL=null,svgURL=null,svgzURL=null;
const human=n=>n<1024?`${n} B`:n<1048576?`${(n/1024).toFixed(1)} KiB`:`${(n/1048576).toFixed(2)} MiB`;
const revoke=()=>{if(svgURL)URL.revokeObjectURL(svgURL);if(svgzURL)URL.revokeObjectURL(svgzURL);svgURL=svgzURL=null};
$('file').onchange=()=>{file=$('file').files[0]||null;$('convert').disabled=!file;if(sourceURL)URL.revokeObjectURL(sourceURL);revoke();if(file){sourceURL=URL.createObjectURL(file);$('source').innerHTML='';let i=new Image;i.src=sourceURL;$('source').append(i);$('status').textContent=`Loaded ${file.name} (${human(file.size)}) — ready.`}};
$('convert').onclick=async()=>{if(!file)return;$('convert').disabled=true;$('status').textContent='Building occupation, merging components, and encoding paths…';let q=new URLSearchParams({colors:$('colors').value,budget:$('budget').value,split_target:$('splitTarget').value,final_target:$('finalTarget').value,area:$('area').value,rounds:$('rounds').value,coarse:$('coarse').value,minimum:$('minimum').value,alpha:$('alpha').value,trim:$('trim').checked?'1':'0',name:file.name});
try{let r=await fetch('/convert?'+q,{method:'POST',headers:{'Content-Type':'image/png'},body:file});let data=await r.json();if(!r.ok)throw Error(data.error||'Conversion failed');revoke();svgURL=URL.createObjectURL(new Blob([data.svg],{type:'image/svg+xml'}));let raw=atob(data.svgz);let bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);svgzURL=URL.createObjectURL(new Blob([bytes],{type:'image/svg+xml'}));$('result').innerHTML='';let o=document.createElement('object');o.type='image/svg+xml';o.data=svgURL;$('result').append(o);let stem=file.name.replace(/\.png$/i,'');$('svg').href=svgURL;$('svg').download=stem+'-v2.svg';$('svgz').href=svgzURL;$('svgz').download=stem+'-v2.svgz';$('svg').classList.remove('disabled');$('svgz').classList.remove('disabled');let d=data.diagnostics;let ratio=d.svgz_bytes/d.source_bytes;let verdict=d.svgz_bytes<d.source_bytes?`<span class="smaller">SVGZ is ${(100*(1-ratio)).toFixed(1)}% smaller than the PNG.</span>`:`SVGZ is ${(ratio).toFixed(2)}× the PNG size.`;$('status').innerHTML=`<span class="metric">MSE ${d.final_mse.toFixed(2)} · ${d.loops.toLocaleString()} loops · ${d.component_merges.toLocaleString()} merges · ${human(d.svg_bytes)} SVG · ${human(d.svgz_bytes)} SVGZ · ${(d.total_ms/1000).toFixed(2)} s</span><br>${verdict}`}
catch(e){$('status').textContent=e.message}finally{$('convert').disabled=false}};
$('clear').onclick=()=>location.reload();
</script></body></html>"""


def convert_request(data: bytes, query: str) -> dict:
    values = parse_qs(query)
    get = lambda key, default: values.get(key, [str(default)])[0]
    config = V2Config(
        colors=int(get("colors", 128)),
        split_budget=int(get("budget", 128)),
        split_target_mse=float(get("split_target", 20.0)),
        final_target_mse=float(get("final_target", 30.0)),
        coarse_side=int(get("coarse", 160)),
        minimum_region=int(get("minimum", 10)),
        merge_maximum_area=int(get("area", 32)),
        merge_rounds=int(get("rounds", 2)),
        alpha_mode=get("alpha", "auto"),
        trim_transparent=get("trim", "1") != "0",
    )
    with Image.open(BytesIO(data)) as image:
        rgba = np.asarray(image.convert("RGBA"))
    result = vectorize_array_v2(
        rgba, config, title=get("name", "converter v2 image")
    )
    diagnostics = {**result.diagnostics, "source_bytes": len(data)}
    return {
        "svg": result.svg,
        "svgz": base64.b64encode(result.svgz).decode("ascii"),
        "diagnostics": diagnostics,
    }


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
            self._send(200, "application/json", json.dumps(payload).encode())
        except Exception as error:
            self._send(
                400, "application/json", json.dumps({"error": str(error)}).encode()
            )

    def log_message(self, format: str, *args) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the converter-v2 web GUI")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Converter V2: {url}")
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
