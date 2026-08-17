"""Local browser interface for perceptual posterization."""

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

from .core import PosterizerConfig, posterize_array


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Posterizer</title><style>
:root{color-scheme:dark;--bg:#0c0b10;--panel:#17141d;--line:#3b3147;--ink:#f7f0ff;--muted:#aa9fb5;--pink:#ff7caf;--gold:#ffd166;--violet:#b89cff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#25172b 0,transparent 34%),var(--bg);color:var(--ink);font:14px system-ui,sans-serif}
header{height:68px;display:flex;align-items:baseline;gap:13px;padding:20px 24px;border-bottom:1px solid var(--line)}h1{font-size:21px;margin:0;letter-spacing:-.03em}header span{color:var(--muted)}
main{height:calc(100vh - 68px);display:grid;grid-template-columns:300px 1fr;gap:16px;padding:16px}aside,.stage{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:15px}aside{padding:17px;overflow:auto}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:var(--pink);margin-bottom:8px}label{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:11px 0;color:var(--muted)}
input[type=number],select{width:96px;background:#100d15;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px}input[type=file]{width:100%;margin:3px 0 12px}.check{justify-content:flex-start}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:15px}button,.download{border:0;border-radius:9px;padding:10px;background:var(--pink);color:#270816;font-weight:780;text-align:center;text-decoration:none;cursor:pointer}.download.png{background:var(--gold);color:#241b00}.download.svgz{background:var(--violet);color:#140c25}button:disabled,.download.disabled{opacity:.34;pointer-events:none}.secondary{background:#342c3d;color:var(--ink)}
.stage{display:grid;grid-template-columns:1fr 1fr;min-width:0;overflow:hidden}.pane{display:grid;grid-template-rows:43px 1fr;border-right:1px solid var(--line);min-width:0}.pane:last-child{border:0}.pane h2{font-size:12px;text-transform:uppercase;letter-spacing:.11em;color:var(--muted);margin:0;padding:14px 16px;border-bottom:1px solid var(--line)}
.canvas{min-height:0;display:flex;align-items:center;justify-content:center;padding:18px;background-color:#dedbe3;background-image:linear-gradient(45deg,#c3bec9 25%,transparent 25%),linear-gradient(-45deg,#c3bec9 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#c3bec9 75%),linear-gradient(-45deg,transparent 75%,#c3bec9 75%);background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0}.canvas img,.canvas object{max-width:100%;max-height:100%;image-rendering:auto}.empty{color:#5c5364;font-size:16px}
#palette{display:flex;gap:4px;flex-wrap:wrap;margin-top:12px}.swatch{width:22px;height:22px;border-radius:6px;border:1px solid #ffffff30}#status{color:var(--muted);margin-top:13px;line-height:1.5}.metric{color:var(--ink)}.note{font-size:12px;color:var(--muted);line-height:1.48;margin-top:13px}
@media(max-width:900px){main{grid-template-columns:1fr;height:auto}.stage{grid-template-columns:1fr}.pane{min-height:430px;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body><header><h1>Posterizer</h1><span>occupied-space OKLCH bifurcation</span></header><main><aside>
<div class="eyebrow">Perceptual palette tree</div><input id="file" type="file" accept="image/png,image/jpeg">
<label>Palette method <select id="method"><option value="oklch">OKLCH bifurcation</option><option value="inherited">Inherited mean palette</option></select></label>
<label>Colors <input id="colors" type="number" min="2" max="256" value="20"></label>
<label>Node separation <input id="separation" type="number" min="0" max="2.5" step="0.01" value="1.08"></label>
<label>Lightness weight <input id="lightness" type="number" min="0" step="0.05" value="1"></label>
<label>Chroma weight <input id="chroma" type="number" min="0" step="0.05" value="1"></label>
<label>Hue weight <input id="hue" type="number" min="0" step="0.05" value="1"></label>
<label>Minimum island <input id="island" type="number" min="0" value="6"></label>
<label>Cleanup rounds <input id="rounds" type="number" min="0" max="8" value="1"></label>
<label>Transparency <select id="alpha"><option value="auto">Auto</option><option value="cutout">Cutout</option><option value="preserve">Preserve</option></select></label>
<label class="check"><input id="trim" type="checkbox" checked> Trim transparent border</label>
<div class="actions"><button id="convert" disabled>Posterize</button><button id="clear" class="secondary">Clear</button></div>
<div class="actions"><a id="png" class="download png disabled">PNG</a><a id="svg" class="download disabled">SVG</a></div><a id="svgz" class="download svgz disabled" style="display:block;margin-top:8px">Compressed SVGZ</a>
<div id="palette"></div><div id="status">Choose a PNG or JPEG. Processing remains local.</div>
<div class="note">Each leaf proposes a two-way split in a local OKLCH tangent metric. The globally largest perceptual error reduction wins. Separation moves the resulting child nodes farther from their parent after the split.</div>
</aside><section class="stage"><div class="pane"><h2>Original</h2><div class="canvas" id="source"><div class="empty">No image loaded</div></div></div><div class="pane"><h2>Posterized structure</h2><div class="canvas" id="result"><div class="empty">Posterized preview</div></div></div></section></main>
<script>
const $=id=>document.getElementById(id);let file=null,sourceURL=null,pngURL=null,svgURL=null,svgzURL=null;const human=n=>n<1024?`${n} B`:n<1048576?`${(n/1024).toFixed(1)} KiB`:`${(n/1048576).toFixed(2)} MiB`;const revoke=()=>{for(let u of [pngURL,svgURL,svgzURL])if(u)URL.revokeObjectURL(u);pngURL=svgURL=svgzURL=null};const bytes=b64=>{let raw=atob(b64),out=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)out[i]=raw.charCodeAt(i);return out};
$('file').onchange=()=>{file=$('file').files[0]||null;$('convert').disabled=!file;if(sourceURL)URL.revokeObjectURL(sourceURL);revoke();if(file){sourceURL=URL.createObjectURL(file);$('source').innerHTML='';let i=new Image;i.src=sourceURL;$('source').append(i);$('status').textContent=`Loaded ${file.name} (${human(file.size)}) — ready.`}};
$('convert').onclick=async()=>{if(!file)return;$('convert').disabled=true;$('status').textContent='Optimizing the perceptual bifurcation tree…';let q=new URLSearchParams({method:$('method').value,colors:$('colors').value,separation:$('separation').value,lightness:$('lightness').value,chroma:$('chroma').value,hue:$('hue').value,island:$('island').value,rounds:$('rounds').value,alpha:$('alpha').value,trim:$('trim').checked?'1':'0',name:file.name});try{let r=await fetch('/convert?'+q,{method:'POST',body:file});let data=await r.json();if(!r.ok)throw Error(data.error||'Posterization failed');revoke();pngURL=URL.createObjectURL(new Blob([bytes(data.png)],{type:'image/png'}));svgURL=URL.createObjectURL(new Blob([data.svg],{type:'image/svg+xml'}));svgzURL=URL.createObjectURL(new Blob([bytes(data.svgz)],{type:'image/svg+xml'}));$('result').innerHTML='';let preview=new Image;preview.src=pngURL;preview.alt='Posterized preview';$('result').append(preview);let stem=file.name.replace(/\.(png|jpe?g)$/i,'');for(let [id,url,ext] of [['png',pngURL,'.png'],['svg',svgURL,'.svg'],['svgz',svgzURL,'.svgz']]){$(id).href=url;$(id).download=stem+'-posterized'+ext;$(id).classList.remove('disabled')}$('palette').innerHTML=data.diagnostics.palette.map(c=>`<span class="swatch" title="${c}" style="background:${c.slice(0,7)}"></span>`).join('');let d=data.diagnostics;$('status').innerHTML=`<span class="metric">${d.visible_palette_colors} visible colors · perceptual RMSE ${d.perceptual_rmse.toFixed(4)} · MSE ${d.rgba_mse_255.toFixed(2)} · ${d.loops.toLocaleString()} loops · ${human(d.svgz_bytes)} SVGZ · ${(d.total_ms/1000).toFixed(2)} s</span>`}catch(e){$('status').textContent=e.message}finally{$('convert').disabled=false}};$('clear').onclick=()=>location.reload();
</script></body></html>"""


def convert_request(data: bytes, query: str) -> dict:
    values = parse_qs(query)
    get = lambda key, default: values.get(key, [str(default)])[0]
    config = PosterizerConfig(
        colors=int(get("colors", 20)),
        method=get("method", "oklch"),
        lightness_weight=float(get("lightness", 1.0)),
        chroma_weight=float(get("chroma", 1.0)),
        hue_weight=float(get("hue", 1.0)),
        node_separation=float(get("separation", 1.08)),
        minimum_island=int(get("island", 6)),
        cleanup_rounds=int(get("rounds", 1)),
        alpha_mode=get("alpha", "auto"),
        trim_transparent=get("trim", "1") != "0",
    )
    with Image.open(BytesIO(data)) as image:
        rgba = np.asarray(image.convert("RGBA"))
    result = posterize_array(rgba, config, title=get("name", "posterized image"))
    png = BytesIO()
    Image.fromarray(result.posterized_rgba, "RGBA").save(png, format="PNG")
    return {
        "png": base64.b64encode(png.getvalue()).decode("ascii"),
        "svg": result.svg,
        "svgz": base64.b64encode(result.svgz).decode("ascii"),
        "diagnostics": {**result.diagnostics, "source_bytes": len(data)},
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/": self._send(404, "text/plain", b"not found"); return
        self._send(200, "text/html; charset=utf-8", PAGE.encode())

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/convert": self._send(404, "application/json", b'{"error":"not found"}'); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 50 * 1024 * 1024: raise ValueError("image must be between 1 byte and 50 MiB")
            payload = convert_request(self.rfile.read(length), parsed.query)
            self._send(200, "application/json", json.dumps(payload).encode())
        except Exception as error:
            self._send(400, "application/json", json.dumps({"error": str(error)}).encode())

    def log_message(self, format: str, *args) -> None: return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the perceptual posterizer GUI")
    parser.add_argument("--port", type=int, default=0); parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv); server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"; print(f"Posterizer: {url}"); print("Press Ctrl-C to stop.")
    if not args.no_browser: threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0


if __name__ == "__main__": raise SystemExit(main())
