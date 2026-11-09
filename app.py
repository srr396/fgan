import base64, io, os, sys

import torch
from flask import Flask, render_template_string, request
from PIL import Image

from gan_core import (Generator, Discriminator, PATTERNS, generate_variants, describe_style,
                      to_pil, verdict)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.environ.get("CKPT", "checkpoints/gan.pt")
if not os.path.exists(CKPT):
    sys.exit("Checkpoint %s not found. Run the training cells in the notebook first." % CKPT)

ck = torch.load(CKPT, map_location=DEVICE)
G = Generator().to(DEVICE); G.load_state_dict(ck["G"]); G.eval()
D = Discriminator().to(DEVICE); D.load_state_dict(ck["D"]); D.eval()
THRESH = float(ck["threshold"])

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>Clothing Restyle GAN</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f6;color:#222}
 header{background:#222;color:#fff;padding:16px 28px}
 main{max-width:1100px;margin:auto;padding:24px}
 form{background:#fff;padding:18px;border-radius:10px;display:flex;flex-wrap:wrap;gap:16px;align-items:end}
 label{font-size:13px;display:flex;flex-direction:column;gap:4px}
 button{background:#4c3fe0;color:#fff;border:0;padding:10px 22px;border-radius:8px;font-size:15px;cursor:pointer}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:16px;margin-top:22px}
 .card{background:#fff;border-radius:10px;padding:10px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 .card img{width:100%;border-radius:6px;image-rendering:auto}
 .tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;color:#fff}
 .real{background:#1e9d57}.gen{background:#d6453d}
 .sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;border:1px solid #ccc}
 .bar{height:8px;background:#e3e3e8;border-radius:4px;margin:6px 0;overflow:hidden}
 .bar>div{height:100%;background:#4c3fe0}
 .err{color:#b00020;margin-top:12px}
 small{color:#666}
</style></head><body>
<header><h2 style="margin:0">Clothing Restyle GAN</h2>
<small style="color:#bbb">Upload a garment, get new patterns and colours. Discriminator threshold: {{ '%.2f' % thresh }}</small></header>
<main>
<form method="post" enctype="multipart/form-data">
 <label>Clothing image<input type="file" name="image" accept="image/*" required></label>
 <label>Variants<input type="number" name="count" value="6" min="1" max="12" style="width:70px"></label>
 <label>Pattern<select name="pattern"><option value="random">random</option>
  {% for p in patterns %}<option value="{{p}}">{{p}}</option>{% endfor %}</select></label>
 <label><span><input type="checkbox" name="use_colors"> use my colours</span>
  <span><input type="color" name="color1" value="#c0392b"> <input type="color" name="color2" value="#f4d03f"></span></label>
 <button type="submit">Generate</button>
</form>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
{% if results %}
 <h3>Input</h3>
 <div class="grid"><div class="card"><img src="{{ source.img }}">
   <div class="bar"><div style="width:{{ source.pct }}%"></div></div>
   D score {{ '%.3f' % source.score }} <span class="tag {{ 'real' if source.verdict=='REAL' else 'gen' }}">{{ source.verdict }}</span></div></div>
 <h3>Generated designs</h3>
 <div class="grid">
 {% for r in results %}
  <div class="card"><img src="{{ r.img }}">
   <div style="margin-top:6px">{{ r.pattern }}
    <span class="sw" style="background:{{ r.c1 }}"></span><span class="sw" style="background:{{ r.c2 }}"></span></div>
   <div class="bar"><div style="width:{{ r.pct }}%"></div></div>
   D score {{ '%.3f' % r.score }} <span class="tag {{ 'real' if r.verdict=='REAL' else 'gen' }}">{{ r.verdict }}</span>
  </div>
 {% endfor %}</div>
 <p><small>Score = discriminator's probability that the image is a real (dataset-style) garment. A GAN generator is trained to fool this score,
 so GENERATED verdicts become rarer as training improves.</small></p>
{% endif %}
</main></body></html>
"""


def b64(t):
    buf = io.BytesIO()
    to_pil(t).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


@app.route("/", methods=["GET", "POST"])
def index():
    results, source, error = None, None, None
    if request.method == "POST":
        f = request.files.get("image")
        try:
            if f is None or f.filename == "":
                raise ValueError("Please choose an image.")
            img = Image.open(f.stream)
            n = max(1, min(12, int(request.form.get("count", 6))))
            pat = request.form.get("pattern", "random")
            pat = None if pat == "random" else PATTERNS.index(pat)
            colors = None
            if request.form.get("use_colors"):
                colors = (hex_to_rgb(request.form["color1"]), hex_to_rgb(request.form["color2"]))
            src, outs, styles, scores, src_score = generate_variants(G, D, img, n=n, pattern=pat,
                                                                      colors=colors, device=DEVICE)
            source = dict(img=b64(src), score=src_score, pct=int(src_score * 100), verdict=verdict(src_score, THRESH))
            results = []
            for i in range(n):
                name, c1, c2 = describe_style(styles[i])
                results.append(dict(img=b64(outs[i]), pattern=name, c1=c1, c2=c2, score=scores[i],
                                    pct=int(scores[i] * 100), verdict=verdict(scores[i], THRESH)))
        except Exception as e:
            error = "Could not process the image: %s" % e
    return render_template_string(PAGE, results=results, source=source, error=error,
                                  patterns=PATTERNS, thresh=THRESH)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Open http://127.0.0.1:%d" % port)
    app.run(host="127.0.0.1", port=port, debug=False)
