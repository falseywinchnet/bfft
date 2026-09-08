# On Bruun, Revisited

The manuscript is organized as an IEEEtran journal paper with one TeX file per
mathematical section and four proof appendices.

Build from this directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The canonical deliverable is copied to `output/pdf/on_bruun_revisited.pdf`
after a clean build and rendered-page inspection.
