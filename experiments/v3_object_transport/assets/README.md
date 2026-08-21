# Frozen control rasters

These are the exact inputs for the first V3 object-recomposition audit.
Research runs must not silently fetch or substitute gallery images.

| file | provenance | SHA-256 |
|---|---|---|
| `pikachu_easy.png` | user-supplied easy control, copied byte-for-byte from `/Users/ultimussecundai/Downloads/025.png` | `759ceafdb4b8d637dfcfe673cf672bc3c472354a3b2e7e0835d9cdcacd8d195d` |
| `pikachu_hard.png` | exact deterministic hard-frame derivative; black exterior and an 8-pixel white wall outside the unchanged original panel | `f1a94868ed5ce13347af73ada8123956fc9d49241395b4dbe70501d3008d34a4` |
| `coffee.png` | `skimage.data.coffee`, scikit-image 0.26.0 bundled raster | `cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7` |
| `astronaut.png` | `skimage.data.astronaut`, scikit-image 0.26.0 bundled raster | `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5` |
| `checker.png` | `skimage.data.checkerboard`, scikit-image 0.26.0 `chessboard_GRAY.png` raster | `3e51870774515af4d07d820bd8827364c70839bf9b573c746e485095e893df90` |
| `coins.png` | `skimage.data.coins`, scikit-image 0.26.0 bundled raster | `f8d773fc9cfa6f4d8e5942dc34d0a0788fcaed2a4fefbbed0aef5398d7ef4cba` |

`build_pikachu_controls.py` reproduces both Pikachu fixtures.  Every pixel in
the original black panel is identical between the easy and hard versions; the
hard version changes only the exterior margin.  The inner top of its white
wall remains at row 35 and the first dark ear-tip pixel is at row 38.

The controls have different intended failure roles; these are hypotheses to
test, not rules supplied to the algorithm:

- hard Pikachu: dark ear tips versus identical dark surround and nearby wall;
- coffee: cup/plate assembly, reluctant spoon, and textured table;
- astronaut: suit/flag attraction, fragmented flag, and occlusion ordering;
- checkerboard: repeated role structure with easy global organization;
- coins: repeated instances sharing appearance and a common surround.
