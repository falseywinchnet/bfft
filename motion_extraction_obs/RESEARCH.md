# Posy motion-extraction mode map

Primary demonstration:

- Posy, *Motion Extraction* (2023):
  https://www.youtube.com/watch?v=NSS6yAMZF78

Useful independent implementation notes:

- Martino Trapanotto reconstructs neutral-gray signed subtraction, variable
  delays, a live ring buffer, additive overlays, and blurred glow:
  https://martinoshelf.neocities.org/posy-motion-part1
- Chigozie Nri's notes distinguish delayed, fixed-frame, and independently
  delayed RGB-channel forms:
  https://gist.github.com/chigozienri/0ac602c3bc7d6c5fab9f0c6839f4ad6f
- Javier Bórquez demonstrates difference-only and highlighted-difference
  compositing:
  https://javier.xyz/blog/motion-extraction-with-mostly-css

## Live translation

| Posy demonstration | Live filter state |
|---|---|
| Move duplicate one frame | Exact short-delay reference |
| Move duplicate seconds | Budgeted native-resolution snapshot history |
| Freeze duplicate | Explicit captured baseline |
| Deer in forest | High-gain absolute difference |
| Changes over time / stones | Frozen-baseline difference |
| Highlight over normal image | Additive motion overlay |
| Blur for large features / wind | Separable blurred signed difference |
| Glow | Blurred absolute motion overlay |
| Red, green, blue time shifting | Independently delayed grayscale channels |
| Change colors | Signed two-color mapping |
| Keep source color | Signed luma extraction retaining source chroma |

The additional acceleration mode mixes first- and second-order temporal
evidence:

`e = (1-a)|I(t)-I(t-d)| + a|I(t)-2I(t-d)+I(t-2d)|`

After a noise floor, `pow(gain*e, exponent)` controls brightness. Its spatial
gradient drives a bounded refraction-like displacement of the current frame.
