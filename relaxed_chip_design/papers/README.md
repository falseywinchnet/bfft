# Research corpus: transport that keeps its inverse

Downloaded and checked on 2026-08-11.  These papers were selected for the
underlying operations in relaxed chip design, not for superficial similarity
to placement: circular transport, synchronization, heterogeneous graph state,
reversible multiscale transforms, sparse elimination, exact tree messages, and
the distinction between mass and identity.

## Admission rule

The proposed runtime is not allowed to emit several placements and select one.
It is also not allowed to hide a destination search inside a multiscale loop.
The useful methods here have one of these forms:

- a closed-form statistic such as a prefix sum or weighted median;
- one sparse factorization, spectral synchronization, or fixed polynomial;
- one causal pass over a tree or stored predecessor DAG;
- an exactly reversible analysis/synthesis transform.

`CORE` means the representation or theorem belongs in the proposed design.
`PARTIAL` means a theorem or invariant is useful but the paper's optimizer does
not belong in the runtime.  `CONTRAST` means the paper clarifies what not to
build.  This classification is about our use, not the quality of the work.

## What each paper contributes

| Local file | Decision | State or operation | Consequence for relaxed chip design |
| --- | --- | --- | --- |
| `delon_rabin_gousseau_circular_transport.pdf` | **CORE** | Circular CDF imbalance modulo one constant; the constant is a weighted median | The visible upper/lower gain reversal can be tested as a circular-cut ambiguity in linear time.  This fixes a global cut; it does **not** recover cell ownership. |
| `singer_angular_synchronization.pdf` | **CORE** | Pairwise `U(1)` relations synchronized by one Hermitian eigenspace | A global row chart can be assembled from local relative phases without trying row assignments.  The SDP alternative is not admitted. |
| `bandeira_singer_spielman_connection_cheeger.pdf` | **CORE** | Connection Laplacian and a consistency guarantee | Gives the right operator and a certificate for a phase-bearing graph rather than a scalar affinity graph. |
| `gao_brodzki_mukherjee_synchronization_geometry.pdf` | **CORE** | Synchronization as a flat bundle; cycle holonomy | Nonzero holonomy identifies phase evidence that cannot be made globally consistent.  It is a diagnostic before unrelaxation, not an HPWL score. |
| `hansen_ghrist_spectral_cellular_sheaves.pdf` | **CORE** | Stalks, restriction maps, sheaf Laplacian, sheaf sparsification | Neighbor relations need not be invertible rotations.  Capacity, phase, and boundary moment can live in a small heterogeneous stalk. |
| `dacunto_sheaf_signal_processing.pdf` | **PARTIAL** | Low-dimensional representation sheaves and natural transforms that intertwine Laplacians | Supports compressing stalk dimension while preserving a fixed filter and exact lifting relation.  The 2026 manuscript is a recent preprint; its greedy sampling-set routine is rejected. |
| `cloninger_connection_resistance.pdf` | **CORE** | Connection effective resistance and path signatures carrying rotations | Supplies a phase-aware importance measure for sparsifying/coarsening connection edges. |
| `singer_wu_vector_diffusion_maps.pdf` | **CORE** | Diffusion of local transformations instead of scalar heat | The initial conditioner should diffuse transported charts, not blur row labels as scalars. |
| `dorfler_bullo_kron_reduction.pdf` | **CORE** | Schur/Kron elimination preserving boundary response | Coarse levels may remove interior variables while keeping their exact boundary action.  Fill must then be sparsified once, not greedily pruned. |
| `shuman_graph_pyramid.pdf` | **CORE** | Graph reduction plus a prediction residual with perfect reconstruction | Never coarse phase alone: keep the detail that makes synthesis exact. |
| `sweldens_lifting_scheme.pdf` | **CORE** | In-place lifting; inverse is the same finite operations in reverse with signs changed | Gives the algebraic shape of relaxation/unrelaxation on irregular supports without an outer descent. |
| `spantini_low_dimensional_couplings.pdf` | **PARTIAL** | Sparse triangular maps implied by conditional independence; exact composition of local maps | Strong evidence that a global coupling may factor into bounded-dimensional local maps.  The paper's variational construction is not admitted as our runtime. |
| `aji_mceliece_generalized_distributive_law.pdf` | **CORE** | Exact semiring messages on junction trees | Capacity and separator summaries can be solved in two causal passes when the separator width is bounded. |
| `kschischang_factor_graphs.pdf` | **CORE** | Sum-product on cycle-free factor graphs | Same constructive lesson: complexity follows separator state, not the number of full assignments.  Loopy repeated belief propagation is not proposed. |
| `pachauri_permutation_synchronization.pdf` | **PARTIAL** | One spectral multiway-consistency solve | Useful only if a separator carries a very small label set.  Full row permutations and Hungarian rounding are too large for the desired representation. |
| `evans_matsen_tree_kr.pdf` | **PARTIAL** | Signed subtree mass imbalance on a tree | An exact linear-time mass flow, but it returns no per-cell identity map. |
| `le_tree_sliced_wasserstein.pdf` | **PARTIAL** | Tree-Wasserstein distance from subtree imbalances | Useful as a capacity channel.  Averaging many sampled trees and treating the result as identity transport are rejected. |
| `carlier_knothe_brenier.pdf` | **PARTIAL** | Conditional triangular transport | Justifies coarse-to-fine conditional coordinates.  The continuation path used to approach the limit is not the proposed algorithm. |
| `coifman_maggioni_diffusion_wavelets.pdf` | **PARTIAL** | Multiscale bases for powers of a diffusion operator | Explains dimension sparsification, but thresholded/QR construction is secondary and discarded detail must be retained by a lifting residual. |
| `loukas_graph_reduction.pdf` | **PARTIAL** | Reduction guarantees for graph operators | Keep the approximation criterion; reject the greedy sequence of local contractions. |
| `chazal_guibas_oudot_skraba_persistence_clustering.pdf` | **PARTIAL** | Persistent merge structure separates durable basins from small local features | A possible fixed diagnostic for phase-boundary scale.  It is not a row decoder and is not on the first implementation path. |
| `merigot_multiscale_optimal_transport.pdf` | **CONTRAST** | Repeated Lloyd simplification and a convex solve at every scale | This is precisely the attractive but iterative multiscale architecture we should not reproduce. |

## Sources

- Julie Delon, Julien Rabin, and Yann Gousseau,
  [Transportation Distances on the Circle and Applications](https://arxiv.org/abs/0906.5499).
- Afonso S. Bandeira, Amit Singer, and Daniel A. Spielman,
  [A Cheeger Inequality for the Graph Connection Laplacian](https://arxiv.org/abs/1204.3873).
- Amit Singer,
  [Angular Synchronization by Eigenvectors and Semidefinite Programming](https://arxiv.org/abs/0905.3174).
- Tingran Gao, Jacek Brodzki, and Sayan Mukherjee,
  [The Geometry of Synchronization Problems and Learning Group Actions](https://arxiv.org/abs/1610.09051).
- Jakob Hansen and Robert Ghrist,
  [Toward a Spectral Theory of Cellular Sheaves](https://arxiv.org/abs/1808.01513).
- Gabriele D'Acunto, Leonardo Di Nino, Paolo Di Lorenzo, and Sergio Barbarossa,
  [Sheaf-theoretic Signal Processing on Graphs](https://arxiv.org/abs/2608.01318).
- Alexander Cloninger, Gal Mishne, Andreas Oslandsbotn, Sawyer Jack Robertson,
  Zhengchao Wan, and Yusu Wang,
  [Random Walks, Conductance, and Resistance for the Connection Graph Laplacian](https://arxiv.org/abs/2308.09690).
- Amit Singer and Hau-Tieng Wu,
  [Vector Diffusion Maps and the Connection Laplacian](https://arxiv.org/abs/1102.0075).
- Florian Dörfler and Francesco Bullo,
  [Kron Reduction of Graphs with Applications to Electrical Networks](https://arxiv.org/abs/1102.2950).
- David I. Shuman, Mohammad Javad Faraji, and Pierre Vandergheynst,
  [A Multiscale Pyramid Transform for Graph Signals](https://arxiv.org/abs/1308.4942).
- Wim Sweldens,
  [The Lifting Scheme: A Construction of Second Generation Wavelets](https://cm-bell-labs.github.io/who/wim/papers/lift2.pdf).
- Alessio Spantini, Daniele Bigoni, and Youssef Marzouk,
  [Inference via Low-Dimensional Couplings](https://arxiv.org/abs/1703.06131).
- Srinivas M. Aji and Robert J. McEliece,
  [The Generalized Distributive Law](https://authors.library.caltech.edu/records/sw1pm-bwj40).
- Frank R. Kschischang, Brendan J. Frey, and Hans-Andrea Loeliger,
  [Factor Graphs and the Sum-Product Algorithm](https://www.mit.edu/~6.454/www_fall_2002/lizhong/factorgraph.pdf).
- Deepti Pachauri, Risi Kondor, and Vikas Singh,
  [Solving the Multi-way Matching Problem by Permutation Synchronization](https://proceedings.neurips.cc/paper/2013/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html).
- Steven N. Evans and Frederick A. Matsen,
  [The Phylogenetic Kantorovich-Rubinstein Metric for Environmental Sequence Samples](https://arxiv.org/abs/1005.1699).
- Tam Le, Makoto Yamada, Kenji Fukumizu, and Marco Cuturi,
  [Tree-Sliced Variants of Wasserstein Distances](https://arxiv.org/abs/1902.00342).
- Guillaume Carlier, Alfred Galichon, and Filippo Santambrogio,
  [From Knothe's Transport to Brenier's Map and a Continuation Method for Optimal Transport](https://arxiv.org/abs/0810.4153).
- Ronald R. Coifman and Mauro Maggioni,
  [Diffusion Wavelets](https://www.math.ucdavis.edu/~saito/data/diffgeomharm/ACHA-Special/diffusionwavelets.pdf).
- Andreas Loukas,
  [Graph Reduction with Spectral and Cut Guarantees](https://arxiv.org/abs/1808.10650).
- Frédéric Chazal, Leonidas J. Guibas, Steve Y. Oudot, and Primoz Skraba,
  [Persistence-Based Clustering in Riemannian Manifolds](https://doi.org/10.1145/2535927).
- Quentin Mérigot,
  [A Multiscale Approach to Optimal Transport](https://doi.org/10.1111/j.1467-8659.2011.02032.x).

The byte-level manifest is in `SHA256SUMS`.
