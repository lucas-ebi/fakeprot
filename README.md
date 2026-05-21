# FakeProt

**A stochastic simulator for synthetic protein families.**

FakeProt is a stochastic simulator for generating synthetic protein families with
known evolutionary histories. It produces multiple-sequence alignments, a gene
tree, a species cladogram, ortholog-group assignments, ancestral sequences, and
per-column physicochemical annotations. The intended use is methodological
benchmarking: aligners, tree inference procedures, orthology pipelines,
ancestral-sequence methods, and downstream comparative-genomics workflows can be
tested against data for which the generating history is known.

## Overview

Simulation studies in molecular evolution often require sequence families whose
ground truth is explicit: the species tree, gene tree, duplication history,
ortholog groups, ancestral states, and column-wise constraints should all be
recoverable. FakeProt implements a discrete, event-driven generator for protein
families. Starting from a single ancestral protein, the simulator repeatedly
selects an extant species, optionally creates a paralog by gene duplication
(Ohno, 1970), and then speciates the selected lineage into two daughter species. Sequence evolution
along every gene-tree edge combines WAG-derived amino-acid substitution
probabilities (Whelan & Goldman, 2001), gamma-distributed site-rate
heterogeneity (Yang, 1994), a TKF92-compatible fragment indel model
(Thorne, Kishino & Felsenstein, 1992), and probabilistic physicochemical
constraints based on overlapping amino-acid property classes (Taylor, 1986).
Insertions are represented as new alignment columns, while deletions are
represented as gaps, so every emitted sequence remains in a common alignment.

FakeProt is not intended to be a fully calibrated biological model of protein
evolution. Its substitution and indel components follow continuous-time
formulations (WAG CTMC, TKF92), but the physicochemical constraint scheme and
rate parameters are not estimated from empirical data. It is a transparent
benchmark generator whose parameters control sequence-family size, root length,
indel frequency, site-rate heterogeneity, and the target number of ortholog
groups.

## Installation

FakeProt requires Python 3.10 or later.

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# 50 extant sequences, root length 200, three ortholog groups, fixed seed
fakeprot 50 200 -n 3 -r 42 -o my_family
```

The command above writes a set of FASTA, tree, CSV, and JSON files using
`my_family` as the filename prefix.

## Command-Line Interface

```text
fakeprot SIZE LENGTH [options]
```

| Argument | Meaning |
| --- | --- |
| `SIZE` | Minimum number of extant, leaf-level sequences to generate. The final number can exceed this value because one simulation step may add multiple leaves. |
| `LENGTH` | Root sequence length before lineage-specific insertions. |

| Option | Default | Meaning |
| --- | --- | --- |
| `-o`, `--out` | `fakeprot_out` | Prefix for all output files. |
| `-d`, `--mu` | `0.05` | TKF91/92 deletion rate $\mu$ per unit branch length; per-edge intensity $\mu \cdot t$. |
| `-i`, `--lam` | `0.001` | TKF91/92 insertion rate $\lambda$ per unit branch length; per-edge intensity $\lambda \cdot t$. |
| `--q` | `0.5` | TKF92 fragment extension probability $q$; geometric run-length parameter. Mean run length $= 1/(1-q)$. |
| `-n`, `--n-orthologs` | `1` | Target number of ortholog-group anchors. |
| `-a`, `--alpha` | `0.75` | Shape parameter $\alpha$ for the gamma site-rate model. Scale is fixed to $1/\alpha$ so that the mean rate equals 1, following the standard convention introduced by Yang (1994). |
| `-b`, `--branch-length` | `0.05` | Expected substitutions per site on a branch of average rate. Controls overall sequence divergence; also scales the per-edge intensities $\mu \cdot t$ and $\lambda \cdot t$. |
| `--dup-boost-factor` | `2.0` | Branch-length multiplier applied to a boosted duplicate edge. |
| `--dup-boost-decay` | `3.0` | E-folding distance in speciation steps for the post-duplication rate burst. Set to 0 for a single-edge burst only. |
| `--branch-cv` | `0.4` | Coefficient of variation for per-branch stochastic rate noise (σ_log ≈ 0.39). Each speciation edge draws its length from a mean-preserving log-normal; `0` gives deterministic (equal-length) branches. Default chosen as the centre of the empirical range reported by Drummond et al. (2006). |
| `-r`, `--seed` | `None` | Random seed for reproducible simulations. |
| `-f`, `--msa-format` | `fasta` | Alignment format: `fasta`, `clustal`, `nexus`, `phylip`, or `stockholm`. |
| `-t`, `--tree-format` | `newick` | Tree format: `newick`, `nexus`, `nexml`, `phyloxml`, or `cdao`. |

## Generative Model

### Notation

Let $A$ be the set of 20 canonical amino acids and let $\pi_a$ be the WAG
equilibrium frequency of amino acid $a \in A$, and $s_{ab}$ the symmetric WAG
exchangeability between $a$ and $b$ (Whelan & Goldman, 2001; values from PAML
`dat/wag.dat`). The WAG rate matrix has off-diagonal entries $Q_{ab} = s_{ab}\,\pi_b$
($a \neq b$), with diagonal set so rows sum to zero. FakeProt normalises these
off-diagonal rates to obtain the row-stochastic Gillespie jump chain $W$:

```math
W_{ab} = \frac{Q_{ab}}{\displaystyle\sum_{k \neq a} Q_{ak}}
       = \frac{s_{ab}\,\pi_b}{\displaystyle\sum_{k \neq a} s_{ak}\,\pi_k},
\qquad a \neq b,
```

where $W_{ab}$ is the conditional probability of proposing residue $b$ given
that residue $a$ undergoes a change. The diagonal is zero by construction.

Let $C$ be the set of physicochemical classes used by the simulator. Each class
$c \in C$ corresponds to a subset $A_c \subset A$; classes are allowed to
overlap, following the Venn-diagram classification of amino-acid properties
introduced by Taylor (1986). The class prior is proportional to the WAG
background mass of its member residues:

```math
\omega_c =
\frac{\sum_{a \in A_c} \pi_a}
     {\sum_{d \in C} \sum_{a \in A_d} \pi_a}.
```

The conditional amino-acid distribution inside class $c$ is

```math
P(X=a \mid c) =
\frac{\pi_a 1[a \in A_c]}{\sum_{b \in A_c} \pi_b}.
```

### Site-Rate Heterogeneity

Given root length $L$, FakeProt assigns a site-specific relative rate
$r_i$ to every alignment site. The first site is fixed at $r_1 = 0$ to
preserve the initial methionine. For the remaining $L - 1$ sites, rates
are drawn independently from

```math
r_{i} \;\sim\; \Gamma\!\left(\alpha,\,\frac{1}{\alpha}\right),
\qquad i = 2,\ldots,L,
```

so the mean rate equals 1, following the standard phylogenetic convention
(Yang, 1994). These rate values are then reordered by a local smoothing procedure
(`wave_shuffle`). Each rate $r_i$ is first mapped to a normalised substitution
probability

```math
\hat{p}_i =
\frac{1 - \exp(-t\,r_i)}{\max_j\bigl(1 - \exp(-t\,r_j)\bigr)},
```

so that all keys lie in $[0, 1]$ regardless of $t$. Starting from a random
site, each next site is chosen from the remaining values with probability
proportional to

```math
\max\bigl(0,\; 1 - |\hat{p}_{\mathrm{previous}} - \hat{p}_{\mathrm{candidate}}|\bigr).
```

The original gamma-derived rates are returned in the order determined by this
traversal, so the marginal rate distribution is unchanged. Using normalised
substitution probabilities as sort keys ensures the kernel has full dynamic range
for any $t$ and places comparisons in a scale where differences correspond
directly to differences in evolutionary variability — encouraging neighboring
sites to have similar mutability, an autocorrelation-along-sequence pattern
motivated by hidden Markov models of rate variation
(Felsenstein & Churchill, 1996; see also Yang, 1995).

### Root Sequence

The ancestral sequence begins with methionine:

```math
X_1 = \mathrm{M}, \qquad c_1 = \text{With sulfur}, \qquad r_1 = 0.
```

For every other site $i$, FakeProt first decides whether the site has a
physicochemical constraint using the same functional form as the substitution
survival probability (see below), with $t$ acting as a tuning parameter for the
proportion of constrained sites rather than as an actual edge length:

```math
P(c_i \neq \varnothing) = \exp(-t\,r_i),
\qquad
P(c_i = \varnothing) = 1 - \exp(-t\,r_i).
```

If constrained, the class is drawn from
$\text{Categorical}(\omega)$ and the amino acid is drawn from the
class-conditional WAG background distribution (using the overlapping class
scheme of Taylor, 1986). If unconstrained, the amino acid is drawn directly
from the WAG background frequencies $\pi$.

### Substitution Model Along an Edge

For each child sequence, FakeProt iterates through the parent sequence from left
to right. At a non-gap site with parent residue $x_i$, site rate $r_i$, and
constraint $c_i$, deletion is considered first:

```math
P(\mathrm{delete}\ i) = 1 - \exp(-r_i\,\mu\,t),
```

where $\mu$ is `--mu` and $t$ is the per-edge branch length. This is
the exact CTMC survival complement, consistent with the substitution formula
below, and bounded in $[0, 1]$ for all $r_i$.

If the site is not deleted, the substitution probability is derived from the
standard continuous-time Markov chain survival formula for a branch of length $t$
at site rate $r_i$:

```math
P(\text{substitution at site } i) = 1 - \exp(-t\,r_i).
```

If there is no physicochemical constraint, the new residue is sampled using the
WAG conditional substitution matrix $W$ (Whelan & Goldman, 2001):

```math
P(X_i' = x_i) = \exp(-t\,r_i),
```

```math
P(X_i' = a) = \bigl(1 - \exp(-t\,r_i)\bigr)\,W_{x_i a}, \qquad a \neq x_i.
```

If the site is constrained to physicochemical class $c_i$, the substitution
proposal is restricted to residues in $A_{c_i}$. Let

```math
T_{x_i,c_i} = \sum_{a \in A_{c_i}} W_{x_i a}.
```

(Since $W$ has zero diagonal entries, the sum implicitly excludes
self-substitution even when $x_i \in A_{c_i}$.)

When $T_{x_i,c_i} > 0$, the constrained substitution distribution is

```math
P(X_i' = x_i) = \exp(-t\,r_i),
```

```math
P(X_i' = a) =
\bigl(1 - \exp(-t\,r_i)\bigr)\,\frac{W_{x_i a}}{T_{x_i,c_i}},
\qquad a \in A_{c_i}.
```

If $T_{x_i,c_i} = 0$, the residue is retained. In all cases, gaps are treated as
alignment characters and are carried forward unless explicitly filled by the gap
process below.

### Insertions, Deletions, and Gap Filling

Gaps in the final alignment arise from two distinct processes with separate
per-branch Poisson intensities $\mu\,t$ (deletion) and $\lambda\,t$
(insertion), where $\mu$ (`--mu`) and $\lambda$ (`--lam`) are the TKF91/92 rate parameters.

**Deletions** accumulate along each lineage's path from the root. For a Yule
(pure-birth) tree with $n$ leaves (Yule, 1925; Aldous, 2001) the expected leaf
depth is $2(H_n - 1) \approx 2\ln n$, giving expected deletion gap fraction

```math
f_d \;\approx\; \mu\,t\cdot 2\ln n.
```

**Insertions** in any one lineage create a new alignment column that appears
as a gap in all other $n-1$ sequences. Each insertion event therefore has an
$n$-fold amplifying effect on total gap cells. The expected insertion gap
fraction is

```math
f_i \;\approx\; \frac{4n\,\lambda\,t}{1 + 4n\,\lambda\,t}.
```

These formulas simplify to the above form because the gamma rates $r_i$ have
mean 1 by construction (scale fixed to $1/\alpha$), so averaging over sites
contributes a factor of 1.

The likelihood that a gappy column with $k$ gaps was caused by a deletion
versus an insertion can be assessed via the subtree-size distribution of a
Yule tree, which is approximately $P(\text{size}=s) \propto 1/s$
(Aldous, 2001):

```math
\frac{P(\text{deletion} \mid k)}{P(\text{insertion} \mid k)}
= \frac{\mu}{\lambda} \cdot \frac{n - k}{k}.
```

Columns with few gaps ($k \ll n/2$) are predominantly deletions; columns with
many gaps ($k \gg n/2$) are predominantly insertions. The crossover is at
$k = n\,\mu/(\mu + \lambda)$.

The defaults are design choices tuned to produce realistic scaling behaviour
rather than values read directly from any single empirical study:

```math
\mu = 0.05, \qquad \lambda = 0.001, \qquad q = 0.5,
```

giving per-edge intensities $\mu\,t$ and $\lambda\,t$ at branch length $t$
(default 0.05). The deletion coefficient (5% of branch length) and insertion
coefficient (0.1%) are not equal because each insertion in a single lineage creates a gap in every other
sequence, amplifying its effect on total gap content by a factor of
approximately $n$. As a result, gap content grows naturally with the number of
sequences: deletion gaps accumulate as $O(\log n)$ (from tree depth) and
insertion gaps as $O(n)$ (from the alignment-wide amplification). This matches
the empirical observation, well known to practitioners working with large
protein-family databases such as Pfam (Finn et al., 2006; Mistry et al.,
2021), that larger and more diverged families are substantially gappier.

After each non-gap site — whether the site survives or is deleted — FakeProt
may introduce an insertion fragment using the TKF92 birth/death model
(Thorne, Kishino & Felsenstein, 1992). This is a fragment generalisation of
TKF91 (Thorne, Kishino & Felsenstein, 1991) in which insertions arrive as
geometric-length runs rather than single residues.

Let $\mu_i = r_i\,\mu\,t$ be the per-site death intensity and
$r_{f,i} = r_i\,\lambda\,t\,(1-q)$ the per-site fragment birth intensity, where
$q$ (`--q`, default 0.5) is the fragment extension probability. The
TKF92 probability that at least one fragment is inserted after site $i$ is

```math
\beta_i =
\frac{r_{f,i}\bigl(1 - \exp(-(\mu_i - r_{f,i}))\bigr)}
     {\mu_i - r_{f,i}\exp(-(\mu_i - r_{f,i}))},
```

with the limit $\beta_i = r_{f,i}/(1 + r_{f,i})$ when $\mu_i = r_{f,i}$.
The process is subcritical (sequences do not grow without bound) whenever
$r_{f,i} < \mu_i$, i.e. $\lambda\,(1-q) < \mu$, which holds under all
reasonable defaults. Given that a fragment starts, each additional residue is
appended with probability $q$:

```math
P(\text{extend by one more} \mid \text{already inserting}) = q.
```

This gives a geometrically distributed run length with mean $1/(1-q) = 2$.
Crucially, the expected total residues inserted after a rate-1 site is
$\beta \cdot 1/(1-q) \approx \lambda\,t$, so `--lam` directly controls the
per-unit-rate residue insertion intensity regardless of $q$.

Deletions do not suppress fragment birth: a deleted position can still birth
a fragment (a "ghost lineage" in TKF terminology), consistent with the
continuous-time formulation.

A geometric run-length distribution is a tractable approximation to the
Zipfian (power-law) distribution empirically observed in real protein
alignments, where the probability of a gap of length $L$ decreases
approximately as $L^{-1.7}$ (Benner, Cohen & Gonnet, 1993;
Chang & Benner, 2004).

Inserted residues are sampled from the WAG background distribution, assigned rate
$1.0$, and assigned no physicochemical class. When an insertion creates a new
alignment site, FakeProt inserts a gap in every previously generated row so
that all sequences remain aligned.

Existing gap runs can also be filled in descendant lineages. The first position
in a gap run is filled with the TKF92 probability $\beta_i$ (same formula as
above, using the gap slot's own rate $r_i$); each subsequent position is filled
with probability $q$, independently. Filling stops after the first
failed attempt. Filled residues are sampled either from the WAG background
distribution or, when the site has a physicochemical class, from the
corresponding class-conditional distribution.

## Species, Gene Trees, and Orthology

FakeProt maintains two directed graphs during simulation: a species tree and a
sequence/gene tree. The species tree contains extant and ancestral species nodes.
The gene tree contains ancestral, duplicated, and extant sequence nodes. The
orthology and paralogy terminology used throughout follows the definitions
introduced by Fitch (1970) and the modern formulation reviewed by
Koonin (2005): orthologs descend from a common ancestor via speciation, while
paralogs descend via gene duplication.

At each simulation step:

1. One extant species is selected uniformly at random.
2. A gene duplication may occur inside that species.
3. The selected species is split into two daughter species.
4. Each paralog in the selected species independently evolves into one child
   sequence in each daughter species.

The process stops when the number of extant sequence leaves is at least `SIZE`.

### Gene Duplication

Let $K$ be the requested number of ortholog groups, $m$ the current number of
ortholog anchors, and $s$ the number of extant species. If $m < K$, duplication
is attempted with probability

```math
P(\mathrm{duplication}) = 2^{-(K-m)/s}.
```

When a duplication occurs, one paralog in the selected species is copied and
mutated. The new copy becomes an ortholog-group anchor. Its rate profile is
redrawn using a perturbed shape parameter

```math
\alpha' \sim N(\alpha, 1) \mid \alpha' \geq 1,
```

with scale fixed to $1/\alpha'$ as usual (Yang, 1994). The rank order of the
parent rates is preserved: sites that were slow or fast in the parent remain
relatively slow or fast in the duplicate, but the numerical rate scale may
shift.

For duplicated sequences, physicochemical constraints may also be relaxed or
changed. Let $t_e$ denote the branch length of the creation edge — equal to
$t$ × `dup_boost_factor` for a boosted duplication, and $t$ otherwise. Let $r_i'$ be the duplicate's redrawn site rate at position $i$.
If a parent site has class $c$, the duplicate first retains any constraint with
probability $\exp(-t_e\,r_i')$. Conditional on retaining a constraint, the class
is preserved with probability $\exp(-t_e\,r_i')$ or changed according to a
class-level transition matrix $M$ derived by averaging WAG probabilities between
the residues in each pair of physicochemical classes:

```math
P(c_i' = c \mid c_i' \neq \varnothing, c_i=c) = \exp(-t_e\,r_i'),
```

```math
P(c_i' = d \mid c_i' \neq \varnothing, c_i=c) =
\bigl(1 - \exp(-t_e\,r_i')\bigr)\,M_{cd},
\qquad d \neq c.
```

If the retained or newly chosen class does not contain the current residue,
FakeProt resamples the residue using the WAG probabilities restricted to the new
class. Sites with no parent constraint ($c_i = \varnothing$) may also *acquire*
a new class during duplication, sampled from the background prior
$\text{Categorical}(\omega)$ with the same probability $\exp(-t_e\,r_i')$; if
the current residue is not a member of the newly assigned class, it is likewise
resampled.

Each duplication modulates the new copy's divergence rate. The duplicate's
creation edge and its downstream lineage carry an elevated branch-length
multiplier that decays exponentially with each subsequent speciation event:

```math
m(d) =
1 + (\texttt{dup\_boost\_factor} - 1)
\exp\!\Bigl(-\tfrac{d}{\texttt{dup\_boost\_decay}}\Bigr),
```

where $d$ is the number of speciation steps separating the edge from the
duplication node. At $d = 0$ (the duplicate's creation edge) the multiplier
equals `dup_boost_factor` (default 2.0). At $d =$ `dup_boost_decay`
(default 3.0) it has decayed by $1/e$. Setting `--dup-boost-decay 0` restricts
the boost to the duplicate's creation edge only. This models the transient rate
asymmetry between paralogs observed in the period immediately following
duplication (Ohno, 1970; Conant & Wagner, 2003; Lynch & Conery, 2000;
Kellis et al., 2004). The boost is directly visible in the output gene tree: edges in the
duplicate's subtree carry their elevated generative branch length, decaying
toward the global rate with each subsequent speciation event.

### Speciation

Speciation replaces one extant species by two daughter species. Every paralog in
the parent species gives rise to one independently mutated copy in each daughter.
Thus a speciation event branches both the species tree and, for every inherited
paralog, the gene tree.

### Ortholog Groups

Ortholog groups are defined by anchor nodes (Fitch, 1970; Koonin, 2005): the
root sequence and every successful duplication copy. A leaf belongs to anchor
$a_j$ if it descends from $a_j$ without crossing another ortholog anchor:

```math
OG_j =
\{\ell :
a_j \rightsquigarrow \ell
\ \mathrm{and\ no\ other\ anchor\ lies\ on\ the\ path}\}.
```

This definition ensures that every extant sequence is assigned to exactly one
ortholog group in the emitted CSV file.

### Branch Lengths

Gene-tree edge weights are the generative branch-length values recorded at the
time of evolution. For each speciation edge the realised length is drawn
independently from a log-normal with mean $t$ (or $t \cdot m(d)$ inside a
boosted duplicate's subtree) and coefficient of variation `--branch-cv`; with
`--branch-cv 0` every edge uses the mean exactly. Because these values are
stored during simulation rather than derived from the alignment after the fact,
they constitute true ground-truth branch lengths for the simulated family.
The species cladogram is emitted without branch lengths.

## Outputs

For output prefix `PREFIX`, FakeProt writes:

| File | Contents |
| --- | --- |
| `PREFIX_all_sequences.<msa-format>` | All generated sequences, including ancestral and internal sequence nodes. |
| `PREFIX_current_sequences.<msa-format>` | Extant leaf sequences only. |
| `PREFIX_gene_tree.<tree-format>` | Gene tree with generative (ground-truth) branch lengths. |
| `PREFIX_species_cladogram.<tree-format>` | Species cladogram without branch lengths. |
| `PREFIX_ortholog_groups.json` | Mapping from extant sequence identifiers to ortholog-group labels (`{"sp1_seq1": "OG_A", ...}`). |
| `PREFIX_OG_A.<msa-format>`, `PREFIX_OG_B.<msa-format>`, ... | Per-ortholog-group alignments, emitted when `--n-orthologs > 1`. |
| `PREFIX_physicochemical_groups.json` | Per-column amino-acid frequencies and physicochemical class annotations for each ortholog group. |
| `PREFIX_run_info.json` | Version, timestamp, parameters, and random seed for reproducibility. |

## Implementation Notes

Sequences are stored in three aligned NumPy arrays: amino-acid/gap codes,
site-rate values, and physicochemical-class codes. Insertions are handled by
adding columns to the global alignment store and filling earlier rows with gap
sentinels. This makes output generation straightforward because every internal
and extant sequence is already represented in a shared alignment.

Trees and traversal operations are implemented with NetworkX. Sequence and tree
serialization use Biopython, and tabular outputs use pandas.

## Assumptions and Limitations

FakeProt is a benchmark simulator rather than an inference-calibrated biological
model. In particular:

1. Physicochemical classes overlap (Taylor, 1986), so class priors are
   normalized over class masses rather than over a partition of amino acids.
2. The requested number of ortholog groups is a target, not a hard guarantee; if
   too few duplication opportunities occur before the leaf target is reached, the
   simulator emits a warning.
3. The requested size is a lower bound because one speciation event can add more
   than one sequence leaf.

These design choices make the simulator easy to inspect and useful for controlled
benchmarking, but they should be considered when interpreting results as
biological evolutionary histories.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use this simulator in your research, please cite the original software and the associated publication (if any). The original module was developed by **Lucas Carrijo de Oliveira** (<lucas@ebi.ac.uk>). For now, you may reference the repository directly:

```text
Lucas C. de Oliveira. FakeProt. 2018–2026.
https://github.com/lucas-ebi/fakeprot
```

A formal publication is in preparation; check the repository for updates.

### Substitution Model

> Whelan, S. and Goldman, N. (2001). A general empirical model of protein
> evolution derived from multiple protein families using a maximum-likelihood
> approach. *Molecular Biology and Evolution*, 18(5), 691–699.

### Site-Rate Heterogeneity and Rate Autocorrelation

> Yang, Z. (1994). Maximum likelihood phylogenetic estimation from DNA
> sequences with variable rates over sites: approximate methods.
> *Journal of Molecular Evolution*, 39(3), 306–314.
>
> Felsenstein, J. and Churchill, G. A. (1996). A Hidden Markov Model approach
> to variation among sites in rate of evolution. *Molecular Biology and
> Evolution*, 13(1), 93–104.
>
> Yang, Z. (1995). A space-time process model for the evolution of DNA
> sequences. *Genetics*, 139(2), 993–1005.

### Physicochemical Amino-Acid Classes

> Taylor, W. R. (1986). The classification of amino acid conservation.
> *Journal of Theoretical Biology*, 119(2), 205–218.

### Indel Processes and Gap Length Distributions

> Thorne, J. L., Kishino, H. and Felsenstein, J. (1991). An evolutionary
> model for maximum likelihood alignment of DNA sequences.
> *Journal of Molecular Evolution*, 33(2), 114–124.
>
> Thorne, J. L., Kishino, H. and Felsenstein, J. (1992). Inching toward
> reality: an improved likelihood model of sequence evolution.
> *Journal of Molecular Evolution*, 35(1), 3–16.
>
> Benner, S. A., Cohen, M. A. and Gonnet, G. H. (1993). Empirical and
> structural models for insertions and deletions in the divergent evolution
> of proteins. *Journal of Molecular Biology*, 229(4), 1065–1082.
>
> Chang, M. S. S. and Benner, S. A. (2004). Empirical analysis of protein
> insertions and deletions determining parameters for the correct placement
> of gaps in protein sequence alignments. *Journal of Molecular Biology*,
> 341(2), 617–631.

### Yule (Pure-Birth) Trees and Their Statistics

> Yule, G. U. (1925). A mathematical theory of evolution, based on the
> conclusions of Dr. J. C. Willis, F.R.S. *Philosophical Transactions of
> the Royal Society of London, Series B*, 213, 21–87.
>
> Aldous, D. J. (2001). Stochastic models and descriptive statistics for
> phylogenetic trees, from Yule to today. *Statistical Science*, 16(1), 23–34.

### Orthology and Paralogy

> Fitch, W. M. (1970). Distinguishing homologous from analogous proteins.
> *Systematic Zoology*, 19(2), 99–113.
>
> Koonin, E. V. (2005). Orthologs, paralogs, and evolutionary genomics.
> *Annual Review of Genetics*, 39, 309–338.

### Branch-Length Rate Variation

> Drummond, A. J., Ho, S. Y. W., Phillips, M. J. and Rambaut, A. (2006).
> Relaxed phylogenetics and dating with confidence. *PLoS Biology*, 4(5), e88.

### Gene Duplication and Post-Duplication Rate Asymmetry

> Ohno, S. (1970). *Evolution by Gene Duplication*. Springer-Verlag, Berlin.
>
> Conant, G. C. and Wagner, A. (2003). Asymmetric sequence divergence of
> duplicate genes. *Genome Research*, 13(9), 2052–2058.
>
> Lynch, M. and Conery, J. S. (2000). The evolutionary fate and consequences
> of duplicate genes. *Science*, 290(5494), 1151–1155.
>
> Kellis, M., Birren, B. W. and Lander, E. S. (2004). Proof and evolutionary
> analysis of ancient genome duplication in the yeast *Saccharomyces cerevisiae*.
> *Nature*, 428(6983), 617–624.

### Protein-Family Database

> Finn, R. D., Mistry, J., Schuster-Böckler, B., Griffiths-Jones, S.,
> Hollich, V., Lassmann, T., Moxon, S., Marshall, M., Khanna, A., Durbin, R.
> et al. (2006). Pfam: clans, web tools and services. *Nucleic Acids
> Research*, 34, D247–D251.
>
> Mistry, J., Chuguransky, S., Williams, L., Qureshi, M., Salazar, G. A.,
> Sonnhammer, E. L. L., Tosatto, S. C. E., Paladin, L., Raj, S.,
> Richardson, L. J., Finn, R. D. and Bateman, A. (2021). Pfam: the protein
> families database in 2021. *Nucleic Acids Research*, 49(D1), D412–D419.

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

Copyright (C) 2018 Lucas Carrijo de Oliveira.
