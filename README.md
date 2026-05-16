# FakeProt: a stochastic simulator for synthetic protein families

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
selects an extant species, optionally creates a paralog by gene duplication, and
then speciates the selected lineage into two daughter species. Sequence evolution
along every gene-tree edge combines WAG-derived amino-acid substitution
probabilities (Whelan & Goldman, 2001), gamma-distributed site-rate
heterogeneity (Yang, 1994), indel events, and probabilistic physicochemical
constraints based on overlapping amino-acid property classes (Taylor, 1986).
Insertions are represented as new alignment columns, while deletions are
represented as gaps, so every emitted sequence remains in a common alignment.

FakeProt is not intended to be a fully calibrated continuous-time model of
protein evolution. Instead, it is a transparent benchmark generator whose
parameters control sequence-family size, root length, indel frequency, site-rate
heterogeneity, and the target number of ortholog groups.

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
| `-d`, `--p-del` | derived | Per-branch per-site deletion probability $p_d$. Defaults to 5% of the mean site substitution rate (see below). |
| `-i`, `--p-ins` | derived | Per-branch per-site insertion probability $p_i$. Defaults to 0.1% of the mean site substitution rate (see below). |
| `-n`, `--n-orthologs` | `1` | Target number of ortholog-group anchors. |
| `-a`, `--shape` | `0.75` | Shape parameter $\alpha$ for the gamma site-rate model. Scale is fixed to $1/\alpha$ so that the mean rate equals 1, following the standard convention introduced by Yang (1994). |
| `-r`, `--seed` | `None` | Random seed for reproducible simulations. |
| `-f`, `--msa-format` | `fasta` | Alignment format: `fasta`, `clustal`, `nexus`, `phylip`, or `stockholm`. |
| `-t`, `--tree-format` | `newick` | Tree format: `newick`, `nexus`, `nexml`, `phyloxml`, or `cdao`. |

## Generative Model

### Notation

Let $A$ be the set of 20 canonical amino acids and let $\pi_a$ be the WAG
background frequency of amino acid $a \in A$ (Whelan & Goldman, 2001). FakeProt
stores the WAG empirical off-diagonal substitution probabilities in a
row-stochastic matrix $W$, where $W_{ab}$ is the conditional probability of
proposing residue $b$ from residue $a$, with $a \neq b$.

Let $C$ be the set of physicochemical classes used by the simulator. Each class
$c \in C$ corresponds to a subset $S_c \subset A$; classes are allowed to
overlap, following the Venn-diagram classification of amino-acid properties
introduced by Taylor (1986). The class prior is proportional to the WAG
background mass of its member residues:

```math
\omega_c =
\frac{\sum_{a \in S_c} \pi_a}
     {\sum_{d \in C} \sum_{a \in S_d} \pi_a}.
```

The conditional amino-acid distribution inside class $c$ is

```math
P(X=a \mid c) =
\frac{\pi_a 1[a \in S_c]}{\sum_{b \in S_c} \pi_b}.
```

### Site-Rate Heterogeneity

Given root length $L$, FakeProt assigns a site-specific mutation probability
$r_i$ to every alignment site, following the discrete-gamma approach of
Yang (1994). The first site is fixed at $r_1 = 0$ to preserve the initial
methionine. For the remaining $L - 1$ sites, rates are drawn from a
$\Gamma(\alpha,\,\beta)$ distribution with $\beta = 1/\alpha$ (so the mean rate
equals 1, the standard phylogenetic convention). Specifically, $L-1$ evenly
spaced quantile values are taken between the 1st and 99th percentiles:

```math
(x_1,\ldots,x_{L-1}) =
\text{linspace}
\left(
F^{-1}_{\Gamma(\alpha,\,1/\alpha)}(0.01),\;
F^{-1}_{\Gamma(\alpha,\,1/\alpha)}(0.99),\;
L-1
\right),
```

and then normalised to $[0,1]$ by the 99th-percentile value $x_{L-1}$:

```math
r_{j+1} = \frac{x_j}{x_{L-1}},
\qquad j = 1,\ldots,L-1.
```

These rate values are then reordered by a local smoothing procedure
(`wave_shuffle`): starting from a random site, each next site is chosen from the
remaining values with probability proportional to

```math
\max(0, 1 - |r_{\mathrm{previous}} - r_{\mathrm{candidate}}|).
```

This preserves the marginal set of gamma-derived rates while encouraging
neighboring sites to have similar mutability, an autocorrelation-along-sequence
pattern motivated by hidden Markov models of rate variation
(Felsenstein & Churchill, 1996; see also Yang, 1995).

### Root Sequence

The ancestral sequence begins with methionine:

```math
X_1 = \mathrm{M}, \qquad c_1 = \text{With sulfur}, \qquad r_1 = 0.
```

For every other site $i$, FakeProt first decides whether the site has a
physicochemical constraint:

```math
P(c_i \neq \varnothing) = 1 - r_i,
\qquad
P(c_i = \varnothing) = r_i.
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
P(\mathrm{delete}\ i) = p_d\,r_i.
```

If the site is not deleted and has no physicochemical constraint, it is retained
with probability $1 - r_i$; otherwise a new residue is sampled using the WAG row
for the current residue (Whelan & Goldman, 2001):

```math
P(X_i' = x_i) = 1 - r_i,
```

```math
P(X_i' = a) = r_i W_{x_i a}, \qquad a \neq x_i.
```

If the site is constrained to physicochemical class $c_i$, the mutation proposal
is restricted to residues in $S_{c_i}$. Let

```math
T_{x_i,c_i} = \sum_{a \in S_{c_i}} W_{x_i a}.
```

When $T_{x_i,c_i} > 0$, the constrained substitution distribution is

```math
P(X_i' = x_i) = 1 - r_i,
```

```math
P(X_i' = a) =
r_i \frac{W_{x_i a}}{T_{x_i,c_i}},
\qquad a \in S_{c_i}.
```

If $T_{x_i,c_i} = 0$, the residue is retained. In all cases, gaps are treated as
alignment characters and are carried forward unless explicitly filled by the gap
process below.

### Insertions, Deletions, and Gap Filling

Gaps in the final alignment arise from two distinct processes with separate
per-branch, per-site probabilities $p_d$ (deletion) and $p_i$ (insertion).

**Deletions** accumulate along each lineage's path from the root. For a Yule
(pure-birth) tree with $n$ leaves (Yule, 1925; Aldous, 2001) the expected leaf
depth is $2(H_n - 1) \approx 2\ln n$, giving expected deletion gap fraction

```math
f_d \;\approx\; \bar{r}\,p_d\,2\ln n.
```

**Insertions** in any one lineage create a new alignment column that appears
as a gap in all other $n-1$ sequences. Each insertion event therefore has an
$n$-fold amplifying effect on total gap cells. The expected insertion gap
fraction is

```math
f_i \;\approx\; \frac{4n\,\bar{r}\,p_i}{1 + 4n\,\bar{r}\,p_i}.
```

The likelihood that a gappy column with $k$ gaps was caused by a deletion
versus an insertion can be assessed via the subtree-size distribution of a
Yule tree, which is approximately $P(\text{size}=s) \propto 1/s$
(Aldous, 2001):

```math
\frac{P(\text{deletion} \mid k)}{P(\text{insertion} \mid k)}
= \frac{p_d}{p_i} \cdot \frac{n - k}{k}.
```

Columns with few gaps ($k \ll n/2$) are predominantly deletions; columns with
many gaps ($k \gg n/2$) are predominantly insertions. The crossover is at
$k = n\,p_d/(p_d + p_i)$.

Default values are calibrated to the mean per-branch substitution rate
$\bar{r}$, defined as the mean of the linspace-normalised gamma quantiles:

```math
\bar{r} = \frac{F^{-1}_{\Gamma(\alpha,\,1/\alpha)}(0.01) + F^{-1}_{\Gamma(\alpha,\,1/\alpha)}(0.99)}
               {2\,F^{-1}_{\Gamma(\alpha,\,1/\alpha)}(0.99)}.
```

The defaults are design choices tuned to produce realistic scaling behaviour
rather than values read directly from any single empirical study:

```math
p_d = 0.05\,\bar{r}, \qquad p_i = 0.001\,\bar{r}.
```

The deletion coefficient (5% of the mean substitution rate) and insertion
coefficient (0.1%) are not equal because each insertion in a single lineage
creates a gap in every other sequence, amplifying its effect on total gap
content by a factor of approximately $n$. As a result, gap content grows
naturally with the number of sequences: deletion gaps accumulate as
$O(\log n)$ (from tree depth) and insertion gaps as $O(n)$ (from the
alignment-wide amplification). This matches the empirical observation, well known to practitioners working
with large protein-family databases such as Pfam (Finn et al., 2006;
Mistry et al., 2021), that larger and more diverged families are
substantially gappier.

After an undeleted non-gap site, FakeProt may introduce an insertion run using a
two-stage model that decouples start probability from run length. This is a
discrete-time analogue of the geometric-length indel model first formalised
by Thorne, Kishino & Felsenstein (1991, the "TKF91" model). An insertion
begins with probability

```math
P(\text{start insertion after site } i) = r_i\,p_i,
```

and, once started, each additional residue is appended with a fixed extension
probability $p_{\text{ext}} = 0.5$, giving a geometrically distributed run
length with mean $1/(1-p_{\text{ext}}) = 2$:

```math
P(\text{extend by one more} \mid \text{already inserting}) = p_{\text{ext}}.
```

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
in a gap run is filled with probability $p_i$; each subsequent position is filled
with probability $p_{\text{ext}}$, independently. Filling stops after the first
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
changed. If a parent site has class $c$, the duplicate first retains any
constraint with probability $1 - r_i$. Conditional on retaining a constraint,
the class is preserved with probability $1 - r_i$ or changed according to a
class-level transition matrix $M$ derived by averaging WAG probabilities between
the residues in each pair of physicochemical classes:

```math
P(c_i' = c \mid c_i' \neq \varnothing, c_i=c) = 1 - r_i,
```

```math
P(c_i' = d \mid c_i' \neq \varnothing, c_i=c) = r_i M_{cd},
\qquad d \neq c.
```

If the retained or newly chosen class does not contain the current residue,
FakeProt resamples the residue using the WAG probabilities restricted to the new
class.

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

Gene-tree branch lengths are computed after simulation as normalized Hamming
distances between parent and child alignment rows:

```math
\ell(u,v) =
\frac{1}{L^{*}}
\sum_{i=1}^{L^{*}} 1[X_{u,i} \neq X_{v,i}],
```

where $L^{*}$ is the final alignment length, including inserted columns and gaps.
The species cladogram is emitted without branch lengths.

## Outputs

For output prefix `PREFIX`, FakeProt writes:

| File | Contents |
| --- | --- |
| `PREFIX_all_sequences.<msa-format>` | All generated sequences, including ancestral and internal sequence nodes. |
| `PREFIX_current_sequences.<msa-format>` | Extant leaf sequences only. |
| `PREFIX_gene_tree.<tree-format>` | Gene tree with normalized Hamming branch lengths. |
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

1. Site-rate values are used directly as event probabilities after evaluating a
   gamma density; they are not continuous-time branch-scaled rates in the sense
   of a Yang (1994) discrete-gamma likelihood model.
2. Branch lengths are post hoc normalized mismatch fractions, not parameters
   used to generate substitutions.
3. Physicochemical classes overlap (Taylor, 1986), so class priors are
   normalized over class masses rather than over a partition of amino acids.
4. The requested number of ortholog groups is a target, not a hard guarantee; if
   too few duplication opportunities occur before the leaf target is reached, the
   simulator emits a warning.
5. The requested size is a lower bound because one speciation event can add more
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

If you use FakeProt, please cite FakeProt itself and, where appropriate, the
original sources of the models and concepts it implements.

### Substitution model

> Whelan, S. and Goldman, N. (2001). A general empirical model of protein
> evolution derived from multiple protein families using a maximum-likelihood
> approach. *Molecular Biology and Evolution*, 18(5), 691–699.

### Site-rate heterogeneity and rate autocorrelation

> Yang, Z. (1994). Maximum likelihood phylogenetic estimation from DNA
> sequences with variable rates over sites: approximate methods.
> *Journal of Molecular Evolution*, 39(3), 306–314.

> Felsenstein, J. and Churchill, G. A. (1996). A Hidden Markov Model approach
> to variation among sites in rate of evolution. *Molecular Biology and
> Evolution*, 13(1), 93–104.

> Yang, Z. (1995). A space-time process model for the evolution of DNA
> sequences. *Genetics*, 139(2), 993–1005.

### Physicochemical amino-acid classes

> Taylor, W. R. (1986). The classification of amino acid conservation.
> *Journal of Theoretical Biology*, 119(2), 205–218.

### Indel processes and gap length distributions

> Thorne, J. L., Kishino, H. and Felsenstein, J. (1991). An evolutionary
> model for maximum likelihood alignment of DNA sequences.
> *Journal of Molecular Evolution*, 33(2), 114–124.

> Benner, S. A., Cohen, M. A. and Gonnet, G. H. (1993). Empirical and
> structural models for insertions and deletions in the divergent evolution
> of proteins. *Journal of Molecular Biology*, 229(4), 1065–1082.

> Chang, M. S. S. and Benner, S. A. (2004). Empirical analysis of protein
> insertions and deletions determining parameters for the correct placement
> of gaps in protein sequence alignments. *Journal of Molecular Biology*,
> 341(2), 617–631.

### Yule (pure-birth) trees and their statistics

> Yule, G. U. (1925). A mathematical theory of evolution, based on the
> conclusions of Dr. J. C. Willis, F.R.S. *Philosophical Transactions of
> the Royal Society of London, Series B*, 213, 21–87.

> Aldous, D. J. (2001). Stochastic models and descriptive statistics for
> phylogenetic trees, from Yule to today. *Statistical Science*, 16(1), 23–34.

### Orthology and paralogy

> Fitch, W. M. (1970). Distinguishing homologous from analogous proteins.
> *Systematic Zoology*, 19(2), 99–113.

> Koonin, E. V. (2005). Orthologs, paralogs, and evolutionary genomics.
> *Annual Review of Genetics*, 39, 309–338.

### Protein-family database

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