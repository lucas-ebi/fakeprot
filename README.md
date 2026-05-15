# FakeProt

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
probabilities, gamma-distributed site-rate heterogeneity, indel events, and
probabilistic physicochemical constraints. Insertions are represented as new
alignment columns, while deletions are represented as gaps, so every emitted
sequence remains in a common alignment.

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
| `-g`, `--p-gap` | `1/SIZE` | Baseline gap/indel probability, denoted below by $p_g$. |
| `-n`, `--n-orthologs` | `1` | Target number of ortholog-group anchors. |
| `-a`, `--shape` | `0.75` | Shape parameter $\alpha$ for the gamma site-rate model. Scale is fixed to $1/\alpha$ so that the mean rate equals 1. |
| `-r`, `--seed` | `None` | Random seed for reproducible simulations. |
| `-f`, `--msa-format` | `fasta` | Alignment format: `fasta`, `clustal`, `nexus`, `phylip`, or `stockholm`. |
| `-t`, `--tree-format` | `newick` | Tree format: `newick`, `nexus`, `nexml`, `phyloxml`, or `cdao`. |

## Generative Model

### Notation

Let $A$ be the set of 20 canonical amino acids and let $\pi_a$ be the WAG
background frequency of amino acid $a \in A$. FakeProt stores the WAG empirical
off-diagonal substitution probabilities in a row-stochastic matrix $W$, where
$W_{ab}$ is the conditional probability of proposing residue $b$ from residue
$a$, with $a \neq b$.

Let $C$ be the set of physicochemical classes used by the simulator. Each class
$c \in C$ corresponds to a subset $S_c \subset A$; classes are allowed to overlap.
The class prior is proportional to the WAG background mass of its member
residues:

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
$r_i$ to every alignment site. The first site is fixed at $r_1 = 0$
to preserve the initial methionine. For the remaining $L - 1$ sites, rates
are drawn from a $\Gamma(\alpha,\,\beta)$ distribution with
$\beta = 1/\alpha$ (so the mean rate equals 1, following the standard
phylogenetic convention). Specifically, $L-1$ evenly spaced quantile values
are taken between the 1st and 99th percentiles:

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
neighboring sites to have similar mutability.

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
class-conditional WAG background distribution. If unconstrained, the amino acid
is drawn directly from the WAG background frequencies $\pi$.

### Substitution Model Along an Edge

For each child sequence, FakeProt iterates through the parent sequence from left
to right. At a non-gap site with parent residue $x_i$, site rate $r_i$, and
constraint $c_i$, deletion is considered first:

```math
P(\mathrm{delete}\ i) = p_g r_i.
```

If the site is not deleted and has no physicochemical constraint, it is retained
with probability $1 - r_i$; otherwise a new residue is sampled using the WAG row
for the current residue:

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

After an undeleted non-gap site, FakeProt may introduce an insertion run. The
probability of the $j$-th inserted residue after site $i$ decays geometrically:

```math
P(\mathrm{insert}_{i,j}) = r_i p_g 2^{-j},
\qquad j = 0,1,2,\ldots
```

Inserted residues are sampled from the WAG background distribution, assigned rate
$1.0$, and assigned no physicochemical class. When an insertion creates a new
alignment site, FakeProt inserts a gap in every previously generated row so
that all sequences remain aligned.

Existing gap runs can also be filled in descendant lineages. For the $j$-th
position in a consecutive gap run, the fill probability is

```math
P(\mathrm{fill}_{j}) = p_g 2^{-j}.
```

Filling stops after the first failed attempt in that run. Filled residues are
sampled either from the WAG background distribution or, when the site has a
physicochemical class, from the corresponding class-conditional distribution.

## Species, Gene Trees, and Orthology

FakeProt maintains two directed graphs during simulation: a species tree and a
sequence/gene tree. The species tree contains extant and ancestral species nodes.
The gene tree contains ancestral, duplicated, and extant sequence nodes.

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
redrawn using perturbed gamma parameters

```math
\alpha' \sim N(\alpha, 1) \mid \alpha' \geq 1,
\qquad
\beta' \sim N(\beta, 1) \mid \beta' \geq 1,
```

while preserving the rank order of the parent rates. That is, sites that were
slow or fast in the parent remain relatively slow or fast in the duplicate, but
the numerical rate scale may shift.

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

Ortholog groups are defined by anchor nodes: the root sequence and every
successful duplication copy. A leaf belongs to anchor $a_j$ if it descends from
$a_j$ without crossing another ortholog anchor:

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
| `PREFIX_ortholog_groups.csv` | Mapping from extant sequence identifiers to ortholog-group labels. |
| `PREFIX_OG_A.<msa-format>`, `PREFIX_OG_B.<msa-format>`, ... | Per-ortholog-group alignments, emitted when `--n-orthologs > 1`. |
| `PREFIX_physicochemical_groups.csv` | Per-column amino-acid frequencies and physicochemical class annotations for each ortholog group. |
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
   gamma density; they are not continuous-time branch-scaled rates.
2. Branch lengths are post hoc normalized mismatch fractions, not parameters
   used to generate substitutions.
3. Physicochemical classes overlap, so class priors are normalized over class
   masses rather than over a partition of amino acids.
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

If you use FakeProt, please cite FakeProt itself where appropriate and cite the
WAG substitution model used by the simulator:

> Whelan, S. and Goldman, N. (2001). A general empirical model of protein
> evolution derived from multiple protein families using a maximum-likelihood
> approach. Molecular Biology and Evolution, 18(5), 691-699.

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

Copyright (C) 2018 Lucas Carrijo de Oliveira.
