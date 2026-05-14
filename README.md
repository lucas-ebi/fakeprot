# FakeProt

Generate artificial protein families by simulating protein evolution.

FakeProt builds a species tree by repeatedly applying speciation and gene duplication events. At each node, sequences evolve under the [WAG substitution model](https://doi.org/10.1093/oxfordjournals.molbev.a003851) with gamma-distributed site rates and optional stereochemical constraints on amino acid identity. The result is a set of synthetic sequences with known phylogeny, ortholog groups, and ancestral states — useful for benchmarking aligners, tree reconstruction methods, and ortholog detection tools.

## Installation

Requires Python 3.11+.

```bash
pip install .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Usage

```text
fakeprot SIZE LENGTH [options]
```

| Argument | Description |
| --- | --- |
| `SIZE` | Minimum number of extant (leaf) sequences |
| `LENGTH` | Root sequence length (may grow due to insertions) |

| Option | Default | Description |
| --- | --- | --- |
| `-o`, `--out` | `fakeprot_out` | Output file prefix |
| `-g`, `--p-gap` | `1/SIZE` | Indel probability |
| `-n`, `--n-orthologs` | `1` | Number of ortholog groups |
| `-a`, `--shape` | `2.0` | Gamma shape for site rates |
| `-b`, `--scale` | `1.0` | Gamma scale for site rates |
| `-r`, `--seed` | `None` | Random seed |
| `-f`, `--msa-format` | `fasta` | Alignment format (`fasta`, `clustal`, `nexus`, `phylip`, `stockholm`) |
| `-t`, `--tree-format` | `newick` | Tree format (`newick`, `nexus`, `nexml`, `phyloxml`, `cdao`) |

### Example

```bash
# 50 sequences, root length 200, 3 ortholog groups, fixed seed
fakeprot 50 200 -n 3 -r 42 -o my_family
```

## Outputs

| File | Contents |
| --- | --- |
| `*_all_sequences.fasta` | Every sequence including ancestral nodes |
| `*_current_sequences.fasta` | Leaf (extant) sequences only |
| `*_gene_tree.nwk` | Gene phylogeny with branch lengths |
| `*_species_cladogram.nwk` | Species tree (no branch lengths) |
| `*_ortholog_groups.csv` | Sequence → ortholog group mapping |
| `*_OG_A.fasta`, `*_OG_B.fasta`, … | Per-ortholog-group alignments (when `-n > 1`) |
| `*_stereochemistry.csv` | Per-column amino acid frequencies and physicochemical class |
| `*_run_info.json` | Full parameter set and timestamp for reproducibility |

## Running tests

```bash
pip install pytest
pytest
```

## Citation

If you use FakeProt, please cite the WAG substitution model:

> Whelan, S. and Goldman, N. (2001). A general empirical model of protein evolution derived from multiple protein families using a maximum-likelihood approach. *Molecular Biology and Evolution*, 18(5), 691–699.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

Copyright (C) 2018 Lucas Carrijo de Oliveira
