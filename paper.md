---
title: 'Polyatomic Complexes: A Software Framework for Topologically Accurate Representations of Molecules'
tags:
  - chemistry
  - molecular machine learning
  - topological data analysis
  - computational chemistry
  - representation learning
authors:
  - name: Rahul Khorana
    affiliation: 1
    orcid: 0000-0001-8795-1623
affiliations:
  - name: Imperial College London
    index: 1
date: 17 April 2025
bibliography: paper.bib
---

# Summary

Developing robust representations of chemical structures that enable models to learn topological inductive biases is challenging. Polyatomic complexes are a novel learning representation for atomistic systems that addresses this challenge. The representation satisfies numerous structural, geometric, efficiency, and generalizability constraints.
At a high level, the representation is formed by systematically representing electrons, protons, and neutrons as an interconnected topological structure, namely an atomic complex. The atomic complexes are then composed to form molecular-level representations termed Polyatomic Complexes. The representation is suitable for property prediction and similarity-based screening tasks.

# Statement of Need

Current molecular representations, such as SMILES, SELFIES, or graph-based fingerprints, are limited in their ability to accurately reflect electronic structure and higher-order topological features. Additionally, existing representations in computational chemistry fail to satisfy some non-empty subset of accuracy, generalizability, and efficiency constraints [@khorana2024polyatomiccomplexestopologicallyinformedlearning].
Polyatomic Complexes fill this gap by providing a physics-informed and topologically accurate representation that is general across chemical domains, modular, and compatible with standard ML workflows. The software is helpful to researchers in cheminformatics, quantum chemistry, and materials science seeking a theoretically grounded approach to feature generation.

# Software Description

Polyatomic Complexes construct atom-level representations using CW-complexes to model electrons, protons, and neutrons. 
These are then assembled into higher-dimensional molecular complexes using gluing maps. 
The result is a representation satisfying the following properties:

- Invariance under changes in atom indexing, and those fundamental to physics (rotation, reflection, translations)
- Continuity and Differentiability with respect to atomic positions
- Topological accuracy up to electronic structure
- Generalizability/Generality, in essence, the representation can encode any atomistic system
- Well informed by physics and chemistry, in essence, they describe the geometry of individual atoms, encode radial functions efficiently, and can consider long-range interactions

The software is modular, extensible, and written in Python. 
It supports input from standard molecular formats and returns representations and features suitable for use with ML libraries.

# Installation

## Using pip
Ensure you have Python `== 3.11.11` and set up a virtual environment:

```bash
pip install virtualenv
virtualenv .env --python=python3.11.11
source .env/bin/activate
pip install -U polyatomic-complexes==1.0.8
```

# Example Usage

```python
from polyatomic_complexes.src.complexes import PolyatomicGeometrySMILE
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex

pg = PolyatomicGeometrySMILE(smile="CC(=O)OC", mode="abstract")
abstract_mol = pg.smiles_to_geom_complex()

bonds = abstract_mol.get_bonds()
structure = abstract_mol.get_atomic_structure()
incidence = abstract_mol.get_incidence()
skeleta = abstract_mol.get_skeleta()
adjacencies = abstract_mol.get_adjacencies()
spec_chains = abstract_mol.get_spectral_k_chains()
```

# Acknowledgements

This work builds on foundational insights in topological data analysis, computational chemistry, and equivariant machine learning.
We would like to acknowledge Dr. Jin Qian and Dr. Marcus Noack for their support and mentorship during the genesis of this project.

# References
