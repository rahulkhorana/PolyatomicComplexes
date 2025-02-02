# Polyatomic Complexes


!!! abstract
    Developing robust representations of chemical structures that enable models to learn topological inductive biases is challenging. In order to address this challenge we developed Polyatomic Complexes. 

    The name Polyatomic Complex is derived from ancient Greek. "Poly-" (πολύς, polýs) meaning many, and "Atomic" (ἄτομος, átomos) is derived from "atomos", meaning "indivisible" or "uncuttable". Therefore, a Polyatomic Complex is a "many atom complex" or a "complex consisting of many atoms". The complex being referred to is a CW Complex. A CW Complex is a topological space constructed in an inductive manner by gluing cells of varying dimensions in particular ways.

    More formally, a Polyatomic Complex is a representation of atomistic systems. This particular learning representation is notable as it satisifes geometric, generality, structural, efficiency and chemical informedness constraints. In essence, we provide a general algorithm to correctly encode any atomistic system.

    For more details regarding this representation please see the following research paper, [Polyatomic Complexes: A topologically-informed
    learning representation for atomistic systems](https://arxiv.org/pdf/2409.15600).


## Our API is structured as follows:

###### - `PolyatomicGeometrySMILE` : an interface for converting SMILES to polytomic complexes. More details can be found on its associated [page](polyatomic_geom_smile.md).
###### - `AbstractComplex` : the base class and general purpose option. More details can be found on its associated [page](abstract_complex.md).
###### - `ForceComplex` : inherits from AbstractComplex and leverages methods from chemistry to provide detailed intermolecular force information. More details can be found on its associated [page](force_complex.md).
###### - `QuantumComplex`: inherits from AbstractComplex and leverages the B3LYP functional and DFT to provide highly accurate chemical information. More details can be found on its associated [page](quantum_complex.md).
###### - `QuantumWavesComplex`: inherits from QuantumComplex and provides long-range interactions, information about quantum wavefunctions. More details can be found on its associated [page](quantum_waves.md).
###### - `Datasets`: the general datasets api. Currently supports the `ESOL`, `photoswitches`, `FreeSolv` and `Lipophilicity` datasets. An API for `Materials Project` and `Matbench` will be released in version 2.0. More details can be found on its associated [page](datasets.md).


??? Example Notebook
    Some example code can be found at the following Google Colab [notebook](https://colab.research.google.com/drive/19uyB67lXdk937AI5y48PYzIjXypR86Sr?usp=sharing).
