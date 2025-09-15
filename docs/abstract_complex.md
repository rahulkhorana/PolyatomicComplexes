# AbstractComplex

## Overview
`AbstractComplex` is the **base class** for molecular complex representations. All other polyatomic complexes inherit from this class.

=== "**Provided Methods**"
- **`get_bonds()`** → Returns bond information within the molecule.
- **`get_laplacians()`** → Returns Laplacian matrices.
- **`get_atomic_structure()`** → Returns atomic structure details.
- **`get_all_cell_coadj()`** → Returns all cell coadjacencies.
- **`get_incidence()`** → Returns incidence matrices.
- **`get_skeleta()`** → Returns skeletal molecular structure.
- **`get_adjacencies()`** → Returns adjacency matrices for molecular structures.
- **`get_spectral_k_chains()`** → Returns k-chains of complex with additional spectral features.
- **`get_raw_k_chains()`** → Returns k-chains of complex.
- **`abstract_mol.k_chains_formal_sum()`** → Represents k-chains symbolically (as a formal sum).


## Inheritance Example
```python title="Inheritance" linenums="1"
from polyatomic_complexes.src.complexes import PolyatomicGeometrySMILE
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex

class YourComplex(AbstractComplex):
    def __init__(self, smile, target_dimension, atoms, bonds):
        super().__init__(smile, target_dimension, atoms, bonds)
        self.smile = smile
        self.dim = target_dimension
        self.atoms = atoms
        self.bnds = bonds
        self.roc = self.rank_order_complex()
        ... # add anything you like here
    
    def unpack_roc(self):
        self._molecule, self._molecule_feat = self.roc["molecule"]
        self._nucleus, self._nucleus_feat = self.roc["nuclear_structure"]
        self._electrons, self._electron_feat = self.roc["electronic_structure"]
        return

    def electrostatics(self) -> np.ndarray:
        # override/add at your will
    
    def compute_dispersion_energy(self) -> Optional[List]:
        # override/add at your will
    
    def your_function_here(self) -> Any:
        # add at your will
```

## Applied Example

```python title="Abstract Complex" linenums="1"
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

## Methods Explained

!!! note "get_bonds()"
    Returns all chemical bonds between pairs of atoms. The return type is `List[Tuple]` where each tuple is of length `3`. 
    The first two values being the atoms and the third value is a list contaning the kind of chemical bond and its numerical equivalent.
    For example: 
    ```md
    [('C', 'C', ['SINGLE', 1.0]),
    ('C', 'O', ['DOUBLE', 2.0]),
    ('C', 'O', ['SINGLE', 1.0]),
    ('O', 'C', ['SINGLE', 1.0])]
    ```

!!! note "get_laplacians()"
    In this context the method returns the Hodge Laplacian. The Hodge Laplacian is a generalization of the graph Laplacian.
    More formally for a CW complex $X$ let $C_{k}(X)$ denote the set of all $k$-chains on $X$. $C_{k}(X)$ has the algebraic structure 
    of a free Abelian group with basis $\{e_{\alpha}^{k}\}_{\alpha = 1}^{N}$. Then let $d_{k}: C^{k}(X) \to C^{k+1}(X)$ be the coboundary operator, and $d_{k}^{*}$ its adjoint. Then the Hodge Laplacian is:
    $$
        \Delta_{k} := d_{k-1} \circ d_{k-1}^{\ast} + d_{k}^{\ast} \circ d_{k}
    $$.
    This method will return the Hodge Laplacians for all $k$. 
    The return type is a `Dict` with a single key `"molecule_laplacians"` whose value is a `List[Tuple]` wherein one can find the laplacians, a `numpy.matrix`, for each $k$. Note that the dictionary returned contains all laplacians at the atomic level for every atom. One can retrieve the laplacians at the electronic structure level, however we leave such functionality for the `get_atomic_structure()` method.

!!! note "get_atomic_structure()"
    This method enables one to iterate over the entire molecular structure. Essentially one can iterate over every atom and its corresponding protons, neutrons, and electrons. This explicitly describes the entire structure along with the laplacians, incidence and even weights at every level up to electronic structure. Note that there are byte arrays corresponding to approximate sampled locations, one will have to use `np.frombuffer(value, dtype=np.uint8)` in order to convert these into matrix form. Note that `value` here is a proton, neutron, or electron. Additionally note that all dirac matricies are stored as `scipy` COOrdinate sparse matricies of dtype `int64`.

!!! note "get_all_cell_coadj()"
    This returns the standard coadjacency matrix of all cells with respect to all nodes. The return type is a `defaultdict` with key `"molecule_all_cell_coadj"`.

!!! note "get_incidence()"
    This method returns a `defaultdict` with key `molecule_incidence` that contains the incidence relations across all levels, up to electronic structure. The information returned is conceptually equivalent to that of incidence matricies in a graph, but for CW complexes.

!!! note "get_skeleta()"
    This method returns a `defaultdict` enabling one to iterate over the skeleton of the molecule. The only key is `'molecule_skeleta'`.

!!! note "get_adjacencies()"
    This method will return the adjacency matricies for the entire molecule. An adjacency matrix in this context can be thought of as equivalent to the usual definition for graphs.



!!! note "get_spectral_k_chains()"
    This method will return all $k$-chains for the complex with additional spectral features computed aka scaling, persistence. Note that we don't provide a $0$-chain due to reliance on higher $k$ to compute the additional features. To use $k=0$ rely on `get_raw_k_chains()`.


!!! note "get_raw_k_chains()"
    This method will return all $k$-chains for the complex. Note that we provide the $0$ chains explicitly.


!!! note "abstract_mol.k_chains_formal_sum()"
    This method will return the symbolic computation for `get_raw_k_chains()` aka represent the computation of all $k$ chains as a formal sum.