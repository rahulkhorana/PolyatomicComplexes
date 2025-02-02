# ForceComplex

## Overview
`ForceComplex` inherits from `AbstractComplex` and handles **force-field-based molecular simulations**.

=== "**Methods**"
- **`get_forces()`** → Returns force matrix.
- **`get_positions()`** → Returns atomic positions.
- **`get_dist_matrix()`** → Returns a distance matrix.
- **`get_electrostatics()`** → Returns electrostatic potential.

---

## Usage Example

```python title="Force Complex" linenums="1"
from polyatomic_complexes.src.complexes import PolyatomicGeometrySMILE
from polyatomic_complexes.src.complexes.force_complex import ForceComplex

pg = PolyatomicGeometrySMILE(smile="CC(=O)OC", mode="force-field")
force_mol = pg.smiles_to_geom_complex()
forces = force_mol.get_forces()
positions = force_mol.get_positions()
distance_matrix = force_mol.get_dist_matrix()
electrostatics = force_mol.get_electrostatics()
```

## Methods Explained

!!! note "get_forces()"
    This method returns the intermolecular forces as a `numpy.array`. Note that by default we add `Hs`. This default behavior can be overriden however. Example output:
    ```
        converged SCF energy = -264.852220234109
    --------------- DFRKS gradients ---------------
            x                y                z
    0 C     0.0411629896    -0.0347283423     0.0128927555
    1 C    -0.0358298109     0.0795204431     0.0140715557
    2 O    -0.0092124180    -0.0058840035    -0.0018293663
    3 O     0.0448447455    -0.0536829330    -0.0159012623
    4 C    -0.0339840735    -0.0098635221     0.0047397579
    5 H    -0.0051317586     0.0032685701    -0.0107188785
    6 H     0.0062313470    -0.0031662535    -0.0077637258
    7 H    -0.0070985148    -0.0048328582     0.0012088583
    8 H     0.0010782595     0.0125247699     0.0025383527
    9 H     0.0040834903     0.0116269609     0.0082846011
    10 H    -0.0061499693     0.0051703469    -0.0075109818
    ----------------------------------------------
    array([[ 0.04116299, -0.03472834,  0.01289276],
        [-0.03582981,  0.07952044,  0.01407156],
        [-0.00921242, -0.005884  , -0.00182937],
        [ 0.04484475, -0.05368293, -0.01590126],
        [-0.03398407, -0.00986352,  0.00473976],
        [-0.00513176,  0.00326857, -0.01071888],
        [ 0.00623135, -0.00316625, -0.00776373],
        [-0.00709851, -0.00483286,  0.00120886],
        [ 0.00107826,  0.01252477,  0.00253835],
        [ 0.00408349,  0.01162696,  0.0082846 ],
        [-0.00614997,  0.00517035, -0.00751098]])
    ```

!!! note "get_positions()"
    This returns the refined intermolecular positions for the molecule as a `numpy.array`. For example:
    ```
    array([[ 0.00809919, -0.18503061],
       [-0.66857748, -0.1984552 ],
       [-2.9123631 , -0.74324273],
       [ 1.16703482,  0.38378306],
       [ 0.48831073,  0.36161884],
       [-0.14786039, -2.14779956],
       [ 1.99960679,  0.33425885],
       [-1.2441163 ,  1.05021712],
       [-0.8191337 ,  1.89353409],
       [ 2.31292821,  0.80769894],
       [-0.18392878, -1.55658279]])
    ```

!!! note "get_dist_matrix()"
    The distance matrix contains interatomic distances. The return type is `numpy.array`. For example:
    ```
    array([[0.        , 0.67680983, 2.9733316 , 1.29099992, 0.72761854,
        1.96895539, 2.05809719, 1.75894303, 2.23712885, 2.50953158,
        1.38492964],
       [0.67680983, 0.        , 2.30897539, 1.92573984, 1.28533003,
        2.01769416, 2.72084391, 1.37492817, 2.09739991, 3.14670023,
        1.44201071],
       [2.9733316 , 2.30897539, 0.        , 4.23221863, 3.57565401,
        3.10084749, 5.02876305, 2.44939699, 3.36663056, 5.45060449,
        2.847082  ],
       [1.29099992, 1.92573984, 4.23221863, 0.        , 0.67908589,
        2.85269346, 0.8340436 , 2.50155633, 2.49483738, 1.22179226,
        2.36434394],
       [0.72761854, 1.28533003, 3.57565401, 0.67908589, 0.        ,
        2.58880173, 1.51154369, 1.86426152, 2.01399486, 1.8783547 ,
        2.03258541],
       [1.96895539, 2.01769416, 3.10084749, 2.85269346, 2.58880173,
        0.        , 3.28210744, 3.38069337, 4.09670423, 3.84583566,
        0.59231597],
       [2.05809719, 2.72084391, 5.02876305, 0.8340436 , 1.51154369,
        3.28210744, 0.        , 3.32179706, 3.22127881, 0.56772866,
        2.88844416],
       [1.75894303, 1.37492817, 2.44939699, 2.50155633, 1.86426152,
        3.38069337, 3.32179706, 0.        , 0.94434831, 3.56530233,
        2.81414344],
       [2.23712885, 2.09739991, 3.36663056, 2.49483738, 2.01399486,
        4.09670423, 3.22127881, 0.94434831, 0.        , 3.3149434 ,
        3.50810372],
       [2.50953158, 3.14670023, 5.45060449, 1.22179226, 1.8783547 ,
        3.84583566, 0.56772866, 3.56530233, 3.3149434 , 0.        ,
        3.43862224],
       [1.38492964, 1.44201071, 2.847082  , 2.36434394, 2.03258541,
        0.59231597, 2.88844416, 2.81414344, 3.50810372, 3.43862224,
        0.        ]])
    ```

!!! note "get_electrostatics()"
    This method returns the electrostatic potential as a `numpy.array`. Note that this is the usual molecular electrostatic potential (MEP).
