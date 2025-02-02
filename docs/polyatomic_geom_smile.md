# PolyatomicGeometrySMILE

## Overview
`PolyatomicGeometrySMILE` is an interface that enables users to pass a **SMILES string** as input and select a particular kind of **polyatomic complex** they want to instantiate. 

=== "mode"
    - "abstract" → `AbstractComplex`.
    - "force-field" → `ForceComplex`.
    - "quantum" → `QuantumComplex`
    - "quantum-waves" → `QuantumWavesComplex`.

=== "smile"
    - Must be a valid/well-formed SMILE string.

Each of these polyatomic complexes has its own specialized methods.

### **Primary Method**

- **`smiles_to_geom_complex()`** → Converts SMILES representation into a polyatomic complex.

---

## Usage Example

```py title="Polyatomic Complexes Interface" linenums="1"
from polyatomic_complexes.src.complexes import PolyatomicGeometrySMILE

# Initialize with a SMILES string
gm = PolyatomicGeometrySMILE(smile="CC(=O)OC", mode="quantum")

# Convert to geometric complex
quantum_mol = gm.smiles_to_geom_complex()
```