# PolyatomicComplexes

<p align="center">
  <img src="https://github.com/rahulkhorana/PolyatomicComplexes/actions/workflows/build.yml/badge.svg" alt="Build Status">
  <img src="https://github.com/rahulkhorana/PolyatomicComplexes/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT%202.0-blue.svg" alt="MIT License">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style: Black">
  </a>
  <img src="https://img.shields.io/pypi/v/polyatomic-complexes?style=plastic&logo=%233775A9&logoSize=auto&labelColor=%233775A9&color=%23e1ad01&link=https%3A%2F%2Fpypi.org%2Fproject%2Fpolyatomic-complexes%2F0.0.8%2F" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/format/polyatomic-complexes" alt="PyPI Format">
  <a href="https://pepy.tech/project/polyatomic-complexes">
  <img src="https://static.pepy.tech/badge/polyatomic-complexes" alt="Downloads">
</a>
  <a href="https://socket.dev/pypi/package/polyatomic-complexes/overview/1.0.7/tar-gz">
    <img src="https://socket.dev/api/badge/pypi/package/polyatomic-complexes/1.0.7?artifact_id=tar-gz" alt="Socket Badge">
  </a>
  <a href="https://www.codefactor.io/repository/github/rahulkhorana/polyatomiccomplexes/overview/master">
    <img src="https://www.codefactor.io/repository/github/rahulkhorana/polyatomiccomplexes/badge/master" alt="CodeFactor">
  </a>
  <a style="border-width:0" href="https://doi.org/10.21105/joss.08828">
  <img src="https://joss.theoj.org/papers/10.21105/joss.08828/status.svg" alt="DOI badge" >
</a>
</p>



##  **[Documentation](https://rahulkhorana.github.io/PolyatomicComplexes/)**



## 🚀 Installation  

### **Using `pip`**
1. Ensure you have python >= 3.11.11 and set up a virtual environment.
```sh
pip install virtualenv
virtualenv .env --python=python3.11.11
source .env/bin/activate
```
2. Run the following
```sh
pip install -U polyatomic-complexes==1.0.8
```
Note: If you are having trouble with the environment setup please see the following demo in colab:
[![Environment Setup](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19uyB67lXdk937AI5y48PYzIjXypR86Sr?usp=sharing)


### **Using the `repo`**

1. Clone the repo.

2. Ensure you have python >= 3.11.11 and set up a virtual environment.
```sh
pip install virtualenv
virtualenv .env --python=python3.11.11
source .env/bin/activate
```

3. Install the relevant packages.

For standard/minimal usage:
```sh
pip install -Ur requirements/requirements.txt
```

For graph based experiments:
```sh
pip install -Ur requirements/requirements_graph.txt
```

For materials based experiments:
```sh
pip install -Ur requirements/requirements_mat.txt
```

4. Get all large files from git lfs

```sh
git lfs fetch --all
git lfs pull
```

## 🚀 Datasets
All dataset csv's come from the original manuscripts as cited in the paper namely:
1. ESOL:  DELANEY, J. S. Esol: Estimating aqueous solubility directly from molecular structure. Journal
of Chemical Information and Computer Sciences 44, 3 (2004), 1000–1005. PMID: 15154768.
2. FreeSolv: MOBLEY, D. L., AND GUTHRIE, J. P. Freesolv: a database of experimental and calculated
hydration free energies, with input files. Journal of computer-aided molecular design 28 (2014),
711–720.
3. Lipophilicity:  GAULTON, A., BELLIS, L. J., BENTO, A. P., CHAMBERS, J., DAVIES, M., HERSEY, A.,
LIGHT, Y., MCGLINCHEY, S., MICHALOVICH, D., AL-LAZIKANI, B., AND OVERINGTON,
J. P. ChEMBL: a large-scale bioactivity database for drug discovery. Nucleic Acids Research
40, D1 (09 2011), D1100–D1107.
4. Photoswtiches: GRIFFITHS, R.-R., GREENFIELD, J. L., THAWANI, A. R., JAMASB, A. R., MOSS, H. B.,
BOURACHED, A., JONES, P., MCCORKINDALE, W., ALDRICK, A. A., FUCHTER, M. J.,
AND LEE, A. A. Data-driven discovery of molecular photoswitches with multioutput gaussian
processes. Chem. Sci. 13 (2022), 13541–13551.


## 📜 License
**This project is licensed under the [MIT License](https://github.com/rahulkhorana/PolyatomicComplexes/blob/master/LICENSE).**

## Community

We use [GitHub Discussions](https://github.com/rahulkhorana/PolyatomicComplexes/discussions) for:

- 💬 Asking questions or requesting help
- 🐞 Reporting bugs or issues
- 💡 Suggesting new features or improvements

Please feel free to start a discussion if you're interested in contributing!


## 🔬 Reference
```
@misc{khorana2024polyatomiccomplexestopologicallyinformedlearning,
      title={Polyatomic Complexes: A topologically-informed learning representation for atomistic systems}, 
      author={Rahul Khorana and Marcus Noack and Jin Qian},
      year={2024},
      eprint={2409.15600},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.15600}, 
}

```
