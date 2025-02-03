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
  <img src="https://img.shields.io/pypi/dm/polyatomic-complexes" alt="PyPI Downloads">
  <img src="https://img.shields.io/pepy/dt/polyatomic-complexes" alt="Total Downloads">
  <a href="https://socket.dev/pypi/package/polyatomic-complexes/overview/1.0.7/tar-gz">
    <img src="https://socket.dev/api/badge/pypi/package/polyatomic-complexes/1.0.7?artifact_id=tar-gz" alt="Socket Badge">
  </a>
  <a href="https://www.codefactor.io/repository/github/rahulkhorana/polyatomiccomplexes/overview/master">
    <img src="https://www.codefactor.io/repository/github/rahulkhorana/polyatomiccomplexes/badge/master" alt="CodeFactor">
  </a>
</p>



## 📖 **[Documentation](https://rahulkhorana.github.io/PolyatomicComplexes/)**



## 🚀 Installation  

### **Using `pip`**
1. Ensure you have python == 3.11.11 and set up a virtual environment.
```sh
pip install virtualenv
virtualenv .env --python=python3.11.11
source .env/bin/activate
```
2. Run the following
```sh
pip install -U polyatomic-complexes==1.0.7
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


## 📜 License
**This project is licensed under the [MIT License](https://github.com/rahulkhorana/PolyatomicComplexes/blob/master/LICENSE).**


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
