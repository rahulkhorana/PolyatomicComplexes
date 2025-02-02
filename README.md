# PolyatomicComplexes

<div align="center">

### Build & CI Status  
![workflow](https://github.com/rahulkhorana/PolyatomicComplexes/actions/workflows/build.yml/badge.svg)
![workflow](https://github.com/rahulkhorana/PolyatomicComplexes/actions/workflows/ci.yml/badge.svg)  
[![CodeFactor](https://www.codefactor.io/repository/github/rahulkhorana/polyatomiccomplexes/badge/master)](https://www.codefactor.io/repository/github/rahulkhorana/polyatomiccomplexes/overview/master)

### Licensing & Code Style  
[![Github License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### PyPI Information  
![PyPI - Version](https://img.shields.io/pypi/v/polyatomic-complexes?style=plastic&logo=%233775A9&logoSize=auto&labelColor=%233775A9&color=%23e1ad01&link=https%3A%2F%2Fpypi.org%2Fproject%2Fpolyatomic-complexes%2F0.0.8%2F)  
![PyPI - Format](https://img.shields.io/pypi/format/polyatomic-complexes)
![PyPI - Downloads](https://img.shields.io/pypi/dm/polyatomic-complexes)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/polyatomic-complexes)  
[![Socket Badge](https://socket.dev/api/badge/pypi/package/polyatomic-complexes/0.0.8?artifact_id=tar-gz)](https://socket.dev/pypi/package/polyatomic-complexes/overview/0.0.8/tar-gz)

</div>


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
