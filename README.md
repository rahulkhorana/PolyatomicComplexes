# PolyatomicComplexes

<h4 align="center">
  
![workflow](https://github.com/rahulkhorana/PolyatomicComplexes/actions/workflows/build.yml/badge.svg)
![workflow](https://github.com/rahulkhorana/PolyatomicComplexes/actions/workflows/ci.yml/badge.svg)
[![Github License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Version](https://img.shields.io/pypi/v/polyatomic-complexes?style=plastic&logo=%233775A9&logoSize=auto&labelColor=%233775A9&color=%23e1ad01&link=https%3A%2F%2Fpypi.org%2Fproject%2Fpolyatomic-complexes%2F0.0.8%2F)
![PyPI - Format](https://img.shields.io/pypi/format/polyatomic-complexes)
![PyPI - Downloads](https://img.shields.io/pypi/dm/polyatomic-complexes)

</h4>

## Installation (pip)

1. Ensure you have python >= 3.11.6 and set up a virtual environment.
```sh
pip install virtualenv
virtualenv .env --python=python3.11.6
source .env/bin/activate
```
2. Run the following
```sh
pip install -U polyatomic-complexes==0.0.8
```
Note: If you are having trouble with the environment setup please see the following demo in colab:
[![Environment Setup](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m9AXhypRekLd0Lbomaff4UN8xkX28Cao#scrollTo=uvnToYOTGD3K)




## Installation (Repo)

1. Clone the repo.

2. Ensure you have python >= 3.11.6 and set up a virtual environment.
```sh
pip install virtualenv
virtualenv .env --python=python3.11.6
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


## License

[MIT License](https://github.com/rahulkhorana/PolyatomicComplexes/blob/master/LICENSE).

## Reference

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
