# PolyatomicComplexes

<h4 align="center">
  
![workflow](https://github.com/rahulkhorana/PolyatomicComplexes/actions/workflows/build.yml/badge.svg)
![workflow](https://github.com/rahulkhorana/PolyatomicComplexes/actions/workflows/ci.yml/badge.svg)
[![Github License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</h4>

## Installation

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
@inproceedings{
khorana2024polyatomiccomplexes,
title={Polyatomic Complexes},
author={Rahul Khorana, Marcus M. Noack, Jin Qian},
booktitle={Submitted},
year={2024},
url={https://openreview.net/forum?id=}
}

```
