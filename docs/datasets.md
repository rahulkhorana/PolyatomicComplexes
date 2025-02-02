# Datasets

## Overview

The following functions can be used to load in the `ESOL`, `photoswitches`, `FreeSolv` and `Lipophilicity` datasets. 


```py title="Datasets" linenums="1"
import pandas as pd
import importlib.resources as pkg_resources
from polyatomic_complexes.src.complexes import *

def load_esol_data():
    data_path = pkg_resources.files('polyatomic_complexes.dataset.esol') / 'ESOL.csv'
    df = pd.read_csv(data_path)
    return df

def load_photoswitches_data():
    data_path = pkg_resources.files('polyatomic_complexes.dataset.photoswitches') / 'photoswitches.csv'
    df = pd.read_csv(data_path)
    return df

def load_freesolv_data():
  data_path = pkg_resources.files('polyatomic_complexes.dataset.free_solv') / 'FreeSolv.csv'
  df = pd.read_csv(data_path)
  return df

def load_lipophil_data():
  data_path = pkg_resources.files('polyatomic_complexes.dataset.lipophilicity') / 'Lipophilicity.csv'
  df = pd.read_csv(data_path)
  return df

```