import os
import sys

sys.path.append("..")
from polyatomic_complexes.src.experiment.process import BaseProcess


class ProcessLipophilicity(BaseProcess):
    def __init__(self):
        super().__init__(
            name_dataset_csv="Lipophilicity.csv",
            source_path=os.getcwd() + "/polyatomic_complexes/dataset/lipophilicity/",
            target_path=os.getcwd() + "/polyatomic_complexes/dataset/lipophilicity/",
            smiles_column="smiles",
        )

if __name__ == "__main__":
    prc = ProcessLipophilicity()
    # prc.process()
    # prc.process_deep_complexes()
    prc.process_stacked()
