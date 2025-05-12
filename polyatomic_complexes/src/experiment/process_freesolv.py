import os
import sys
sys.path.append("..")
from polyatomic_complexes.src.experiment.process import BaseProcess

class ProcessFreeSolv(BaseProcess):
    def __init__(self):
        super().__init__(
            name_dataset_csv="FreeSOLV.csv",
            source_path=os.path.join(os.getcwd(), "polyatomic_complexes", "dataset", "free_solv"),
            target_path=os.path.join(os.getcwd(), "polyatomic_complexes", "dataset", "free_solv"),
            smiles_column="smiles",
        )

if __name__ == "__main__":
    prc = ProcessFreeSolv()
    # prc.process()
    # prc.process_deep_complexes()
    prc.process_stacked()