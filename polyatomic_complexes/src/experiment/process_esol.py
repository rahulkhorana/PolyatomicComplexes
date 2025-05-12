import os
import sys

sys.path.append("..")
from polyatomic_complexes.src.experiment.process import BaseProcess

class ProcessESOL(BaseProcess):
    def __init__(self):
        super().__init__(
            name_dataset_csv="ESOL.csv",
            source_path=os.path.join(os.getcwd(), "polyatomic_complexes", "dataset", "esol"),
            target_path=os.path.join(os.getcwd(), "polyatomic_complexes", "dataset", "esol"),
            smiles_column="smiles",
        )

if __name__ == "__main__":
    prc = ProcessESOL()
    # prc.process()
    # prc.process_deep_complexes()
    prc.process_stacked()