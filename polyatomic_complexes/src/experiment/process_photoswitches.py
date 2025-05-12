import os
import sys

sys.path.append("..")
from polyatomic_complexes.src.experiment.process import BaseProcess

class ProcessPhotoswitches(BaseProcess):
    def __init__(self):
        super().__init__(
            name_dataset_csv="photoswitches.csv",
            source_path=os.path.join(os.getcwd(), "polyatomic_complexes", "dataset", "photoswitches"),
            target_path=os.path.join(os.getcwd(), "polyatomic_complexes", "dataset", "photoswitches"),
            smiles_column="SMILES",
        )

if __name__ == "__main__":
    prc = ProcessPhotoswitches()
    # prc.process()
    # prc.process_deep_complexes()
    prc.process_stacked()
