from abc import ABC, abstractmethod


class PolyatomicComplex(ABC):
    @property
    @abstractmethod
    def polyatomcomplex(self):
        pass

    @property
    @abstractmethod
    def abstract_complex(self):
        pass

    @property
    @abstractmethod
    def atomic_structure(self):
        pass

    @property
    @abstractmethod
    def bonds(self):
        pass

    @property
    @abstractmethod
    def forces(self):
        pass

    @property
    @abstractmethod
    def electrostatics(self):
        pass

    @property
    @abstractmethod
    def wavefunctions(self):
        pass

    @abstractmethod
    def get_pc_matrix(self):
        pass

    @abstractmethod
    def get_atomic_structure(self):
        pass

    @abstractmethod
    def get_complex(self):
        pass

    @abstractmethod
    def get_bonds(self):
        pass

    @abstractmethod
    def get_forces(self):
        pass

    @abstractmethod
    def get_electrostatics(self):
        pass
