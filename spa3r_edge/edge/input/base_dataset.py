from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def next(self):
        pass
