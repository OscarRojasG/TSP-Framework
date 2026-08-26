from abc import ABC, abstractmethod
from settings import INSTANCE_FOLDER
from instances.instances import read_instances

class Solver(ABC):
    def __init__(self, name):
        self.name = name
        
    def solve(self, instance_file):
        instance_path = INSTANCE_FOLDER / instance_file
        instances = read_instances(instance_path)
        solutions = []
        for instance in instances:
            sol = self.solve_instance(instance)
            solutions.append(sol)

        return solutions

    @abstractmethod
    def solve_instance(self, instance):
        pass