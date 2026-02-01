from abc import ABC, abstractmethod

class Storage(ABC):

    @abstractmethod
    def get_logs(self, factory_id: str):
        pass
    @abstractmethod
    def push_embedding(self, vector: list , factory_id: str , type: str , statement: str , code: str) -> bool:
        pass
    @abstractmethod
    def get_embedding(self, factory_id: str , emp_code: str , machine_code: str , date: str):
        pass
    @abstractmethod
    def get_factories(self):
        pass
    @abstractmethod
    def get_knowledge(self, factory_id: str , type: str):
        pass
    @abstractmethod
    def get_employees(self, factory_id: str):
        pass
    @abstractmethod
    def get_machines(self, factory_id: str):
        pass
    @abstractmethod
    def get_machine_performance(self, factory_id: str, machine_code: str):
        pass
    @abstractmethod
    def get_employee_performance(self, factory_id: str, emp_code: str):
        pass
    

