from interfaces.Storage import Storage
from integration.supabase_client import sp
import numpy as np
class SupabaseStorage(Storage):

    def __init__(self):
        super().__init__()

    def get_logs(self, factory_id: str):
        try : 
            data = sp.table("production_logs").select("event_type, event_duration, employee(full_name,code), machine(code)").eq("factory_id", factory_id).execute()
            return data.data
        except Exception as e:
            return f"DATABASE ERROR: {e}"
    @staticmethod
    def _format_vector(vector) -> str:
        """
        Converts a numpy array or list to pgvector format (comma-separated string).
        Example: [0.1, 0.2, 0.3] -> '[0.1,0.2,0.3]'
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        # Handle nested lists - if vector is a list of arrays, take the first one
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], (list, np.ndarray)):
            vector = vector[0]
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
        # Format as comma-separated values
        formatted = ','.join(str(float(v)) for v in vector)
        return f'[{formatted}]'
    
    def push_embedding(self, vector: list , factory_id: str , type: str , statement: str , code: str):
        
        try:
            data = {}
            if type == "employee":
                data = {
                "factory_id": factory_id,
                "statement": statement,
                "type": type,
                "emp_code": code,
                "embedding": self._format_vector(vector[0])
                }
            else:
                data = {
                "factory_id": factory_id,
                "statement": statement,
                "type": type,
                "machine_code": code,
                "embedding": self._format_vector(vector[0])
                }
            
            res = sp.table("factory_knowledge_base").insert(data).execute()
            return res
        except Exception as e:
            print(f"Error pushing vector to DB: {e}")
            raise ValueError("Error pushing vector to DB")


    def get_embedding(self, factory_id: str , emp_code: str = None , machine_code: str = None , date: str = None):
        try:
            query = sp.table("factory_knowledge_base").select("embedding, statement").eq("factory_id", factory_id)
            if emp_code:
                query = query.eq("emp_code", emp_code)
            if machine_code:
                query = query.eq("machine_code", machine_code)
            response = query.execute()
            records = response.data
            if not records:
                return "ERROR: No data found for this factory."
            return records
        except Exception as e:
            return f"DATABASE ERROR: {e}"
        

    def get_factories(self):
        data = sp.table("factory").select("factory_id").execute()
        stack = [item['factory_id'] for item in data.data]
        return stack

    def get_knowledge(self, factory_id: str, type: str):
        try:
            response = (
                sp.table("factory_knowledge_base")
                .select("embedding, statement")
                .eq("factory_id", factory_id)
                .eq("type", type)
                .execute()
            )
            records = response.data
            if not records:
                return "ERROR: No data found for this factory."
            return records
        except Exception as e:
            return f"DATABASE ERROR: {e}"

    def get_employees(self, factory_id: str):
        try : 
            data = sp.table("employee").select("full_name, code").eq("factory_id", factory_id).execute()
            return data.data
        except Exception as e:
            return f"DATABASE ERROR: {e}"

    def get_machines(self, factory_id: str):
        try : 
            data = sp.table("machine").select("code").eq("factory_id", factory_id).execute()
            return data.data
        except Exception as e:
            return f"DATABASE ERROR: {e}"
    def get_machine_performance(self, factory_id: str, machine_code: str):
        try : 
            data = sp.table("machine_performance").select("*").eq("factory_id", factory_id).eq("machine_code", machine_code).execute()
            return data.data
        except Exception as e:
            return f"DATABASE ERROR: {e}"

    def get_employee_performance(self, factory_id: str, emp_code: str):
        try : 
            data = sp.table("employee_performance").select("*").eq("factory_id", factory_id).eq("emp_code", emp_code).execute()
            return data.data
        except Exception as e:
            return f"DATABASE ERROR: {e}"
    