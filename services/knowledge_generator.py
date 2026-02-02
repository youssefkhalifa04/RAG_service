
from interfaces.Storage import Storage
from services.data_encoder import DataEncoder
from interfaces.EmbeddingModel import EmbeddingModel
from utils.employee import generate_statement_new_format, get_average_comparison, get_single_employee_deeds , generate_employee_statement
from utils.machine import  generate_machine_statement, generate_machine_statement_new_format
class KnowledgeGenerator:
    def __init__(self, storage : Storage , embed_model : EmbeddingModel , encoder: DataEncoder = None):
        self.storage = storage
        self.embed_model = embed_model
        self.encoder = encoder if encoder is not None else DataEncoder(self.storage , self.embed_model)
    
    def generate_knowledge(self):
        queue = self.storage.get_factories()
        
        for factory_id in queue:
            print("Processing a factory...")
            logs = self.storage.get_logs(factory_id)
            if len(logs) == 0:
                print(f"No logs found for factory {factory_id}. Skipping...")
                continue

            else:
                # Further processing can be done here
                emp_mach_codes = set((log['employee']['code'], log['employee']['full_name'], log['machine']['code']) for log in logs)
                '''stats = {
                "total_breakdowns" : len([log for log in logs if log['event_type'] == 'breakdown']) or 0,
                "total_breakdowns_duration" :sum(log['event_duration'] for log in logs if log['event_type'] == 'breakdown') or 0,
                "total_pauses" : sum(log['event_duration'] for log in logs if log['event_type'] == 'pause') or 0,
                "total_pauses_count" : len([log for log in logs if log['event_type'] == 'pause']) or 0,
                "total_good_pieces" : len([log for log in logs if log['event_type'] == 'good_piece']) or 0,
                "total_bad_pieces" : len([log for log in logs if log['event_type'] == 'bad_piece']) or 0,
                }'''
                #avg_comp = get_average_comparison(stats, logs)
                for emp_code, full_name, machine_code in emp_mach_codes:
                    deeds = get_single_employee_deeds(emp_code , logs)
                    mach_stmt = generate_machine_statement_new_format(machine_code, deeds)
                    statement = generate_statement_new_format(full_name=full_name, employee_code=emp_code, machine_code=machine_code, deeds=deeds)
                    try:
                        self.encoder.encode_data(statement , "employee" , factory_id , emp_code)
                    except Exception as e:
                        print(f"Error generating vector for employee {emp_code} in factory {factory_id}: {e}")
                        raise ValueError("Error generating vector for employee")
                  
                    try:
                        self.encoder.encode_data(mach_stmt , "machine" , factory_id , machine_code)
                    except Exception as e:
                        print(f"Error generating vector for machine {machine_code} in factory {factory_id}: {e}")
                        raise ValueError("Error generating vector for machine")
                    
        return True
    