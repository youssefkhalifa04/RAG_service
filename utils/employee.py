from utils.helpers import percentage, get_comparison
def get_single_employee_deeds(employee_code: str , logs: list):
    employee_logs = [log for log in logs if log['employee']['code'] == employee_code]
    breakdowns = len([log for log in employee_logs if log['event_type'] == 'breakdown'])
    breakdowns_duration = sum(log['event_duration'] for log in employee_logs if log['event_type'] == 'breakdown')
    pauses = sum(log['event_duration'] for log in employee_logs if log['event_type'] == 'pause')
    pauses_count = len([log for log in employee_logs if log['event_type'] == 'pause'])
    good_pieces = len([log for log in employee_logs if log['event_type'] == 'good_piece'])
    bad_pieces = len([log for log in employee_logs if log['event_type'] == 'bad_piece'])
    return {
        "breakdowns": breakdowns,
        "breakdowns_duration": breakdowns_duration,
        "pauses": pauses,
        "pauses_count": pauses_count,
        "good_pieces": good_pieces,
        "bad_pieces": bad_pieces
    }
def get_single_employee_deeds_compare_avg(employee_code: str , logs: list, overall: dict):
    deeds = get_single_employee_deeds(employee_code, logs)
    avg_breakdowns = overall['total_breakdowns'] / len(set(log['employee']['code'] for log in logs)) if overall['total_breakdowns'] else 0
    avg_breakdowns_duration = overall['total_breakdowns_duration'] / len(set(log['employee']['code'] for log in logs)) if overall['total_breakdowns_duration'] else 0
    avg_good_pieces = overall['total_good_pieces'] / len(set(log['employee']['code'] for log in logs)) if overall['total_good_pieces'] else 0
    avg_bad_pieces = overall['total_bad_pieces'] / len(set(log['employee']['code'] for log in logs)) if overall['total_bad_pieces'] else 0
    deeds['breakdowns_vs_avg'] = deeds['breakdowns'] - avg_breakdowns
    deeds['breakdowns_duration_vs_avg'] = deeds['breakdowns_duration'] - avg_breakdowns_duration
    deeds['good_pieces_vs_avg'] = deeds['good_pieces'] - avg_good_pieces
    deeds['bad_pieces_vs_avg'] = deeds['bad_pieces'] - avg_bad_pieces
    return deeds
def get_average_comparison(overall: dict, logs: list):
    num_employees = len(set(log['employee']['code'] for log in logs))
    if num_employees == 0:
        return {
            "avg_breakdowns": 0,
            "avg_breakdowns_duration": 0,
            "avg_good_pieces": 0,
            "avg_bad_pieces": 0,
            "avg_pauses": 0,
            "avg_pauses_count": 0
        }
    return {
        "avg_breakdowns": overall['total_breakdowns'] / num_employees,
        "avg_breakdowns_duration": overall['total_breakdowns_duration'] / num_employees,
        "avg_good_pieces": overall['total_good_pieces'] / num_employees,
        "avg_bad_pieces": overall['total_bad_pieces'] / num_employees,
        "avg_pauses": overall['total_pauses'] / num_employees,
        "avg_pauses_count": overall['total_pauses_count'] / num_employees
    }



def generate_employee_statement(employee_code: str, machine_code: str, deeds: dict, overall: dict, avg_comp: dict):
    # 1. Calculate percentages
    breakdown_p = percentage(deeds['breakdowns'], overall['total_breakdowns'])
    duration_p  = percentage(deeds['breakdowns_duration'], overall['total_breakdowns_duration'])
    good_p      = percentage(deeds['good_pieces'], overall['total_good_pieces'])
    bad_p       = percentage(deeds['bad_pieces'], overall['total_bad_pieces'])
    
    # 2. Get relative descriptors using our helper
    comp = {
        'br': get_comparison(deeds['breakdowns'], avg_comp['avg_breakdowns']),
        'dr': get_comparison(deeds['breakdowns_duration'], avg_comp['avg_breakdowns_duration']),
        'gp': get_comparison(deeds['good_pieces'], avg_comp['avg_good_pieces']),
        'bp': get_comparison(deeds['bad_pieces'], avg_comp['avg_bad_pieces']),
        'pc': get_comparison(deeds['pauses_count'], avg_comp['avg_pauses_count']),
        'pp': get_comparison(deeds['pauses'], avg_comp['avg_pauses'])
    }

    # 3. Formulate the narrative
    stmt = (
        f"Employee {employee_code} (Machine {machine_code}) Performance Report: "
        f"Responsible for {breakdown_p:.2f}% of factory breakdowns ({deeds['breakdowns']} incidents), "
        f"which is {comp['br']} the average of {avg_comp['avg_breakdowns']}. "
        f"This caused {deeds['breakdowns_duration']} minutes of downtime ({duration_p:.2f}% of total loss), "
        f"placing the employee {comp['dr']} the factory average duration. "
        f"Output included {deeds['good_pieces']} good units ({good_p:.2f}% of total, {comp['gp']} average) "
        f"and {deeds['bad_pieces']} defective units ({bad_p:.2f}% of waste, {comp['bp']} average). "
        f"Operational pauses: {deeds['pauses_count']} incidents ({comp['pc']} average) "
        f"totaling {deeds['pauses']} minutes ({comp['pp']} average)."
    )
    print("Employee statement generated:", stmt)
    return stmt