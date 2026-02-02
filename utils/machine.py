from utils.helpers import percentage, get_comparison
def get_machine_details(machine_code: str, logs: list):
    machine_logs = [log for log in logs if log['machine']['code'] == machine_code]
    total_events = len(machine_logs)
    good_pieces = len([log for log in machine_logs if log['event_type'] == 'good_piece'])
    bad_pieces = len([log for log in machine_logs if log['event_type'] == 'bad_piece'])
    breakdowns = len([log for log in machine_logs if log['event_type'] == 'breakdown'])
    breakdowns_duration = sum(log['event_duration'] for log in machine_logs if log['event_type'] == 'breakdown')
    pauses = sum(log['event_duration'] for log in machine_logs if log['event_type'] == 'pause')
    pauses_count = len([log for log in machine_logs if log['event_type'] == 'pause'])
    return {
        "total_events": total_events,
        "good_pieces": good_pieces,
        "bad_pieces": bad_pieces,
        "breakdowns": breakdowns,
        "breakdowns_duration": breakdowns_duration,
        "pauses": pauses,
        "pauses_count": pauses_count
    }

def get_machine_performance_comparison(machine_code: str, logs: list, overall: dict):
    details = get_machine_details(machine_code, logs)
    num_machines = len(set(log['machine']['code'] for log in logs))
    if num_machines == 0:
        return {
            "breakdowns_vs_avg": 0,
            "breakdowns_duration_vs_avg": 0,
            "good_pieces_vs_avg": 0,
            "bad_pieces_vs_avg": 0
        }
    avg_breakdowns = overall['total_breakdowns'] / num_machines if overall['total_breakdowns'] else 0
    avg_breakdowns_duration = overall['total_breakdowns_duration'] / num_machines if overall['total_breakdowns_duration'] else 0
    avg_good_pieces = overall['total_good_pieces'] / num_machines if overall['total_good_pieces'] else 0
    avg_bad_pieces = overall['total_bad_pieces'] / num_machines if overall['total_bad_pieces'] else 0
    details['breakdowns_vs_avg'] = details['breakdowns'] - avg_breakdowns
    details['breakdowns_duration_vs_avg'] = details['breakdowns_duration'] - avg_breakdowns_duration
    details['good_pieces_vs_avg'] = details['good_pieces'] - avg_good_pieces
    details['bad_pieces_vs_avg'] = details['bad_pieces'] - avg_bad_pieces
    return details

def generate_machine_statement(machine_code: str, deeds: dict, overall: dict, avg_comp: dict):
    # 1. Calculate percentages
    breakdown_p = percentage(deeds['breakdowns'], overall['total_breakdowns']) if overall['total_breakdowns'] else 0
    duration_p = percentage(deeds['breakdowns_duration'], overall['total_breakdowns_duration']) if overall['total_breakdowns_duration'] else 0
    good_p = percentage(deeds['good_pieces'], overall['total_good_pieces']) if overall['total_good_pieces'] else 0
    bad_p = percentage(deeds['bad_pieces'], overall['total_bad_pieces']) if overall['total_bad_pieces'] else 0

    # 2. Determine comparisons
    comp = {
        "br": get_comparison(deeds['breakdowns'], avg_comp['avg_breakdowns']),
        "dr": get_comparison(deeds['breakdowns_duration'], avg_comp['avg_breakdowns_duration']),
        "gp": get_comparison(deeds['good_pieces'], avg_comp['avg_good_pieces']),
        "bp": get_comparison(deeds['bad_pieces'], avg_comp['avg_bad_pieces']),
        "pc": get_comparison(deeds['pauses_count'], avg_comp['avg_pauses_count']),
        "pp": get_comparison(deeds['pauses'], avg_comp['avg_pauses'])
    }

    # 3. Formulate the narrative
    stmt = (
        f"Machine {machine_code} Performance Report: "
        f"Responsible for {breakdown_p:.2f}% of factory breakdowns ({deeds['breakdowns']} incidents), "
        f"which is {comp['br']} the average of {avg_comp['avg_breakdowns']}. "
        f"This caused {deeds['breakdowns_duration']} minutes of downtime ({duration_p:.2f}% of total loss), "
        f"placing the machine {comp['dr']} the factory average duration. "
        f"Output included {deeds['good_pieces']} good units ({good_p:.2f}% of total, {comp['gp']} average) "
        f"and {deeds['bad_pieces']} defective units ({bad_p:.2f}% of waste, {comp['bp']} average). "
        f"Operational pauses: {deeds['pauses_count']} incidents ({comp['pc']} average) "
        f"totaling {deeds['pauses']} minutes ({comp['pp']} average)."
    )
    print("Machine statement generated:", stmt)
    return stmt


def generate_machine_statement_new_format( machine_code: str, deeds: dict):
    stmt = (
        f"Performance Report on Machine {machine_code}:\n"
        f"- Breakdowns: {deeds['breakdowns']} incidents causing {deeds['breakdowns_duration']} minutes of downtime.\n"
        f"- Production: {deeds['good_pieces']} good units and {deeds['bad_pieces']} defective units produced.\n"
        f"- Pauses: {deeds['pauses_count']} incidents totaling {deeds['pauses']} minutes.\n"
    )
    print("New format statement generated:", stmt)
    return stmt