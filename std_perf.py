import time
from datetime import datetime
import tracemalloc

def time_log(time_start):
    # Run time
    log_messages = ['', '=', "  Programm successfully executed.  "]
    time_delta = round(time.time() - time_start, 1)
    log_messages.append("  Run time ....... " + str(time_delta) + " s")
    # End run time
    now = datetime.now()
    dt_string = now.strftime("%H:%M:%S")
    log_messages.append("  Run ended at ... " + str(dt_string))
    # Memory
    memory_peak = round(float(tracemalloc.get_traced_memory()[1]) / 1024**2, 3)
    log_messages.append("  Memory peak .... " + "%d MB" % memory_peak)
    # Decoration
    decoration = ''.join(['='] * len(log_messages[2]))
    log_messages[1] = decoration
    log_messages.append(decoration)
    log_messages.append('')
    return "\n".join(log_messages)