# -*- coding: utf-8 -*-
"""
    Author: B.Delorme
    Mail: delormebenoit211@gmail.com
    Creation date: 18 mai 2023
    Main purpose: divers functions.
"""

import platform
import time
from datetime import datetime
import tracemalloc
import os


def time_log(start_time):
    """Provide divers information on program execution length, and memory taken."""
    # Run time
    log_messages = ['', '=', "  Programm successfully executed.  "]
    time_delta = round(time.time() - start_time, 1)
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
    print("\n".join(log_messages))


def pretty_log(log_level, message, metrics_value, metrics_unit):
    """Print a standardized log messag, to facilitate log reading in the stdout"""
    # ==========
    part_1 = log_level + ' ' * max(0, 7 - len(log_level))
    # ==========
    if len(message) < 35:
        part_2 = message + ' ' * (35 - len(message))
    elif len(message) > 35:
        part_2 = message[:32] + '...'
    # ==========
    part_3 = str(metrics_value) + ' ' + metrics_unit + '.'
    print(f"# {part_1} | {part_2} | {part_3}")


def log(message, logfile_path):
    """Save given message in a log file.
    Format is Year-Monthname-Day-Hour:Minute:Second"""
    timestamp_format = '%Y-%h-%d-%H:%M:%S'
    now = datetime.now()
    timestamp = now.strftime(timestamp_format)
    with open(logfile_path, "a") as f:
        f.write(timestamp + ',' + message + '\n')


def print_debug_message(func):
    """Decorator to help debugging. Displays function name & execution time."""
    def wrapper(*func_args, **func_kwargs):
        time_start = time.time()
        return_value = func(*func_args, **func_kwargs)
        execution_time = round(time.time() - time_start, 2)
        blanks = max(0, 30 - len(func.__name__))
        print("# DEBUG: {} | {} s".format(
            func.__name__ + ' ' * blanks,
            execution_time
            )
        )
        return return_value
    return wrapper


def get_os_type():
    """Get operating system kind: Windows or Linux"""
    os_type = platform.platform()
    os_type = os_type.split('-')[0]
    if os_type.lower() not in ['windows', 'linux', 'mac', 'android']:
        print('# ERROR: Operating system cannot be identified')
        raise OSError
    return os_type


def set_os_separator():
    """Get separator specific to operating system: / or \\ """
    os_type = get_os_type()
    if not isinstance(os_type, str):
        raise TypeError
    if os_type == 'Windows':
        os_sep = '\\'
    elif os_type in ['Linux', 'Mac', 'Android']:
        os_sep = '/'
    else:
        print('# ERROR: Wrong input for operating system')
        raise NameError
    return os_sep


def get_today_date():
    """Get today date, to be included in directory names."""
    today_date = str(datetime.now())
    today_date = today_date.replace(' ', '_')
    today_date = today_date.replace(':', '')
    today_date = today_date.replace('-', '')
    today_date = today_date.split('.', maxsplit=1)[0]
    return today_date


def get_root_path(root_dir='local'):
    """Return current root directory"""
    relative_dir = os.getcwd().split(root_dir)[1]
    root_path = os.getcwd().split(relative_dir)[0]
    return root_path


def create_log_folder(path: str, level: str):
    """Save the given file in a time tree structure."""
    today = datetime.today().date()
    year = str(today.year)
    month = str(today.month)
    day = str(today.day)
    original_path = os.getcwd()
    os.chdir(path)
    if year not in os.listdir():
        os.mkdir(year)
    if level in ['month', 'day'] and month not in os.listdir():
        os.chdir(year)
        os.mkdir(month)
    if level == 'day' and day not in os.listdir():
        os.chdir(month)
        os.mkdir(day)
    os.chdir(original_path)
