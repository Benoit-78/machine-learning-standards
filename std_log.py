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


def print_debug_message(func):
    """
    Decorator to help debugging. Displays function name & execution time.
    """
    def wrapper(*func_args, **func_kwargs):
        time_start = time.time()
        return_value = func(*func_args, **func_kwargs)
        execution_time = round(time.time() - time_start, 2)
        blanks = max(0, 30 - len(func.__name__))
        print("# DEBUG | {} | {} s".format(func.__name__ + ' ' * blanks,
                                           execution_time))
        return return_value
    return wrapper