from collections import defaultdict

def log(*args, **kwargs):
    """
    Helper function for logging output.

    Inputs:
        *args, **kwargs     Arguments for the print() call
    """
    print(*args, **kwargs)

warnings_issued = defaultdict(lambda:False)

def log_singleton(warning_name, *args, **kwargs):
    """
    Helper function for issuing singleton warnings.

    Inputs:
        warning_name        The identifier for this warning
        *args, **kwargs     The arguments for the log() call
    """
    if not warnings_issued[warning_name]:
        log(*args, **kwargs)
        warnings_issued[warning_name] = True

def error(text):
    log(text)
    raise UserWarning(text)
