"""
Copyright (c) 2020 Simon Donné, Max Planck Institute for Intelligent Systems, Tuebingen, Germany

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
    """
    Helper function for raising an error, but only after logging it correctly.

    Inputs:
        text                The error message to print
    """
    log(text)
    raise UserWarning(text)
