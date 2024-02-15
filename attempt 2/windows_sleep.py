import ctypes
import time

# Define constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_sleep():
    """
    Prevents the system from going to sleep.
    """
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def allow_sleep():
    """
    Allows the system to go to sleep.
    """
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS)
