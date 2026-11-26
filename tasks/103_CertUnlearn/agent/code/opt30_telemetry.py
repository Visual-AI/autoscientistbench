"""No-op replacement for unused W&B telemetry; target metrics use the local logger."""
def init(*args, **kwargs):
    return None

def log(*args, **kwargs):
    return None
