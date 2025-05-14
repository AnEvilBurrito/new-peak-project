from joblib import Parallel, delayed
from functools import wraps

def parallelize_joblib(n_jobs=-1, backend='loky'):
    """Decorator to parallelize function calls over an iterable using joblib"""
    def decorator(func):
        @wraps(func)
        def wrapper(iterable):
            return Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(func)(item) for item in iterable
            )
        return wrapper
    return decorator
