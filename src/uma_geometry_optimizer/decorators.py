from functools import wraps
from time import time

def time_it(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('Function:%r took: %.2f sec to complete' % \
          (f.__name__, te-ts))
        return result
    return wrap