from functools import wraps
from time import time
import logging

logger = logging.getLogger(__name__)

def time_it(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info("Function: %r took: %.2f sec to complete", f.__name__, te - ts)
        return result
    return wrap