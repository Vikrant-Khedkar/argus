"""Backward-compat shim. Argus v2 moved this module into the `argus` package.

For new code, prefer:
    from argus import log_row, read_log, new_request_id
"""

from argus.observability import *  # noqa: F401, F403
from argus.observability import log_row, read_log, new_request_id, LOG_PATH  # noqa: F401
