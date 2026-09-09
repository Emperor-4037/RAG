"""
slowapi rate-limiting middleware configuration.
Limits are applied per remote IP address.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
