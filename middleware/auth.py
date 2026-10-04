"""
EasyML — API Key Authentication Middleware
---------------------------------------------
Optional API key guard. When EASYML_API_KEY is set in the environment,
all protected endpoints require a matching X-API-Key header.
When not set, endpoints remain open (backwards-compatible).
"""

import os
from functools import wraps
from flask import request, jsonify

API_KEY = os.environ.get('EASYML_API_KEY')


def require_api_key(f):
    """
    Decorator that checks for a valid API key.

    The key can be provided via:
      - X-API-Key header (preferred)
      - ?api_key= query parameter (fallback)

    If EASYML_API_KEY is not configured, all requests pass through.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not API_KEY:
            # No key configured — open access (dev mode)
            return f(*args, **kwargs)

        provided_key = (
            request.headers.get('X-API-Key')
            or request.args.get('api_key')
        )
        if provided_key != API_KEY:
            return jsonify({'error': 'Invalid or missing API key'}), 401

        return f(*args, **kwargs)
    return wrapper
