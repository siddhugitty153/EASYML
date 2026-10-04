"""
EasyML — Request Validation Middleware
-----------------------------------------
JSON schema validation decorator for API endpoints.
"""

from functools import wraps
from flask import request, jsonify


# ── Schemas ──────────────────────────────────────────────────

PIPELINE_RUN_SCHEMA = {
    'type': 'object',
    'required': ['target_col'],
    'properties': {
        'target_col': {'type': 'string', 'minLength': 1},
        'task': {'type': 'string', 'enum': ['classification', 'regression']},
        'models': {'type': 'array', 'items': {'type': 'string'}},
        'auto_explore': {'type': 'boolean'},
    },
    'additionalProperties': True,
}

MERGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'method': {'type': 'string', 'enum': ['concat', 'merge']},
        'on': {'type': 'string'},
        'how': {'type': 'string', 'enum': ['inner', 'outer', 'left', 'right']},
    },
    'additionalProperties': True,
}

WHATIF_SCHEMA = {
    'type': 'object',
    'required': ['features'],
    'properties': {
        'features': {'type': 'object'},
    },
}


# ── Decorator ────────────────────────────────────────────────

def validate_json(schema):
    """Decorator that validates request.json against a JSON schema."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 415

            try:
                from jsonschema import validate, ValidationError
                validate(instance=request.json, schema=schema)
            except ImportError:
                # jsonschema not installed — skip validation
                pass
            except ValidationError as e:
                return jsonify({
                    'error': 'Validation error',
                    'detail': e.message,
                    'field': '.'.join(str(p) for p in e.absolute_path) or '(root)',
                }), 422
            return f(*args, **kwargs)
        return wrapper
    return decorator
