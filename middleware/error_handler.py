"""
EasyML — Centralized Error Handling
--------------------------------------
Registers Flask error handlers that return consistent JSON responses
instead of HTML error pages. Also logs errors with tracebacks.
"""

import logging
import traceback

from flask import jsonify

logger = logging.getLogger('easyml.errors')


def register_error_handlers(app):
    """Register all HTTP error handlers on the Flask app."""

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({
            'error': 'Bad request',
            'detail': str(e.description if hasattr(e, 'description') else e),
        }), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({'error': 'Method not allowed'}), 405

    @app.errorhandler(413)
    def too_large(e):
        max_bytes = app.config.get('MAX_CONTENT_LENGTH', 'unknown')
        return jsonify({
            'error': 'File too large',
            'max_bytes': max_bytes,
        }), 413

    @app.errorhandler(429)
    def rate_limited(e):
        return jsonify({
            'error': 'Rate limit exceeded. Please try again later.',
        }), 429

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal server error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500
