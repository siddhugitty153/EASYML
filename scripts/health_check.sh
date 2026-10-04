#!/bin/bash
# ============================================================
# EasyML — Container Health Check
# ============================================================
# Used by Docker HEALTHCHECK to verify the app is responding.
# ============================================================

curl -f http://localhost:5000/ || exit 1
