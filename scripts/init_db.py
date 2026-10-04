#!/usr/bin/env python3
"""
EasyML — Database Initialization Script
========================================
Run inside the container:
    docker compose exec web python scripts/init_db.py

Creates tables if they don't exist. Safe to run multiple times.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db


def main():
    print("🔧 Initializing EasyML database...")
    try:
        init_db()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
