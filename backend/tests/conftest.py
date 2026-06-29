"""Shared fixtures and generators for backend tests."""

import os
import sys

import pytest

# Ensure the backend directory is on the path so tests can be run from the project root
_backend_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_backend_dir))
