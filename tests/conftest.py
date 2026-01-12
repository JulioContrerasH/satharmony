"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset numpy random state before each test."""
    np.random.seed(None)
    yield
