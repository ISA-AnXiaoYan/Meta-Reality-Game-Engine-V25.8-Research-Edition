# SPDX-License-Identifier: AGPL-3.0-only
"""Public-safe adapters only: mock and file."""

from .null_adapter import NullAdapter
from .null_bridge import NullBridge

__all__ = ["NullAdapter", "NullBridge"]
