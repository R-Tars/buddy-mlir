"""Compatibility alias for the diagnostics-only legacy validation workflow."""

import sys

from .diagnostics import validation_workflow as _validation_workflow


sys.modules[__name__] = _validation_workflow
