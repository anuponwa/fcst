# -*- coding: utf-8 -*-
"""
automation sub-package
~~~~
Automation, make it easy for common forecasting tasks.
"""

from ._forecasting_automation import (
    run_forecasting_automation,
    run_forecasting_pipeline,
)

__all__ = ["run_forecasting_automation", "run_forecasting_pipeline"]
