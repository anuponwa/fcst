# -*- coding: utf-8 -*-
"""
evaluation sub-package
~~~~
Provides evaluation framework and methods including back-testing,
simulating past performance and model selection.
"""

from .model_evaluation import backtest_evaluate, get_backtest_periods
from .model_selection import select_best_models

__all__ = ["get_backtest_periods", "backtest_evaluate", "select_best_models"]
