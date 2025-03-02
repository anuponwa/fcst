# -*- coding: utf-8 -*-
"""
forecasting sub-package
~~~~
Provides all the useful functionalities about forecasting.
"""

from .ensemble import ensemble_forecast
from .forecasting import forecast

__all__ = ["ensemble_forecast", "forecast"]
