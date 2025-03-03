# -*- coding: utf-8 -*-
"""
preprocessing sub-package
~~~~
Provides all the useful functionalities about data
and time-series preprocessing before feeding to the model.
"""

from ._preprocessing import prepare_timeseries

__all__ = ["prepare_timeseries"]