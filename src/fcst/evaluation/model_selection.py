from typing import Tuple

import numpy as np

from ..common.types import ModelResults


def select_best_models(
    model_results: ModelResults, top_n: int = 2
) -> list[str | Tuple[str, str]]:
    """Returns Top N model names

    Parameters
    ----------
        model_results (ModelResults): Sorted model results dictionary
            It can be in this form: {"model1": 0.21, "model2": 0.18}, or
            It can contain tags like: {"model1": (0.21, "1_univatiate"), "model2": (0.18, "2_multivariate)}.

        top_n (int): Top N models to return

    Returns
    -------
        list:
            A list of Top N model names, or
            A list of Top N model names with their tag
    """

    # Check type
    a_value = list(model_results.values())[0]
    # No tag
    if isinstance(a_value, float | int):
        no_tag = True
    else:
        no_tag = False

    # Filter out nan
    # No tag
    if no_tag:
        model_results = dict(filter(lambda x: ~np.isnan(x[1]), model_results.items()))

    # With tag
    else:
        model_results = dict(
            filter(lambda x: ~np.isnan(x[1][0]), model_results.items())
        )

    model_results = dict(sorted(model_results.items(), key=lambda x: x[1]))

    # No tag
    if no_tag:
        return list(model_results.keys())[:top_n]

    # With tag
    else:
        results_list = [(k, v[1]) for k, v in model_results.items()]
        return results_list[:top_n]
