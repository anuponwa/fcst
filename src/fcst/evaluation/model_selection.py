from ..common.types import ModelResults


def select_best_models(model_results: ModelResults, top_n: int = 2) -> list[str]:
    """Returns Top N model names

    Parameters
    ----------
    model_results (ModelResults): Sorted model results dictionary

    top_n (int): Top N models to return

    Returns
    -------
    list: A list of Top N model names
    """

    return list(model_results.keys())[:top_n]
