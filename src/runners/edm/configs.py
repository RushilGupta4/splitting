_EDM_PARAMS = {
    "S_churn": 40.0,
    "S_min": 0.0,
    "S_max": 80.0,
    "S_noise": 1.0,
}

_DPMPP_2S_PARAMS = {
    "stochastic_churn_rate": 20,
    "churn_min_noise_level": 0.05,
    "churn_max_noise_level": 50.0,
    "noise_level_inflation_factor": 1.0,
}

__all__ = ["_DPMPP_2S_PARAMS", "_EDM_PARAMS"]
