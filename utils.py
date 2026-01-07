"""AutoRewardDrive: Utility Functions"""

import json
import math


def write_json(data, path):
    """Save configuration to JSON file"""
    config_dict = {}
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in data.items():
            if isinstance(v, str) and v.isnumeric():
                config_dict[k] = int(v)
            elif isinstance(v, dict):
                config_dict[k] = {k_inner: str(v_inner) for k_inner, v_inner in v.items()}
                config_dict[k] = str(config_dict[k])
            else:
                config_dict[k] = str(v)
        json.dump(config_dict, f, indent=4)


def lr_schedule(initial_value: float, end_value: float, rate: float):
    """
    Learning rate schedule with exponential decay
    """
    def func(progress_remaining: float) -> float:
        if progress_remaining <= 0:
            return end_value
        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))

    func.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
    return func
