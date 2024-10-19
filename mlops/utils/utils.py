from datetime import datetime


def generate_run_name(model_name, hyperparameters):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Select up to 3 important hyperparameters
    important_params = []
    for k, v in hyperparameters.items():
        if len(important_params) < 3:
            # Remove 'estimator__' prefix if present
            param_name = k.replace('estimator__', '')
            important_params.append(f"{param_name[:3]}_{v}")

    param_str = "_".join(important_params)

    # Limit the total length of the run name
    max_length = 100
    base_name = f"{model_name}_{param_str}_{timestamp}"
    if len(base_name) > max_length:
        base_name = base_name[:max_length - 3] + "..."

    return base_name
