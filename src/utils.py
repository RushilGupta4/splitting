def validate_split_percentages(split_percentages):
    """Validate split percentages expressed as fractions of sampling steps."""
    if len(split_percentages) == 0:
        raise ValueError("split_percentages must have at least one element")

    for i, percentage in enumerate(split_percentages):
        if percentage <= 0.0 or percentage >= 1.0:
            raise ValueError(
                f"split_percentages[{i}]={percentage} must be in the open interval (0, 1)"
            )

    for i in range(len(split_percentages) - 1):
        if split_percentages[i] <= split_percentages[i + 1]:
            raise ValueError(
                "split_percentages must be strictly decreasing, "
                f"got split_percentages[{i}]={split_percentages[i]} <= "
                f"split_percentages[{i + 1}]={split_percentages[i + 1]}"
            )
