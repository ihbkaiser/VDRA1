"""Small runtime helpers used by the TreePO integration patches."""

import numpy as np


def random_fallback_block(total_blocks: int, max_fallback_window: int = -1) -> int:
    """Choose TreePO's random fallback block, including empty responses.

    An empty response has no block to sample, so block zero maps to fallback
    index zero. For non-empty responses this matches TreePO's original range.
    """
    if total_blocks <= 0:
        return 0
    choice_limit = (
        min(max_fallback_window, total_blocks)
        if max_fallback_window > 0
        else total_blocks
    )
    return int(np.random.randint(1, choice_limit + 1))


def concat_non_tensor_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Concatenate row-wise metadata while preserving ragged per-row values.

    Tree rollouts can produce nested object arrays whose trailing dimensions
    differ between generation batches. DataProto's non-tensor fields are
    row-oriented, so in that case preserve each row as one object instead of
    asking NumPy to align incompatible nested shapes.
    """
    values = [np.asarray(array) for array in arrays]
    if not values:
        return np.empty((0,), dtype=object)
    if any(value.ndim == 0 for value in values):
        raise ValueError("non-tensor batch values must have a leading row dimension")
    if len({value.shape[1:] for value in values}) == 1:
        return np.concatenate(values, axis=0)

    result = np.empty((sum(value.shape[0] for value in values),), dtype=object)
    offset = 0
    for value in values:
        for row in value:
            result[offset] = row
            offset += 1
    return result
