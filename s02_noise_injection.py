import numpy as np


def inject_symmetric_noise(y_true, noise_rate, n_classes, random_state=None):
    assert 0.0 <= noise_rate <= 1.0
    assert y_true.ndim == 1
    assert y_true.min() >= 0 and y_true.max() < n_classes
    rng = np.random.RandomState(random_state)
    y_noisy = y_true.copy()
    flip_mask = rng.rand(len(y_true)) < noise_rate
    for idx in np.where(flip_mask)[0]:
        true_label = y_true[idx]
        candidates = [c for c in range(n_classes) if c != true_label]
        y_noisy[idx] = rng.choice(candidates)
    return y_noisy, flip_mask


def inject_asymmetric_noise(y_true, noise_rate, n_classes, transition_map=None, random_state=None):
    assert 0.0 <= noise_rate <= 1.0
    assert y_true.ndim == 1
    assert y_true.min() >= 0 and y_true.max() < n_classes
    if transition_map is None:
        transition_map = {c: (c + 1) % n_classes for c in range(n_classes)}
    rng = np.random.RandomState(random_state)
    y_noisy = y_true.copy()
    flip_mask = rng.rand(len(y_true)) < noise_rate
    for idx in np.where(flip_mask)[0]:
        true_label = y_true[idx]
        if true_label in transition_map:
            y_noisy[idx] = transition_map[true_label]
    actual_flip_mask = (y_noisy != y_true)
    return y_noisy, actual_flip_mask


NOISE_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40]
