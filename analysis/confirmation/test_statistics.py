import numpy as np
from analyze import inference, holm, eligible


def test_holm_known_family():
    assert np.allclose(holm([0.04, 0.001, 0.03]), [0.06, 0.003, 0.06])


def test_exact_all_positive_effect_and_zero_effect():
    effect, lo, hi, p, exact, n = inference([1, 1, 1, 1], 1)
    assert (effect, lo, hi, p, exact, n) == (1, 1, 1, 0.125, True, 16)
    assert inference([0, 0, 0], 1)[3] == 1


def test_missing_recovery_is_excluded():
    assert not eligible({'condition': 'fault', 'applied': True, 'raw_recovery_detected': None}, 'recorded_recovery')
    assert eligible({'condition': 'fault', 'applied': True, 'raw_recovery_detected': False}, 'recorded_recovery')
