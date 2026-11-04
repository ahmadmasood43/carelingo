from carelingo.stats import proportion, wilson_interval


def test_zero_events_still_have_an_upper_bound():
    low, high = wilson_interval(0, 48)
    assert low == 0 and 0.06 < high < 0.08


def test_known_value():
    low, high = wilson_interval(5, 24)
    assert round(low, 3) == 0.092 and round(high, 3) == 0.405


def test_proportion_formatting():
    assert proportion(1, 24).startswith("1/24 (4%, CI 1%–")
