import numpy as np

from imagelab.optim.mm import huberhinge_majorizer


def test_huberhinge_majorizer():
    huberhinge = (
        lambda t, delta: (t < 1 - delta) * (1 - t - delta / 2)
        + (t < 1) * (t >= 1 - delta) * (t - 1) ** 2 / 2 / delta
    )
    delta_set = [0.4, 0.9, 1.0, 3.0]
    t = np.linspace(-10, 10, 500)
    for delta in delta_set:
        s_set = [
            (1 - delta) - 2,  # s < 1 - delta
            1 - delta,  # s = 1 - delta
            1 - delta / 3,  # 1-delta < s < 1
            1,  # s = 1
            7,
        ]  # s > 1
        for s in s_set:
            c0, c1, c2 = huberhinge_majorizer(s, delta)

            def maj(t, s):
                return c0 + c1 * (t - s) + 0.5 * c2 * (t - s) ** 2

            # Is it a majorizer?
            diff = maj(t, s) - huberhinge(t, delta)
            assert (
                diff > -1e-15
            ).all(), f"delta={delta}, s={s}, bad_values={diff[diff<=-1e-15]}"
            # Is it equal at the current point?
            assert np.isclose(maj(s, s), huberhinge(s, delta)), f"delta={delta}, s={s}"

            # Test equality at second point? we'd need to commute it...
