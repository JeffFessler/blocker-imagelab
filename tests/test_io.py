import pytest

from imagelab.io import (
    barbara,
    cameraman,
    house,
    house2,
    hubble,
    kodak,
    mandrill,
    peppers,
)


@pytest.mark.parametrize(
    "func",
    [
        # lena,
        barbara,
        # elaine,
        hubble,
        mandrill,
        peppers,
        cameraman,
        house,
        house2,
        lambda: kodak(4),
        lambda: kodak(13),
    ],
)
def test_get_test_image(func):
    img = func()
    assert img.size > 200 * 200
