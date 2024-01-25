"""imagelab/misc.py

Miscellaneous functions
"""

# https://codegolf.stackexchange.com/a/74047
# this might as well be machine code for clarity
import math
import unicodedata


def ordinal(n):
    return f"{n}{'tsnrhtdd'[n%5*(n%100^15>4>n%10)::4]}"


def unicode_to_ascii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
        # and c in all_letters
    )


def ansi_color_code_chart():
    print("ANSI COLOR CODES")
    print("_" * 70)
    for ii in range(18):
        print(
            f"{ii+00:3d}:\x1b[{ii+00}m Hello,\x1b[0m World!\t"
            + f"{ii+30:3d}:\x1b[{ii+30}m Hello,\x1b[0m World!\t"
            + f"{ii+90:3d}:\x1b[{ii+90}m Hello,\x1b[0m World!"
        )


def precision_and_scale(x):
    # https://stackoverflow.com/a/3019027/5026175
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    if magnitude >= max_digits:
        return (magnitude, 0)
    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    scale = int(math.log10(frac_digits))
    return (magnitude + scale, scale)
