# flake8: noqa
import numpy as np

# fmt: off
font_names = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
              'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
              's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

font_list = [  # GPL v3:  https://opengameart.org/content/8x8-ascii-bitmap-font-with-c-source
    0x0, 0x808080800080000, 0x2828000000000000, 0x287C287C280000, 0x81E281C0A3C0800, 0x6094681629060000, 0x1C20201926190000,
    0x808000000000000, 0x810202010080000, 0x1008040408100000, 0x2A1C3E1C2A000000, 0x8083E08080000, 0x81000, 0x3C00000000, 0x80000,
    0x204081020400000, 0x1824424224180000, 0x8180808081C0000, 0x3C420418207E0000, 0x3C420418423C0000, 0x81828487C080000, 0x7E407C02423C0000,
    0x3C407C42423C0000, 0x7E04081020400000, 0x3C423C42423C0000, 0x3C42423E023C0000, 0x80000080000, 0x80000081000, 0x6186018060000,
    0x7E007E000000, 0x60180618600000, 0x3844041800100000, 0x3C449C945C201C, 0x1818243C42420000, 0x7844784444780000, 0x3844808044380000,
    0x7844444444780000, 0x7C407840407C0000, 0x7C40784040400000, 0x3844809C44380000, 0x42427E4242420000, 0x3E080808083E0000,
    0x1C04040444380000, 0x4448507048440000, 0x40404040407E0000, 0x4163554941410000, 0x4262524A46420000, 0x1C222222221C0000,
    0x7844784040400000, 0x1C222222221C0200, 0x7844785048440000, 0x1C22100C221C0000, 0x7F08080808080000, 0x42424242423C0000,
    0x8142422424180000, 0x4141495563410000, 0x4224181824420000, 0x4122140808080000, 0x7E040810207E0000, 0x3820202020380000,
    0x4020100804020000, 0x3808080808380000, 0x1028000000000000, 0x7E0000, 0x1008000000000000, 0x3C023E463A0000, 0x40407C42625C0000,
    0x1C20201C0000, 0x2023E42463A0000, 0x3C427E403C0000, 0x18103810100000, 0x344C44340438, 0x2020382424240000, 0x800080808080000,
    0x800180808080870, 0x20202428302C0000, 0x1010101010180000, 0x665A42420000, 0x2E3222220000, 0x3C42423C0000, 0x5C62427C4040,
    0x3A46423E0202, 0x2C3220200000, 0x1C201804380000, 0x103C1010180000, 0x2222261A0000, 0x424224180000, 0x81815A660000, 0x422418660000,
    0x422214081060, 0x3C08103C0000, 0x1C103030101C0000, 0x808080808080800, 0x38080C0C08380000, 0x324C000000,
]
# fmt: on

bitmap_letter = {}
for flong in font_list:
    fstr = format(flong, "064b")
    ffloat = [float(f) for f in fstr]
    f_np = np.array(ffloat).reshape((8, 8))
    f_name = font_names.pop(0)
    bitmap_letter[f_name] = f_np


def continuous_letter(a):
    def continuous_a(y, x):
        x_interp = np.rint(x).astype(int)
        y_interp = np.rint(y).astype(int)
        mask = (
            (x_interp >= 0) * (x_interp < 8) * (y_interp >= 0) * (y_interp < 8)
        ).astype(int)
        return mask * bitmap_letter[a][mask * y_interp, mask * x_interp]

    return continuous_a


def bitmap_string(s):
    s = s.replace("\t", "    ")
    lines = s.split("\n")
    lin_len = max([len(line) for line in lines])
    res = []
    for line in lines:
        while len(line) < lin_len:
            line += " "
        res_line = bitmap_letter[line[0]]
        for c in line[1:]:
            res_line = np.hstack((res_line, bitmap_letter[c]))
        res.append(res_line)
    return np.vstack(res)
