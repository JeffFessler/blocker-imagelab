def huberhinge_majorizer(s, delta=0.1):
    r"""
    Syntax: coef = huberhinge_sol(s; delta=?)

    Coefficients of optimal quadratic majorizer q(t;s) for Huber hinge function:
    h(t; \delta) = \leftbrace{
    1 - t - \delta/2, & t \leq 1 - \delta \\
    \frac{1}{2\delta} (t - 1)^2, & 1 - \delta \leq t \leq 1 \\
    0, & 1 \leq t }


    in
    s     expansion point for which q(s;s) = h(s;\delta)

    option
    delta     0 < delta, corner rounding value; default 0.1

    out
    coef  [c0,c1,c2] for q(t;s) = c0 + c1 (t - s) + c2/2 (t-s)^2
    """

    if s < 1 - delta:
        return [1 - s - delta / 2, -1, 1 / (2 * (1 - s - delta / 2))]
    elif s < 1:
        return [1 / (2 * delta) * (s - 1) ** 2, (s - 1) / delta, 1 / delta]
    else:
        return [0, 0, 1 / (2 * (s - 1 + delta / 2))]  # 1 < s
