Broadcasting Noise
==================


How often have you written something like

.. code::

    import numpy as np
    y = A@x
    y = y + 0.01*(y.max()-y.min())*np.random.randn(*y.shape)

imagelab provides a noise object that automagically scales and broadcasts. All of the following are equivalent to the above:

.. code::

    from imagelab import noise
    y = A@x + noise('20db')
    y = A@x + noise('1%')
    y = A@x + noise(0.01, autoscale=True)
    y = A@x + 0.01*noise(autoscale=True)
    y = A@x + 0.01*noise('100%')
    y = A@x + 0.01*noise

optionally if you didn't want autoscaling...

.. code::

    y = A@x + 0.01*noise()

noise represents an **instance** of a random variable. If used repeatedly, it will give the same results. Use `~` to get a new instance with the same parameters.

.. code::

    z = np.arange(1,26).reshape(5,5)
    eta = noise
    eps = ~noise
    assert z + noise == z + eta
    assert z + noise != z + eps
