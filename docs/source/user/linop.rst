Linear Operators (LinOps)
=========================

LinOps provide the functionality of a matrix, but without the memory requirements.
.. code::

    import imagelab as il
    import numpy as np
    diag = np.random.rand(5) + np.random.rand(5)*1j
    A = il.linop.Diagonal(diag)
    y = A@x # computes diag*x under the hood
    y = A.H@x # computes conj(diag)*x under the hood
    assert A.shape == (5,5)


Similar classes exist in other packages and languages. See also:

- `scipy.sparse.linalg.LinearOperator <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_
- `PyLops <https://github.com/equinor/pylops>`_ (built ontop of scipy.sparse.linalg.LinearOperator and actively developed)
- `linop <https://pythonhosted.org/linop/index.html>`_ (abandoned?)
- `PyOp <https://github.com/ryanorendorff/pyop>`_ (abandoned?)
- Fatrices in the `MIRT <https://web.eecs.umich.edu/~fessler/code/>`_ toolbox for MATLAB
- `Spot <http://www.cs.ubc.ca/labs/scl/spot/>`_ (MATLAB)
- `LinearOperator.jl <https://github.com/JuliaSmoothOptimizers/LinearOperators.jl>`_ (Julia)
- `LinearMap.jl <https://github.com/Jutho/LinearMaps.jl>`_ (Julia)  

The main difference this package has with many of these is that it does not require that the input and outputs of a LinOp to be 1-dimensional arrays. The philosophy of this package is that all arrays, regardless of shape, are vectors belonging to some vector space, so there is no need to force things to be 1D. That said, when working with linear algebra methods, it is necessary for shapes to match up and so setting ``vec_out=True`` will give that effect. A decorator is also provided that will temporarily set this attribute within functions ``@linop.util.ensure_vec('arg_name')``.