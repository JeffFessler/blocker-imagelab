import numpy as np

from ..utils import export
from .base import AbstractLinOp


@export
class VStackLinOp(AbstractLinOp):
    def __init__(self, op_list):
        super(VStackLinOp, self).__init__()
        # prevent unnecessary compositions of compositions
        tmp_list = [
            [Op] if not isinstance(Op, VStackLinOp) else Op.op_list for Op in op_list
        ]
        self.op_list = [item for sublist in tmp_list for item in sublist]
        in_shape = (
            [op.in_shape for op in self.op_list if op.in_shape is not None] or [None]
        )[0]
        for op in self.op_list:
            if op.in_shape is None:
                op.in_shape = in_shape
                if op.out_shape is None:
                    op.out_shape = in_shape
            elif np.prod(op.in_shape) != np.prod(in_shape):
                raise ValueError(
                    "LinOps with different number of input elements can't be vertically stacked"
                )
        if None not in [op.out_shape for op in self.op_list] or in_shape is not None:
            out_shape = (
                np.sum(
                    [
                        np.prod(op.out_shape)
                        if op.out_shape is not None
                        else np.prod(in_shape)
                        for op in self.op_list
                    ]
                ),
            )
        else:
            out_shape = None
        self.in_shape = in_shape
        self.out_shape = out_shape  # should we pass these to super?

    def forward_project(self, x):
        res = self.map(lambda op: (op @ x).reshape(-1), self.op_list)
        return self.backend.concatenate(list(res))

    def back_project(self, y):
        x = 0
        yv = y
        for Op in self.op_list:
            if Op.H.in_shape is not None:
                amt = np.prod(Op.H.in_shape)
            elif self.in_shape is not None:
                amt = np.prod(self.in_shape)  # ? maybe it's square?
            else:  # everything is None then, so divide equally
                amt = y.shape[0] // len(self.op_list)
            x += Op.H @ yv[:amt]
            yv = yv[amt:]
        return x

    def abs(self):
        return VStackLinOp([Op.abs() for Op in self.op_list])


@export
class DStackLinOp(AbstractLinOp):
    def __init__(self, op_list):
        super(DStackLinOp, self).__init__()
        # prevent unnecessary compositions of compositions
        tmp_list = [
            [Op] if not isinstance(Op, DStackLinOp) else Op.op_list for Op in op_list
        ]
        self.op_list = [item for sublist in tmp_list for item in sublist]
        in_shape = (
            [op.in_shape for op in self.op_list if op.in_shape is not None] or [None]
        )[0]
        out_shape = (
            [op.out_shape for op in self.op_list if op.in_shape is not None] or [None]
        )[0]
        if out_shape is not None:
            out_shape = (len(self.op_list), *out_shape)
        self.in_shape = in_shape
        self.out_shape = out_shape  # should we pass these to super?

    def forward_project(self, x):
        res = self.map(lambda op: (op @ x), self.op_list)
        # for Op in self.op_list:
        #     res.append(Op@x)
        return self.backend.array((np.array(list(res))))

    def back_project(self, y):
        x = self.op_list[0].H @ y[0]
        for ii, Op in enumerate(self.op_list[1:]):
            x += Op.H @ y[ii + 1]
        return x

    def abs(self):
        return DStackLinOp([Op.abs() for Op in self.op_list])


@export
def HStackLinOp(op_list):
    return VStackLinOp([Op.H for Op in op_list]).H
