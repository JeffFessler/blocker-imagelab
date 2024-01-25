"""imagelab/optim/hpo.py
Hyper-Parameter Optimization

"""
import bisect
from datetime import datetime
import itertools as it
import socket
import sys
import time

import numpy as np

# from ..utils import export


def grid_points(*args):
    params = []
    for _arg in args:
        param = None
        params.append(param)
    points = it.product(*params)
    return points


class HyperTuner(object):
    """docstring for HPOTuner"""

    def __init__(
        self,
        runner,
        spaces,
        method,
        val_set,
        max_evals=1e10,
        max_time=4e7,
        max_exceptions=0.5,
        batch_size=1,
        metadata=None,
        out_file=None,
        verbose=False,
    ):
        super(HyperTuner, self).__init__()
        # args
        self.runner = runner
        self.spaces = spaces
        if method.lower() not in ["grid", "random"]:
            raise ValueError(f"method {method} not recognized")
        self.method = method.lower()
        self.val_set = val_set
        self.max_evals = max_evals
        self.max_time = max_time
        self.max_exceptions = max_exceptions
        self.batch_size = batch_size
        self.metadata = metadata if metadata is not None else {}
        self.out_file = out_file
        self.verbose = verbose

        # state
        self.rstate = np.random.RandomState()
        self.history = []
        self.eval_count = 0
        self.exceptions = []
        self.fh = None

    def init_metadata(self):
        self.metadata["host"] = socket.gethostname()
        self.metadata["date"] = str(datetime.now())
        self.metadata["pyversion"] = sys.version

    def tune(self):
        self.init_metadata()
        if self.out_file is not None:
            self.fh = open(self.out_file, "w+")
        start_time = time.time()
        elapsed = 0
        try:
            while (
                (self.eval_count < self.max_evals)
                and (elapsed < self.max_time)
                and (len(self.exceptions) < self.max_exceptions)
            ):

                params = self.select_params()
                losses = self.process_val_set(params)
                loss = np.mean(losses)
                self.record_result(loss, params)
                elapsed = time.time() - start_time

        except KeyboardInterrupt:
            if self.fh is not None:
                self.fh.close()
                self.fh = None
        self.elapsed = elapsed
        return self.history[-1]

    def select_params(self):
        if self.method == "random":
            return None
        if self.method == "grid":
            return None
        if self.method == "tpe":
            return None
        if self.method == "hyperband":
            return None
        else:
            raise ValueError(f"Unknown method {self.method}")

    def process_val_set(self, params):
        losses = []
        try:
            for batch in zip(*[iter(self.val_set)] * self.batch_size):
                loss = self.runner(batch, params)
                self.eval_count += 1
                losses.append(loss)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            self.exceptions.append([e, params])
        return losses

    def record_result(self, loss, params):
        bisect.insort(self.history, [loss, params])
        if self.fh is not None:
            self.fh.write(f"{loss}\t{params}\n")

    # def __getstate__(self):
    #     """ For supporting pickling"""
    #     raise NotImplementedError

    # def __setstate__(self, state):
    #     """ For supporting pickling"""
    #     raise NotImplementedError
