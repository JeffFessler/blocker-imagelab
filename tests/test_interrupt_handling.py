import os
import signal
from unittest.mock import Mock

from imagelab.utils import DelayedInterrupt


def test_delayed_interrupt_with_one_signal():
    # check behavior without DelayedInterrupt
    a = Mock()
    b = Mock()
    c = Mock()
    try:
        a()
        os.kill(os.getpid(), signal.SIGINT)
        b()
    except KeyboardInterrupt:
        c()
    a.assert_called_with()
    b.assert_not_called()
    c.assert_called_with()

    # test behavior with DelayedInterrupt
    a = Mock()
    b = Mock()
    c = Mock()
    d = Mock()
    try:
        with DelayedInterrupt(signal.SIGINT):
            a()
            os.kill(os.getpid(), signal.SIGINT)
            b()
        d()
    except KeyboardInterrupt:
        c()
    a.assert_called_with()
    b.assert_called_with()
    c.assert_called_with()
    d.assert_not_called()


def test_delayed_interrupt_with_multiple_signals():
    a = Mock()
    b = Mock()
    c = Mock()
    d = Mock()
    try:
        with DelayedInterrupt([signal.SIGTERM, signal.SIGINT]):
            a()  # This runs since we haven't had an interrupt
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGTERM)
            b()  # This also runs since the interrupt is delayed
        d()  # This does not run, since the interrupt was raised
    except KeyboardInterrupt:
        c()  # This runs since the interrupt was raised
    a.assert_called_with()
    b.assert_called_with()
    c.assert_called_with()
    d.assert_not_called()


# def test_irange_delayed_interrupt():
#     i0 = Mock()
#     i1 = Mock()
#     i2 = Mock()
#     ex = Mock()
#     i = [i0, i1, i2]
#     try:
#         for ii in irange(3, delay_interrupt=True, pbar=False):
#             if ii == 1:
#                 os.kill(os.getpid(), signal.SIGINT)
#             i[ii]()
#     except KeyboardInterrupt:
#         ex() # Interrupt was suppressed, so no call

#     i0.assert_called_with()
#     i1.assert_called_with() # Called since iteration finished before exiting
#     i2.assert_not_called() # Never called since interrupt stopped iterations
#     ex.assert_not_called()

# def test_irange_raised_interrupt():
#     i0 = Mock()
#     i1 = Mock()
#     i2 = Mock()
#     ex = Mock()
#     tests = [i0, i1, i2]
#     try:
#         for ii in irange(3, delay_interrupt='raise', pbar=False):
#             if ii == 1: # the 2nd iteration
#                 os.kill(os.getpid(), signal.SIGINT)
#             tests[ii]()
#     except KeyboardInterrupt:
#         ex() # Interrupt was raised, so called

#     i0.assert_called_with()
#     i1.assert_called_with() # Called since iteration finished before exiting
#     i2.assert_not_called() # Never called since interrupt stopped iterations
#     ex.assert_called_with()

# def test_irange_double_interrupt():
#     i0 = Mock()
#     i1 = Mock()
#     i2 = Mock()
#     ex = Mock()
#     b = Mock()
#     i = [i0, i1, i2]
#     try:
#         for ii in irange(3, delay_interrupt=True, pbar=False):
#             if ii == 1:
#                 os.kill(os.getpid(), signal.SIGINT)
#                 b() # Make sure the first wasn't raised
#                 os.kill(os.getpid(), signal.SIGINT) # Force the interrupt up immediatly
#             i[ii]()
#     except KeyboardInterrupt:
#         ex() # 1st interrupt was suppressed, 2nd was not

#     i0.assert_called_with()
#     i1.assert_not_called() # Never called since we exited immediatly
#     i2.assert_not_called() # Never called since interrupt stopped iterations
#     ex.assert_called_with()
#     b.assert_called_with()

# def test_double_irange_interrupts(): #important for opt methods that call inner opt methods
#     i0 = Mock()
#     i1 = Mock()
#     i2 = Mock()
#     ex = Mock()
#     b = Mock()
#     tests = [i0, i1, i2]
#     try:
#         for ii in irange(2, delay_interrupt=True, pbar=False):
#             for jj in irange(3, delay_interrupt='raise', pbar=False):
#                 if jj == 1: # the 2nd iteration
#                     os.kill(os.getpid(), signal.SIGINT)
#                 tests[jj]()
#             b() # should be called exactly once on first iteration
#     except KeyboardInterrupt:
#         ex() # Interrupt should be raised, but then suppressed

#     i0.assert_called_once()
#     i1.assert_called_once() # Called since iteration finished before exiting
#     i2.assert_not_called() # Never called since interrupt stopped iterations
#     b.assert_called_once() # Should have ran for i == 0 but i==1 never happened
#     ex.assert_not_called()
