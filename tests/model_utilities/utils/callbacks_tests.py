import unittest

import torchbearer
from torchbearer import Trial
from torchbearer.callbacks import Callback

from model_utilities.utils import AtTrainingEpochs, AtTrainingIterations, \
    EveryNBatches, EveryNEpochs, at_training_epochs, \
    at_training_iterations, every_n_batches, every_n_epochs


class CountingCallback(Callback):
    def __init__(self):
        self.calls = []

    def on_step_training(self, state):
        self.calls.append(("batch", state[torchbearer.BATCH]))

    def on_end_epoch(self, state):
        self.calls.append(("epoch", state[torchbearer.EPOCH]))


class TestCallbacks(unittest.TestCase):
    def test_every_n_batches_decorator(self):
        calls = []

        @every_n_batches(2)
        def callback(state):
            calls.append(state[torchbearer.BATCH])

        Trial(None, callbacks=[callback]).for_steps(5).run(1, verbose=0)

        self.assertEqual([1, 3], calls)

    def test_every_n_batches_can_include_first(self):
        calls = []

        @every_n_batches(3, include_first=True)
        def callback(state):
            calls.append(state[torchbearer.BATCH])

        Trial(None, callbacks=[callback]).for_steps(7).run(1, verbose=0)

        self.assertEqual([0, 2, 5], calls)

    def test_every_n_epochs_decorator(self):
        calls = []

        @every_n_epochs(2)
        def callback(state):
            calls.append(state[torchbearer.EPOCH])

        Trial(None, callbacks=[callback]).for_steps(1).run(5, verbose=0)

        self.assertEqual([1, 3], calls)

    def test_every_n_batches_wraps_callback(self):
        counting_callback = CountingCallback()
        callback = EveryNBatches(2, counting_callback)

        Trial(None, callbacks=[callback]).for_steps(5).run(1, verbose=0)

        self.assertEqual([("batch", 1), ("batch", 3)],
                         counting_callback.calls)

    def test_every_n_batches_does_not_run_during_validation(self):
        calls = []

        @every_n_batches(1)
        def callback(state):
            calls.append(state[torchbearer.BATCH])

        Trial(None, callbacks=[callback]).for_steps(2).for_val_steps(3).run(
            1, verbose=0)

        self.assertEqual([0, 1], calls)

    def test_every_n_epochs_wraps_callback(self):
        counting_callback = CountingCallback()
        callback = EveryNEpochs(2, counting_callback)

        Trial(None, callbacks=[callback]).for_steps(1).run(5, verbose=0)

        self.assertEqual([("epoch", 1), ("epoch", 3)],
                         counting_callback.calls)

    def test_at_training_iterations_decorator(self):
        calls = []

        @at_training_iterations([0, 1, 2, 4, 8])
        def callback(state):
            calls.append(state[torchbearer.BATCH])

        Trial(None, callbacks=[callback]).for_steps(5).run(2, verbose=0)

        self.assertEqual([0, 1, 2, 4, 3], calls)

    def test_at_training_iterations_accepts_generator(self):
        calls = []

        def powers_of_two():
            value = 1
            while True:
                yield value
                value *= 2

        @at_training_iterations(powers_of_two())
        def callback(state):
            calls.append(state[torchbearer.BATCH])

        Trial(None, callbacks=[callback]).for_steps(5).run(2, verbose=0)

        self.assertEqual([1, 2, 4, 3], calls)

    def test_at_training_iterations_wraps_callback(self):
        counting_callback = CountingCallback()
        callback = AtTrainingIterations([0, 2], counting_callback)

        Trial(None, callbacks=[callback]).for_steps(4).run(1, verbose=0)

        self.assertEqual([("batch", 0), ("batch", 2)],
                         counting_callback.calls)

    def test_at_training_epochs_decorator(self):
        calls = []

        @at_training_epochs([0, 1, 3])
        def callback(state):
            calls.append(state[torchbearer.EPOCH])

        Trial(None, callbacks=[callback]).for_steps(1).run(5, verbose=0)

        self.assertEqual([0, 1, 3], calls)

    def test_at_training_epochs_accepts_predicate(self):
        calls = []

        @at_training_epochs(lambda epoch: epoch < 2 or epoch % 3 == 0)
        def callback(state):
            calls.append(state[torchbearer.EPOCH])

        Trial(None, callbacks=[callback]).for_steps(1).run(7, verbose=0)

        self.assertEqual([0, 1, 3, 6], calls)

    def test_at_training_epochs_wraps_callback(self):
        counting_callback = CountingCallback()
        callback = AtTrainingEpochs([0, 2], counting_callback)

        Trial(None, callbacks=[callback]).for_steps(1).run(4, verbose=0)

        self.assertEqual([("epoch", 0), ("epoch", 2)],
                         counting_callback.calls)

    def test_invalid_period(self):
        with self.assertRaises(ValueError):
            every_n_batches(0)(lambda state: None)

        with self.assertRaises(TypeError):
            every_n_epochs(1.5)(lambda state: None)

    def test_invalid_schedule(self):
        with self.assertRaises(ValueError):
            Trial(None, callbacks=[
                at_training_epochs([2, 1])(lambda state: None)
            ]).for_steps(1).run(3, verbose=0)

        with self.assertRaises(TypeError):
            Trial(None, callbacks=[
                at_training_iterations([1.5])(lambda state: None)
            ]).for_steps(2).run(1, verbose=0)


if __name__ == '__main__':
    unittest.main()
