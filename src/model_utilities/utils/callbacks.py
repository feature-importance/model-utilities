"""Torchbearer callback helpers."""

import torchbearer

from torchbearer.callbacks import Callback


def _validate_event_index(index):
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("schedule entries must be integers")
    if index < 0:
        raise ValueError("schedule entries must be non-negative")


def _validate_period(period):
    if not isinstance(period, int) or isinstance(period, bool):
        raise TypeError("period must be an integer")
    if period <= 0:
        raise ValueError("period must be a positive integer")


def _should_run(index, period, include_first):
    return (include_first and index == 0) or (index + 1) % period == 0


class _Schedule:
    def __init__(self, schedule):
        self.schedule = schedule
        self.is_predicate = callable(schedule)
        if self.is_predicate:
            self.iterator = None
        else:
            if hasattr(schedule, "__len__"):
                schedule = list(schedule)
                _validate_schedule_entries(schedule)
            self.iterator = iter(schedule)
        self.next_index = None
        self.last_index = None
        self.exhausted = False

    def should_run(self, index, state):
        if self.is_predicate:
            try:
                return self.schedule(index, state)
            except TypeError:
                return self.schedule(index)

        self._advance_to(index)
        if self.exhausted:
            return False
        if self.next_index == index:
            self.next_index = None
            return True
        return False

    def _advance_to(self, index):
        while not self.exhausted and (
                self.next_index is None or self.next_index < index):
            try:
                next_index = next(self.iterator)
            except StopIteration:
                self.exhausted = True
                self.next_index = None
                return

            _validate_event_index(next_index)
            if self.last_index is not None and next_index < self.last_index:
                raise ValueError("schedule entries must be in ascending order")

            self.last_index = next_index
            self.next_index = next_index


def _validate_schedule_entries(schedule):
    last_index = None
    for index in schedule:
        _validate_event_index(index)
        if last_index is not None and index < last_index:
            raise ValueError("schedule entries must be in ascending order")
        last_index = index


def _call_callback(callback, callback_method, state):
    if isinstance(callback, Callback):
        return getattr(callback, callback_method)(state)
    return callback(state)


class EveryNBatches(Callback):
    """Run a callback every ``period`` training batches.

    Args:
        period: Number of batches between calls.
        callback: A callable accepting ``state`` or a Torchbearer callback.
        include_first: If True, also run on the first training batch of each
            epoch.
    """

    def __init__(self, period, callback, include_first=False):
        super().__init__()
        _validate_period(period)
        self.period = period
        self.callback = callback
        self.include_first = include_first

    def state_dict(self):
        if hasattr(self.callback, "state_dict"):
            return {"callback": self.callback.state_dict()}
        return {}

    def load_state_dict(self, state_dict):
        if hasattr(self.callback, "load_state_dict"):
            self.callback.load_state_dict(state_dict.get("callback", {}))
        return self

    def on_step_training(self, state):
        if _should_run(state[torchbearer.BATCH], self.period,
                       self.include_first):
            _call_callback(self.callback, "on_step_training", state)


class EveryNEpochs(Callback):
    """Run a callback every ``period`` training epochs.

    Args:
        period: Number of epochs between calls.
        callback: A callable accepting ``state`` or a Torchbearer callback.
        include_first: If True, also run on the first training epoch.
    """

    def __init__(self, period, callback, include_first=False):
        super().__init__()
        _validate_period(period)
        self.period = period
        self.callback = callback
        self.include_first = include_first

    def state_dict(self):
        if hasattr(self.callback, "state_dict"):
            return {"callback": self.callback.state_dict()}
        return {}

    def load_state_dict(self, state_dict):
        if hasattr(self.callback, "load_state_dict"):
            self.callback.load_state_dict(state_dict.get("callback", {}))
        return self

    def on_end_epoch(self, state):
        if _should_run(state[torchbearer.EPOCH], self.period,
                       self.include_first):
            _call_callback(self.callback, "on_end_epoch", state)


class AtTrainingIterations(Callback):
    """Run a callback on selected global training iterations.

    Args:
        schedule: Ascending iterable/generator of zero-based training iteration
            indices, or a predicate accepting ``index`` or ``index, state``.
        callback: A callable accepting ``state`` or a Torchbearer callback.
    """

    def __init__(self, schedule, callback):
        super().__init__()
        self.schedule = _Schedule(schedule)
        self.callback = callback
        self.iteration = 0

    def state_dict(self):
        state_dict = {"iteration": self.iteration}
        if hasattr(self.callback, "state_dict"):
            state_dict["callback"] = self.callback.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.iteration = state_dict.get("iteration", 0)
        if hasattr(self.callback, "load_state_dict"):
            self.callback.load_state_dict(state_dict.get("callback", {}))
        return self

    def on_step_training(self, state):
        if self.schedule.should_run(self.iteration, state):
            _call_callback(self.callback, "on_step_training", state)
        self.iteration += 1


class AtTrainingEpochs(Callback):
    """Run a callback on selected zero-based training epochs.

    Args:
        schedule: Ascending iterable/generator of zero-based training epoch
            indices, or a predicate accepting ``index`` or ``index, state``.
        callback: A callable accepting ``state`` or a Torchbearer callback.
    """

    def __init__(self, schedule, callback):
        super().__init__()
        self.schedule = _Schedule(schedule)
        self.callback = callback

    def state_dict(self):
        if hasattr(self.callback, "state_dict"):
            return {"callback": self.callback.state_dict()}
        return {}

    def load_state_dict(self, state_dict):
        if hasattr(self.callback, "load_state_dict"):
            self.callback.load_state_dict(state_dict.get("callback", {}))
        return self

    def on_end_epoch(self, state):
        if self.schedule.should_run(state[torchbearer.EPOCH], state):
            _call_callback(self.callback, "on_end_epoch", state)


def every_n_batches(period, include_first=False):
    """Decorate a callback to run every ``period`` training batches.

    The decorated function receives the Torchbearer ``state`` dict. Batches are
    counted from one for the interval check, so ``period=5`` runs on batch
    indices 4, 9, 14, ...
    """

    def decorator(callback):
        return EveryNBatches(period, callback, include_first=include_first)

    return decorator


def every_n_epochs(period, include_first=False):
    """Decorate a callback to run every ``period`` training epochs.

    The decorated function receives the Torchbearer ``state`` dict. Epochs are
    counted from one for the interval check, so ``period=5`` runs on epoch
    indices 4, 9, 14, ...
    """

    def decorator(callback):
        return EveryNEpochs(period, callback, include_first=include_first)

    return decorator


def at_training_iterations(schedule):
    """Decorate a function as a callback for selected training iterations.

    ``schedule`` can be an ascending iterable/generator of zero-based global
    training iteration indices, or a predicate accepting ``index`` or
    ``index, state``.
    """

    def decorator(callback):
        return AtTrainingIterations(schedule, callback)

    return decorator


def at_training_epochs(schedule):
    """Decorate a callback to run on selected zero-based training epochs.

    ``schedule`` can be an ascending iterable/generator of training epoch
    indices, or a predicate accepting ``index`` or ``index, state``.
    """

    def decorator(callback):
        return AtTrainingEpochs(schedule, callback)

    return decorator
