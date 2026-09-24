"""Finite path-scoped fault operator, kept outside receiver-visible state."""
import copy
from mas_faults.multimechanism_faults import SingleBoundaryFault
from mas_faults.multimechanism_matrix import CELLS


class FiniteExposure:
    def __init__(self, condition, *, count, expose_recovery=True, **sources):
        if condition not in CELLS or type(count) is not int or count not in (0, 1, 2):
            raise ValueError('invalid fault condition or finite budget')
        if (condition == 'clean') != (count == 0):
            raise ValueError('only clean has zero exposure budget')
        self.condition, self.count = condition, count
        self.sources = copy.deepcopy(sources)
        self.expose_recovery = expose_recovery
        self.events, self.deliveries = [], []

    def deliver(self, boundary, message, *, recovery=False):
        if boundary != CELLS[self.condition] or self.condition == 'clean':
            return [copy.deepcopy(message)]
        eligible = not recovery or self.expose_recovery
        damaged = eligible and len(self.events) < self.count
        ordinal = len(self.deliveries) + 1
        if damaged:
            operator = SingleBoundaryFault(self.condition, **self.sources)
            result = operator.deliver(boundary, message)
            self.events.extend({**event, 'ordinal': ordinal, 'recovery': recovery}
                               for event in operator.events)
        else:
            result = [copy.deepcopy(message)]
        self.deliveries.append(dict(boundary=boundary, ordinal=ordinal, recovery=recovery,
                                    eligible=eligible, damaged=damaged))
        return result
