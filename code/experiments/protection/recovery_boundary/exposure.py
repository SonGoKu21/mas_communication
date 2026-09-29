"""P3 delivery operator; injector audit metadata never reaches an agent."""
import copy
from mas_faults.multimechanism_faults import SingleBoundaryFault
from mas_faults.multimechanism_matrix import config_digest


class ScopedIdentityExposure:
    PATHS = {'initial_evidence','recovery_evidence','flat_observation','evaluation_readback'}
    PASSTHROUGH_BOUNDARIES = {'action_request', 'action_ack', 'observation_handoff', 'judgment_handoff'}
    SCHEMES = {
        'clean': set(),
        'single_handoff': {'initial_evidence'},
        'persistent_handoff': {'initial_evidence','recovery_evidence'},
        'shared_workflow_evidence': {'initial_evidence','recovery_evidence','flat_observation'},
    }

    def __init__(self, scheme, *, cross_task_evidence):
        if scheme not in self.SCHEMES:
            raise ValueError('unknown predeclared exposure scheme')
        self.scheme=scheme
        self.source=copy.deepcopy(cross_task_evidence)
        self.events=[]
        self.deliveries=[]

    def _record(self, path, message, result, *, eligible, damaged):
        source = self.source if isinstance(self.source, dict) else {}
        self.deliveries.append(dict(
            path=path, ordinal=len(self.deliveries)+1, eligible=eligible, damaged=damaged,
            before_sha256=config_digest(message), after_sha256=config_digest(result[-1]),
            source_sha256=config_digest(self.source) if damaged else None,
            source_task_id=source.get('task_id') if damaged else None,
            source_evidence_id=source.get('evidence_id') if damaged else None))

    def passthrough(self, boundary, message, *, recovery=False):
        if boundary not in self.PASSTHROUGH_BOUNDARIES:
            raise ValueError('unknown unexposed boundary')
        result = [copy.deepcopy(message)]
        self._record(boundary, message, result, eligible=False, damaged=False)
        return result

    def deliver(self, path, message):
        if path not in self.PATHS:
            raise ValueError('unknown evidence path')
        eligible=path in self.SCHEMES[self.scheme]
        damaged=eligible and (self.scheme!='single_handoff' or not self.events)
        ordinal=len(self.deliveries)+1
        if damaged:
            operator=SingleBoundaryFault('contract_consistent_identity_corruption',
                                          cross_task_evidence=self.source)
            result=operator.deliver('evidence_handoff',message)
            self.events.extend(dict(e,path=path,ordinal=ordinal) for e in operator.events)
        else:
            result=[copy.deepcopy(message)]
        self._record(path, message, result, eligible=eligible, damaged=damaged)
        return result
