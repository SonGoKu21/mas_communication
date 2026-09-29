"""Receiver-visible obligations only; no access to fault or evaluator state."""
import copy
from mas_faults.shopping_mitigation import check_evidence


class ReceiverContract:
    def __init__(self, task, *, session_id, action_id, entity_id, minimum_version,
                 known_identity=None, identity_provenance=None, max_readbacks=1):
        if type(minimum_version) is not int or minimum_version < 0:
            raise ValueError('invalid version obligation')
        if type(max_readbacks) is not int or not 0 <= max_readbacks <= 2:
            raise ValueError('invalid readback budget')
        if known_identity and identity_provenance not in ('task_input', 'pre_action_observation'):
            raise ValueError('identity needs a receiver-visible provenance')
        self.task = copy.deepcopy(task)
        self.binding = dict(task_id=task['task_id'], session_id=session_id,
                            action_id=action_id, entity_id=entity_id)
        self.minimum_version = minimum_version
        self.identity = copy.deepcopy(known_identity or {})
        if set(self.identity) - {'product_id', 'sku'}:
            raise ValueError('unsupported identity constraint')
        self.identity_provenance = identity_provenance
        self.max_readbacks, self.readbacks = max_readbacks, 0
        self.events = []

    def issues(self, message):
        if not isinstance(message, dict):
            return ('missing_envelope',)
        issues = list(check_evidence(self.task, message.get('payload'), require_success=False).issues)
        for key, value in self.binding.items():
            if message.get(key) != value:
                issues.append('binding:' + key)
        version = message.get('version')
        if type(version) is not int:
            issues.append('type:version')
        elif version < self.minimum_version:
            issues.append('stale:version')
        for key in ('evidence_id', 'source'):
            if not isinstance(message.get(key), str) or not message[key].strip():
                issues.append('missing:' + key)
        payload = message.get('payload')
        if isinstance(payload, dict):
            for key, expected in self.identity.items():
                if payload.get(key) != expected:
                    issues.append('identity:' + key)
        return tuple(issues)

    def accept(self, message, readback):
        before = self.issues(message)
        candidate = copy.deepcopy(message)
        called = False
        if before and self.readbacks < self.max_readbacks:
            self.readbacks += 1
            called = True
            candidate = readback()
        after = self.issues(candidate)
        accepted = not after
        self.events.append(dict(before_issues=list(before), after_issues=list(after),
                                readback_called=called, readbacks_used=self.readbacks,
                                accepted=accepted, identity_provenance=self.identity_provenance))
        if accepted:
            self.minimum_version = max(self.minimum_version, candidate['version'])
            return copy.deepcopy(candidate)
        return None
