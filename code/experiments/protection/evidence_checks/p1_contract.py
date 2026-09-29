"""P1 changes only the receiver's dedicated readback trigger and budget.

Issues and identity provenance use the preserved historical contract. This class
does not invoke common recovery or read any evaluator state.
"""
from copy import deepcopy
from legacy_contract import ReceiverContract as HistoricalContract

STRATEGIES=('baseline','check_only','always_readback','guarded_readback')

class ReceiverContract(HistoricalContract):
    def __init__(self,*args,strategy,**kwargs):
        if strategy not in STRATEGIES:
            raise ValueError('unsupported P1 strategy')
        if 'max_readbacks' in kwargs:
            raise ValueError('P1 readback budget is fixed by strategy')
        self.strategy=strategy
        self.receiver_calls=0
        super().__init__(*args,max_readbacks=0 if strategy in ('baseline','check_only') else 1,**kwargs)

    def accept(self,message,readback):
        if self.strategy=='baseline':
            return deepcopy(message)
        first_receiver=self.receiver_calls==0
        self.receiver_calls+=1
        before=self.issues(message)
        candidate=deepcopy(message)
        requested=(self.strategy=='always_readback' and first_receiver) or (
            self.strategy=='guarded_readback' and bool(before))
        called=bool(requested and self.readbacks<self.max_readbacks)
        if called:
            self.readbacks+=1
            try:
                candidate=readback()
            except Exception as exc:
                self.events.append(dict(strategy=self.strategy,receiver_call=self.receiver_calls,
                                        before_issues=list(before),after_issues=None,
                                        readback_called=True,readbacks_used=self.readbacks,
                                        accepted=None,identity_provenance=self.identity_provenance,
                                        outcome='readback_error',error_type=type(exc).__name__))
                raise
        after=self.issues(candidate)
        accepted=not after
        self.events.append(dict(strategy=self.strategy,receiver_call=self.receiver_calls,
                                before_issues=list(before),after_issues=list(after),
                                readback_called=called,readbacks_used=self.readbacks,
                                accepted=accepted,identity_provenance=self.identity_provenance))
        if accepted:
            self.minimum_version=max(self.minimum_version,candidate['version'])
            return deepcopy(candidate)
        return None
