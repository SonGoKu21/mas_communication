"""Opt-in adapter: no shared module mutation, registry mutation or API startup."""
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from types import FunctionType
from admin_semantic import CONDITIONS, TableSubstitution, validate_entry


def make_conditions(condition_type):
    return {name:condition_type(name,'semantic_corruption',step,'A6','semantic_corruption',
              fault_cause='cross_task_visible_table_substitution',
              parameters={'replace_only':'visible_evidence' if step==3 else 'payload.visible_evidence',
                          'one_opportunity':True,'historical_inner_evidence_operator':False})
            for name,step in CONDITIONS.items()}


def make_interceptor_class(base_class,entry):
    validate_entry(entry)
    frozen=deepcopy(entry)
    class IndependentTableInterceptor(base_class):
        def __init__(self,condition_cell,**kwargs):
            super().__init__(condition_cell,**kwargs)
            self.table_operator=TableSubstitution(condition_cell.condition,frozen)

        def intercept(self,step,message,*,context=None):
            context=context or {}
            # Historical base supplies timestamps and delivery record type only.
            clean=super().intercept(step,message,context={**context,'eligible':False})
            result=self.table_operator.intercept(step,message,eligible=bool(context.get('eligible',True)))
            self.applied=self.table_operator.applied
            return replace(clean,delivered_messages=(result.delivered_message,),fault_applied=result.fault_applied,
                           fault_id='A6' if result.fault_applied else 'none',
                           fault_type='semantic_corruption' if result.fault_applied else 'clean',
                           fault_family='semantic_corruption' if result.fault_applied else 'clean',
                           fault_cause='cross_task_visible_table_substitution' if result.fault_applied else 'none',
                           fault_parameters=deepcopy(result.parameters),observed_runtime_effect=result.effect,
                           observed_a_symptom='A6_message_semantic_corruption' if result.fault_applied else 'none')
    return IndependentTableInterceptor


def bind_confirmation_runner(confirmation_module,entry):
    """Clone one runner's globals rather than monkey-patching a live module.

    The caller selects the manifest entry using the logical repeat_index, not
    historical matrix_run_index/run_index (which may use an offset).
    """
    validate_entry(entry)
    frozen=deepcopy(entry)
    runtime_projection=getattr(confirmation_module,'project_visible_evidence',None)
    if not callable(runtime_projection) or runtime_projection(frozen['target'],deepcopy(frozen['raw_table']))!=frozen['target_projected_table']:
        raise ValueError('runtime target projection differs from frozen manifest')
    original=confirmation_module.run_admin_confirmation_task
    namespace=dict(original.__globals__)
    namespace['MainCommunicationInterceptor']=make_interceptor_class(confirmation_module.MainCommunicationInterceptor,frozen)
    bound=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    bound.__kwdefaults__=original.__kwdefaults__

    @wraps(original)
    async def run_bound(client,browser,evaluator,task,**kwargs):
        target=frozen['target']
        if str(task.get('task_id'))!=target['task_id'] or task.get('task_stratum')!=target['task_stratum'] or kwargs.get('topology')!=target['topology']:
            raise ValueError('bound donor target identity mismatch')
        cell=kwargs.get('condition_cell')
        if cell is None or cell.condition not in CONDITIONS or cell.injection_step!=CONDITIONS[cell.condition]:
            raise ValueError('bound runner requires one independent table substitution condition')
        row=await bound(client,browser,evaluator,task,**kwargs)
        row['table_substitution_design']={k:deepcopy(frozen[k]) for k in ('entry_sha256','source_sha256','raw_table_sha256','target_projected_sha256','donor','target')}
        return row
    return run_bound
