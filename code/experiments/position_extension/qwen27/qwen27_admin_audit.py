from copy import deepcopy
from admin_semantic import CONDITIONS,validate_entry
def audit_row(row,entry,condition):
    validate_entry(entry)
    errors=[];events=row.get('events',[])
    opportunities=[e for e in events if e.get('fault_parameters',{}).get('injection_opportunity_consumed')]
    applied=[e for e in events if e.get('fault_applied')]
    if len(opportunities)>1 or len(applied)>1:errors.append('multiple_exposure_opportunities')
    if applied and not opportunities:errors.append('applied_without_opportunity_provenance')
    exposure='applied' if applied else ('ineffective' if opportunities else 'not_exposed')
    for event in opportunities:
        if event.get('abstract_step')!=CONDITIONS[condition]:errors.append('wrong_injection_step')
        params=event['fault_parameters']
        for k in ('entry_sha256','source_sha256','raw_table_sha256','target_projected_sha256'):
            if params.get(k)!=entry[k]:errors.append('donor_hash_mismatch:'+k)
        if params.get('donor_run_id')!=entry['donor']['run_id']:errors.append('donor_run_mismatch')
        original=deepcopy(event.get('original_message'));delivered=event.get('delivered_messages')
        if not isinstance(original,dict) or not isinstance(delivered,list) or len(delivered)!=1:
            errors.append('invalid_delivery_shape');continue
        expected=deepcopy(original)
        if CONDITIONS[condition]==3:expected['visible_evidence']=entry['raw_table']
        else:expected['payload']['visible_evidence']=entry['target_projected_table']
        if delivered[0]!=expected:errors.append('replacement_or_unchanged_outer_field_mismatch')
        if bool(event.get('fault_applied'))!=(expected!=original):errors.append('wrong_applied_vs_noop_classification')
    for event in events:
        if event.get('abstract_step')==2 and event.get('source_agent')=='Tool Navigator' and event.get('target_agent')=='WebArena Tool Worker':
            if event.get('delivered_messages')!=[event.get('original_message')]:errors.append('task_request_changed')
        if event.get('abstract_step')==1:
            task=(event.get('original_message') or {}).get('task',{})
            if any(k in task for k in ('eval','expected_answer','reference_answers')):errors.append('reference_answer_in_agent_task')
    if row.get('topology')=='flat':
        handoffs=[e for e in events if e.get('abstract_step')==4 and e.get('source_agent')=='Evidence Worker']
        if row.get('error') in (None,'') and len(handoffs)!=2:errors.append('missing_flat_handoff')
        if len(handoffs)==2:
            if CONDITIONS[condition]==3 and handoffs[0].get('delivered_messages')!=handoffs[1].get('delivered_messages'):
                errors.append('i3_flat_branches_diverge')
            if CONDITIONS[condition]==4 and handoffs[1].get('delivered_messages')!=[handoffs[0].get('original_message')]:
                errors.append('i4_flat_direct_branch_changed')
    return {'exposure':exposure,'errors':errors,'opportunities':len(opportunities),'applied_events':len(applied)}
