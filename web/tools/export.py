"""Offline, read-only-source export. No model client or network requests."""
import os
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get('MAS_SOURCE_ROOT', Path(__file__).resolve().parents[2]))
SITE = Path(__file__).resolve().parents[1] / 'site'
SECRET = re.compile(r'api.?key|authorization|password|secret|access.?token|refresh.?token|cookie|csrf|credential|first.?name|last.?name|full.?name|customer.?name|billing.?address|shipping.?address|telephone|phone_number', re.I)
FIELDS = '''task original_message delivered_message fault_parameters fault_events exposure_deliveries detection_events recovery_events recovery_evidence action_protocol_events semantic_contract_events action_ledger_events final_evidence final_verdict final_answer expected_answer verification environment_state environment_task_success final_commit_allowed recorded_cart_audit observed_A_symptom observed_M_consequence system_consequences semantic_consequences propagation_class topology_branch_inputs topology_declared_edges topology_used_edges graph whole_budget budget judgments source_evidence common_recovery_enabled rq4_mechanisms fault_detection_evidence topology_recovery_evidence first_divergence original_evidence_complete delivered_evidence_complete final_decision_correct official_source_task_success official_final_answer_evaluator semantic_evaluator_evidence system_evaluator_evidence architecture_manifestation_evidence action_contract_valid action_outcome_unknown planned_exposures recovery_path_exposed prevention_detected prevention_type action_prevention_count action_recovery_count error'''.split()

def sanitize(value):
    if isinstance(value, dict):
        return {str(k): '[REDACTED]' if SECRET.search(str(k)) else sanitize(v) for k,v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        if value.lstrip().startswith(('{','[')):
            try: return json.dumps(sanitize(json.loads(value)), ensure_ascii=False)
            except (ValueError, TypeError): pass
        value = re.sub(r'https?://[^\s<>"\\]+', '[URL REDACTED]', value)
        value = re.sub(r'\bBearer\s+[^\s,"\'\\]+|\bsk-[A-Za-z0-9_-]{10,}', '[REDACTED]', value, flags=re.I)
        value = re.sub(r'/(?:Users|home|data\d*|opt|tmp|var)/[^\s"\'<>\\]*', '[PRIVATE PATH]', value)
        value = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b', '[HOST]', value)
        value = re.sub(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', value)
        value = re.sub(r'(?i)(password|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s,;]+', r'\1=[REDACTED]', value)
    return value

def catalog_id(study, source, run_id):
    return hashlib.sha256(json.dumps([study,source,run_id]).encode()).hexdigest()[:24]

def labels(value):
    if value is None: return []
    if not isinstance(value,list): value=[value]
    return [str(v) for v in value if str(v).lower() not in ('none','clean','','null','false')]

def metadata(r, study, domain, model, source, line):
    rid = str(r.get('run_id',line))
    fault = r.get('fault_applied')
    if fault is None and 'fault_events' in r:
        fault = bool(r['fault_events'])
    item = dict(id=catalog_id(study,source+'|'+model,rid),study=study,domain=domain,model=model,
                task_id=str(r.get('task_id',r.get('payload_size_bytes','fixture'))),
                topology=r.get('topology',r.get('protocol','not recorded')),
                condition=r.get('condition',r.get('family','unknown')),
                repetition=r.get('repeat_index',r.get('repeat')),endpoint=r.get('final_task_success'),
                fault_applied=fault,consequences=labels(r.get('observed_M_consequence')),
                response=r.get('recovery_detected'),variant=r.get('variant',r.get('arm','not applicable')),
                position=r.get('injection_step',r.get('boundary')),source_run_id=rid,
                evidence_level='summary only',trace_path=None,
                source_file=source,source_line=line,
                propagation_class=r.get('propagation_class'),
                observation=r.get('primary'),planned_exposures=r.get('planned_exposures'),
                fault_event_count=len(r['fault_events']) if 'fault_events' in r else None,
                tokens=r.get('total_tokens'),latency_ms=r.get('latency_ms'),
                source_evidence_run_id=r.get('source_run_id'),
                severity=r.get('level',r.get('fault_severity')),
                deadline_ms=r.get('deadline_ms'),payload_size_bytes=r.get('payload_size_bytes'))
    if item['position'] is None and 'step' in item['condition']:
        item['position'] = item['condition'].rsplit('step',1)[-1]
    return sanitize(item)

def detail(r, meta):
    events=r.get('events',[])
    if not isinstance(events,list): events=[]
    data={k:r[k] for k in FIELDS if k in r}
    for key in ['decision_correct','source_run_id','source_trace_id']:
        if key in r: data[key]=r[key]
    if meta.get('study')=='bridge': data['fixture_observations']={k:v for k,v in r.items() if k!='result_path'}
    return sanitize(dict(metadata=meta,events=events,evidence=data,
                        event_order='Recorded list order; separate fault/handling logs are not time-aligned unless the source links them.'))

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':')))

def finding_data():
    return json.loads((SITE.parent / 'paper-findings.json').read_text())

def main():
    source_root=ROOT/'reports/rq123_six_condition_inputs_20260916_v2'
    config=json.loads((source_root/'analysis_config.json').read_text())
    sources=[]
    for e in config['experiments']:
        for path in e['paths']:
            sources.append((source_root/Path(path).name,'rq123',e['dataset'],e['label'],e['expected_rows']))
    sources += [(source_root/'reddit_extension.jsonl','extension','WebArena Reddit','from-record',2835),
                (ROOT/'reports/rq4_flash1098_20260914_v1/main_runs.jsonl','rq4','WebArena Shopping','from-record',1098),
                (ROOT/'reports/source_bridge_final_analysis_20260915_v3/runs.jsonl','bridge','Communication fixtures','No LLM',3780)]
    all_items=[];provenance=[];groups=Counter();seen=set()
    for path,study,domain,model,expected in sources:
        print('Exporting',path.name,study,flush=True)
        digest=hashlib.sha256();count=0
        with path.open('rb') as f:
            for count,line in enumerate(f,1):
                digest.update(line);r=json.loads(line)
                s=study
                if study=='rq4':
                    exp=r['experiment'];s='rq4-'+('path' if exp not in ('main','stress') else exp)
                m=r.get('model_version',r.get('model','unknown')) if model=='from-record' else model
                if m=='deepseek-v4-flash': m='DeepSeek-V4-Flash'
                item=metadata(r,s,domain,m,path.name,count)
                assert item['id'] not in seen, ('duplicate',item['id'])
                seen.add(item['id'])
                key=(s,domain,m,item['topology'],item['condition'],item['endpoint'])
                groups[key]+=1
                featured=(domain=='WebArena Reddit' and item['task_id'] in ('27','67')) or item['task_id']=='pydata__xarray-2905' or (domain=='WebArena Admin' and item['task_id'] in ('41','208','292'))
                selected=study in ('rq4','bridge') or featured or groups[key]<=1
                if selected:
                    item['trace_path']='data/runs/'+item['id']+'.json'
                    item['evidence_level']='recorded events' if r.get('events') else 'recorded fields'
                    write(SITE/item['trace_path'],detail(r,item))
                all_items.append(item)
        assert count==expected,(path.name,count,expected)
        provenance.append(dict(file=path.name,study=study,model=model,rows=count,sha256=digest.hexdigest()))
    partitions=[]
    for i in range(0,len(all_items),1000):
        name=f'data/catalog-{i//1000:02}.json';write(SITE/name,all_items[i:i+1000]);partitions.append(name)
    studies=[]
    for s in dict.fromkeys(x['study'] for x in all_items):
        rows=[x for x in all_items if x['study']==s]
        studies.append(dict(id=s,rows=len(rows),tasks=len(set((x['domain'],x['task_id']) for x in rows)),
                            traces=sum(bool(x['trace_path']) for x in rows)))
    findings=finding_data()
    for f in findings:
        candidates=[x for x in all_items if all(x.get(k)==v for k,v in f['filters'].items())]
        if f.get('special')=='propagation': candidates=[x for x in candidates if x['fault_applied'] and x['endpoint'] is True and x['consequences']]
        candidates.sort(key=lambda x:(not bool(x['trace_path']),x['condition']=='clean',x['repetition'] or 0))
        f['examples']=[x['id'] for x in candidates if x['trace_path']][:12]
        f['count']=len(candidates)
        for case in f.get('cases',[]):
            matched=[x for x in all_items if all(x.get(k)==v for k,v in case['query'].items())]
            assert len(matched)==1, (case['label'],len(matched),case['query'])
            assert matched[0]['trace_path'],case['label']
            case['run_id']=matched[0]['id']
        preferred=[c['run_id'] for c in f.get('cases',[]) if c['run_id'] in {x['id'] for x in candidates}]
        f['examples']=list(dict.fromkeys(preferred+f['examples']))[:12]
    manifest=dict(version='2026-09-24.paper7',catalogs=partitions,total=len(all_items),studies=studies,
                  detailed=sum(bool(x['trace_path']) for x in all_items),sources=provenance,
                  limitations=['Historical replay, not live experimental execution.',
                  'Summary-only records do not include a bundled detailed trace.',
                  'Composite endpoints are not necessarily native benchmark answer correctness.',
                  'Repeated executions are not independent model sampling seeds.',
                  'Source-to-symptom fixture observations and MAS outcomes are separate studies.',
                  'Personal/deployment identifiers and URLs have been removed from public traces.'])
    write(SITE/'data/manifest.json',manifest);write(SITE/'data/findings.json',findings)
    write(SITE/'data/coverage.json',dict(studies=studies,total=len(all_items),detailed=manifest['detailed'],source_hashes=provenance))
    # Remove only obsolete generated trace exports, never original inputs.
    referenced={Path(x['trace_path']).name for x in all_items if x['trace_path']}
    for p in (SITE/'data/runs').glob('*.json'):
        if p.name not in referenced: p.unlink()
    print(json.dumps(studies,indent=2))

if __name__=='__main__': main()
