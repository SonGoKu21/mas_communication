"""Independent, offline Admin table-substitution preparation and pure operator.

No model client, browser, authentication, or shared registry imports.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from projection import project_visible_evidence

TASK_ORDER=(4,107,187,199,288,0,62,185,193,194,195,198,200,208,209,210,211,292,41,42)
TOPOLOGIES=('sequential','flat','hierarchical')
CONDITIONS={'visible_table_substitution_step3':3,'visible_table_substitution_step4':4}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _table(value):
    if not isinstance(value,list) or not value or any(not isinstance(row,list) or any(not isinstance(cell,str) for cell in row) for row in value):
        raise ValueError('visible table must be a nonempty list of string rows')
    if not value[0]:
        raise ValueError('visible table must have a header')
    return deepcopy(value)


def _stratum(row):
    value=row.get('task_stratum')
    if not isinstance(value,str) or not value.strip():
        raise ValueError('missing task_stratum; do not infer it')
    return value


def _carrier(row):
    return next((e['original_message'] for e in row.get('events',[]) if e.get('abstract_step')==4 and e.get('source_agent')=='Evidence Worker' and isinstance(e.get('original_message'),dict)),None)


def reconstruct_donor_table(row):
    """Recover the last nonempty delivered I3 table consumed before first I4.

    This reproduces the clean run's accepted-table overwrite semantics. It does
    not search backwards for any convenient matching earlier table.
    """
    if row.get('condition')!='clean':
        raise ValueError('donor must be a recorded clean run')
    _stratum(row)
    raw=None; step=None; carrier=None
    for index,event in enumerate(row.get('events',[])):
        if event.get('fault_applied'):
            raise ValueError('clean donor contains an applied fault')
        if event.get('abstract_step')==4 and event.get('source_agent')=='Evidence Worker' and isinstance(event.get('original_message'),dict):
            carrier=event['original_message'];break
        if event.get('abstract_step')!=3:
            continue
        delivered=event.get('delivered_messages',[])
        if delivered and isinstance(delivered[-1],dict) and delivered[-1].get('visible_evidence'):
            raw=_table(delivered[-1]['visible_evidence'])
            original=event.get('original_message')
            if not isinstance(original,dict) or original.get('visible_evidence')!=raw:
                raise ValueError('clean donor original/delivered table mismatch')
            step={'event_list_index':index,'step_index':event.get('step_index')}
    if raw is None or carrier is None:
        raise ValueError('donor lacks an observed I3 table or I4 carrier')
    projected=_table(carrier.get('payload',{}).get('visible_evidence'))
    if project_visible_evidence(row,raw)!=projected:
        raise ValueError('donor projection does not equal recorded I4 visible table')
    if str(carrier.get('task_id'))!=str(row['task_id']) or carrier.get('source_session')!=row.get('run_id'):
        raise ValueError('donor carrier identity mismatch')
    return raw,projected,step


def build_manifest(source, *, task_order=TASK_ORDER, topologies=TOPOLOGIES, repeats=(1,2,3)):
    """Freeze the original rotation and nearest-repeat donor selection rules."""
    source=Path(source);content=source.read_bytes()
    source_sha=hashlib.sha256(content).hexdigest()
    order=tuple(int(t) for t in task_order)
    if not order or len(set(order))!=len(order):
        raise ValueError('empty or duplicate task order')
    rows=[json.loads(line) for line in content.decode().splitlines() if line.strip()]
    clean=[r for r in rows if r.get('condition')=='clean' and int(r['task_id']) in order]
    metadata={};pool=[];identities=set()
    for r in clean:
        task_id=int(r['task_id']);stratum=_stratum(r)
        if task_id in metadata and metadata[task_id]!=stratum:
            raise ValueError('conflicting task_stratum')
        metadata[task_id]=stratum
        identity=(str(r['topology']),task_id,int(r['repeat_index']))
        if identity in identities:
            raise ValueError('duplicate clean carrier identity')
        identities.add(identity)
        if _carrier(r) is not None:
            pool.append(r)
    if set(metadata)!=set(order):
        raise ValueError('missing task stratum metadata')
    reconstructed={}
    entries=[]
    for task_id in order:
        target_index=order.index(task_id)
        for topology in topologies:
            for repeat in repeats:
                if int(repeat)<1:
                    raise ValueError('repeat index must be positive')
                candidates=[]
                for offset in range(int(repeat),int(repeat)+len(order)):
                    donor_id=order[(target_index+offset)%len(order)]
                    if donor_id==task_id:continue
                    candidates=[r for r in pool if str(r['topology'])==topology and int(r['task_id'])==donor_id]
                    if candidates:break
                if not candidates:
                    raise ValueError(f'no real cross-task stale carrier for {topology}/{task_id}')
                candidates.sort(key=lambda r:(int(r['repeat_index'])!=int(repeat),abs(int(r['repeat_index'])-int(repeat))))
                donor=candidates[0]
                donor_key=(donor['run_id'],topology,int(donor['repeat_index']))
                if donor_key not in reconstructed:
                    reconstructed[donor_key]=reconstruct_donor_table(donor)
                raw,projected,step=reconstructed[donor_key]
                target={'task_id':str(task_id),'task_stratum':metadata[task_id],'topology':topology,'repeat_index':int(repeat)}
                target_table=project_visible_evidence(target,raw)
                entry={'target':target,'donor':{k:donor.get(k) for k in ('task_id','task_stratum','topology','repeat_index','run_id','trace_id','model')},
                       'source_file':str(source.resolve()),'source_sha256':source_sha,'donor_i3_event':step,
                       'raw_table':deepcopy(raw),'donor_projected_table':deepcopy(projected),'target_projected_table':deepcopy(target_table),
                       'raw_table_sha256':digest(raw),'donor_projected_sha256':digest(projected),'target_projected_sha256':digest(target_table)}
                entry['entry_sha256']=digest(entry)
                entries.append(entry)
    return {'schema':'admin_visible_table_substitution/v1','task_order':list(order),'topologies':list(topologies),'repeats':list(repeats),
            'source_file':str(source.resolve()),'source_sha256':source_sha,
            'projection_sha256':hashlib.sha256(Path(__file__).with_name('projection.py').read_bytes()).hexdigest(),
            'conditions':list(CONDITIONS),'donor_selection':'historical task-order rotation; same topology; preferred same repeat, nearest-repeat fallback',
            'entries':entries}


def validate_entry(entry):
    candidate=deepcopy(entry);expected=candidate.pop('entry_sha256',None)
    if digest(candidate)!=expected:
        raise ValueError('manifest entry hash mismatch')
    for field,key in (('raw_table','raw_table_sha256'),('target_projected_table','target_projected_sha256'),('donor_projected_table','donor_projected_sha256')):
        _table(entry[field])
        if digest(entry[field])!=entry[key]:raise ValueError('table hash mismatch')
    if entry['target']['task_id']==str(entry['donor']['task_id']):raise ValueError('donor is not another task')
    if entry['target']['topology']!=entry['donor']['topology']:raise ValueError('donor topology mismatch')
    if project_visible_evidence(entry['target'],entry['raw_table'])!=entry['target_projected_table']:
        raise ValueError('target projection mismatch')


@dataclass(frozen=True)
class TableDelivery:
    delivered_message: Any
    fault_applied: bool
    effect: str
    parameters: dict[str,Any]


class TableSubstitution:
    """One selected opportunity, even when replacement is a no-op."""
    def __init__(self,condition,entry):
        if condition not in CONDITIONS:raise ValueError('unsupported table substitution condition')
        validate_entry(entry)
        self.condition=condition;self.step=CONDITIONS[condition];self.entry=deepcopy(entry)
        self.attempted=False;self.applied=False

    def intercept(self,step,message,*,eligible=True):
        current=deepcopy(message)
        if step!=self.step or not eligible or self.attempted:
            return TableDelivery(current,False,'clean_delivery',{})
        if not isinstance(current,dict):raise ValueError('observation must be an object')
        container=current if step==3 else current.get('payload')
        if not isinstance(container,dict) or 'visible_evidence' not in container:
            raise ValueError('observation lacks visible_evidence')
        current_table=container['visible_evidence']
        if current_table != []:
            _table(current_table)
        self.attempted=True
        replacement=self.entry['raw_table' if step==3 else 'target_projected_table']
        changed=container['visible_evidence']!=replacement
        self.applied=changed
        if changed:container['visible_evidence']=deepcopy(replacement)
        parameters={k:self.entry[k] for k in ('entry_sha256','source_sha256','raw_table_sha256','target_projected_sha256')}
        parameters.update({'donor_run_id':self.entry['donor']['run_id'],'donor_task_id':str(self.entry['donor']['task_id']),
                           'donor_repeat_index':self.entry['donor']['repeat_index'],'target_repeat_index':self.entry['target']['repeat_index'],
                           'operator':'visible_table_substitution','changed_field':'visible_evidence' if step==3 else 'payload.visible_evidence',
                           'injection_opportunity_consumed':True,'effective_content_change':changed})
        return TableDelivery(current,changed,'cross_task_visible_table_substituted' if changed else 'ineffective_visible_table_substitution',parameters)


def main():
    p=argparse.ArgumentParser(description='Offline only: build frozen Admin donor manifest')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();manifest=build_manifest(args.source)
    args.output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'entries':len(manifest['entries']),'source_sha256':manifest['source_sha256'],'output':str(args.output)}))

if __name__=='__main__':main()
