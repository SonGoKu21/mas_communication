"""Strict same-unit matched model/topology comparisons; unknown is never false."""
from collections import Counter
from itertools import product
MODELS=('deepseek','qwen9','qwen27')
TOPS=('sequential','hierarchical','flat')
CONFIGS=tuple(product(MODELS,TOPS))
COHORTS=('common_applied','all_known','clean_success_common_applied')
FAULTS=('non_delivery_step2','non_delivery_step3','non_delivery_step4','semantic_corruption_step3','semantic_corruption_step4')
def select_unit(fault,clean,cohort):
 if cohort not in COHORTS:raise ValueError('unknown cohort')
 if any(c not in fault for c in CONFIGS):return 'missing_fault_configuration'
 if any(type(fault[c].get('success')) is not bool for c in CONFIGS):return 'unknown_fault_outcome'
 if cohort=='clean_success_common_applied':
  if any(c not in clean for c in CONFIGS):return 'missing_clean_configuration'
  if any(type(clean[c].get('success')) is not bool for c in CONFIGS):return 'unknown_clean_outcome'
 if cohort!='all_known' and not all(fault[c].get('applied') is True for c in CONFIGS):return 'not_all_nine_applied'
 if cohort=='clean_success_common_applied' and not all(clean[c]['success'] for c in CONFIGS):return 'not_all_nine_clean_success'
 return 'included'
def pct(a,b):return 100*a/b if b else None
def paired(base,candidate):
 if len(base)!=len(candidate) or any(type(x) is not bool for x in [*base,*candidate]):raise ValueError('matched known bool required')
 ct=Counter(str(int(a))+str(int(b)) for a,b in zip(base,candidate));ct={k:ct[k] for k in ('00','01','10','11')}
 n=len(base);fail=ct['00']+ct['01'];success=ct['10']+ct['11']
 return dict(n=n,four_states=ct,baseline_failures=fail,baseline_successes=success,rescued=ct['01'],regressed=ct['10'],baseline_success_pct=pct(success,n),candidate_success_pct=pct(ct['01']+ct['11'],n),rescued_among_baseline_failures_pct=pct(ct['01'],fail),regressed_among_baseline_successes_pct=pct(ct['10'],success),net_pp=pct(ct['01']-ct['10'],n))
def jaccard(a,b):
 union=a|b
 return dict(intersection=len(a&b),union=len(union),left_only=len(a-b),right_only=len(b-a),jaccard=len(a&b)/len(union) if union else None)
def overlap(base,flat,other):
 b={i for i,v in enumerate(base) if not v};f={i for i in b if flat[i]};m={i for i in b if other[i]};u=f|m
 return dict(baseline_failures=len(b),both=len(f&m),flat_only=len(f-m),model_only=len(m-f),neither=len(b-u),rescue_union=len(u),rescue_jaccard=len(f&m)/len(u) if u else None)
