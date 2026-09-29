from analyze_protection import state,pair_summary,LABELS

def row(success,errors=0,http=1,model=1,cluster='product1'):
 return {'final_task_success':success,'evidence_acceptance_errors':errors,'http_receipt_count':http,'model_calls':model,'product_cluster':cluster}
def test_four_labels_and_unknowns_do_not_infer_detection():
 assert [state(row(True,0)),state(row(True,2)),state(row(False,1)),state(row(False,0))]==LABELS
 assert state(row(None,0))=='unknown_outcome'
 assert state(row(True,None))=='unknown_acceptance_record'
 assert state(row(True,-1))=='unknown_acceptance_record'
def test_total_increment_uses_all_pairs_not_only_rescues():
 pairs=[(row(False,http=2),row(True,http=3)),(row(True,http=2),row(True,http=7))]
 out=pair_summary(pairs,bootstrap=100)
 assert out['rescued']==1 and out['total_http_difference']==6
 assert out['http_per_rescued']['estimate']==6
 assert {(x['baseline'],x['arm'],x['n']) for x in out['joint_state_transitions']}=={(LABELS[0],LABELS[0],1),(LABELS[3],LABELS[0],1)}
def test_zero_and_negative_denominators_remain_null():
 zero=pair_summary([(row(True),row(True,http=4))],bootstrap=100)
 assert zero['http_per_net_additional_success']['estimate'] is None
 negative=pair_summary([(row(True),row(False,http=4))],bootstrap=100)
 assert negative['net_additional_successes']==-1 and negative['http_per_net_additional_success']['estimate'] is None
 assert negative['http_per_rescued']['estimate'] is None
def test_cluster_resampling_keeps_repeated_product_together_and_withholds_ratio_ci():
 pairs=[(row(False,cluster='a'),row(True,http=2,cluster='a'))]*2+[(row(True,cluster='b'),row(False,http=2,cluster='b'))]
 out=pair_summary(pairs,bootstrap=1000)
 assert out['product_clusters']==2 and out['net_additional_successes']==1
 assert 0<out['http_per_net_additional_success']['bootstrap_positive_denominator_fraction']<1
 assert out['http_per_net_additional_success']['cluster_ci95'] is None
 assert out['success_rate_difference_cluster_ci95']==[-1,1]
def test_unknown_outcomes_counted_and_excluded_from_paired_cost_denominator():
 out=pair_summary([(row(None,http=99),row(True)),(row(False),row(True,http=3))],bootstrap=100)
 assert out['n_paired']==2 and out['n_unknown_outcome_pairs']==1 and out['n_known_pairs']==1
 assert out['total_http_difference']==2 and sum(x['n'] for x in out['joint_state_transitions'])==2
