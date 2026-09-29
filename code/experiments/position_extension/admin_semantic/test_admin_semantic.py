"""Offline behavioral tests; no model, browser, auth or network imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import os
import unittest
from copy import deepcopy

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('MAS_HISTORICAL_ADMIN_RECORDS', 'external/experiment_12.jsonl'))
try:
    import admin_semantic as sut
except ModuleNotFoundError:
    sut = None


def row(task_id, repeat=1, topology='flat', table=None, stratum='order_payment_aggregation'):
    table = table or [['ID', 'Purchase Date', 'Grand Total (Base)', 'Status', 'Extra'], [str(task_id), 'date', '$7', 'Complete', 'extra']]
    projected = [r[:4] for r in table]
    return {'condition':'clean', 'model':'deepseek-v4-flash', 'task_id':str(task_id), 'task_stratum':stratum, 'repeat_index':repeat, 'topology':topology,
            'run_id':f'run-{task_id}-{topology}-{repeat}', 'trace_id':f'trace-{task_id}-{repeat}',
            'events':[{'abstract_step':3, 'step_index':5, 'source_agent':'WebArena Tool Worker', 'original_message':{'visible_evidence':table}, 'delivered_messages':[{'visible_evidence':table}]},
                      {'abstract_step':4, 'source_agent':'Evidence Worker', 'original_message':{'task_id':str(task_id),'source_session':f'run-{task_id}-{topology}-{repeat}', 'payload':{'visible_evidence':projected}}}]}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(sut, 'independent Admin semantic implementation is missing')

    def manifest(self, rows=None, order=(4,107,187), repeats=(1,)):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.jsonl'
            p.write_text(''.join(json.dumps(r)+'\n' for r in (rows or [row(i) for i in order])))
            return sut.build_manifest(p, task_order=order, topologies=('flat',), repeats=repeats)

    def entry(self):
        return self.manifest()['entries'][0]

    def test_order_is_preserved_for_donor_rotation(self):
        m=self.manifest(order=(187,4,107))
        self.assertEqual([e['target']['task_id'] for e in m['entries']], ['187','4','107'])
        self.assertEqual(m['entries'][0]['donor']['task_id'], '4')
        self.assertEqual(m['entries'][0]['donor']['run_id'], 'run-4-flat-1')

    def test_repeat_fallback_records_actual_repeat(self):
        rs=[row(i) for i in (4,107,187)]
        m=self.manifest(rs,repeats=(2,))
        self.assertEqual(m['entries'][0]['donor']['task_id'],'187')
        self.assertEqual(m['entries'][0]['donor']['repeat_index'],1)
        self.assertEqual(m['entries'][0]['target']['repeat_index'],2)

    def test_missing_stratum_is_not_guessed(self):
        rs=[row(i) for i in (4,107,187)];rs[0].pop('task_stratum')
        with self.assertRaisesRegex(ValueError,'stratum'):
            self.manifest(rs)

    def test_wrong_donor_projection_fails_closed(self):
        rs=[row(i) for i in (4,107,187)]
        rs[1]['events'][-1]['original_message']['payload']['visible_evidence']=[['wrong']]
        with self.assertRaisesRegex(ValueError,'projection'):
            self.manifest(rs)

    def test_duplicate_carrier_identity_rejected(self):
        rs=[row(i) for i in (4,107,187)];rs.append(deepcopy(rs[0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.manifest(rs)

    def test_step3_changes_only_visible_table_and_only_once(self):
        entry=self.entry(); op=sut.TableSubstitution('visible_table_substitution_step3',entry)
        current={'url':'current','title':'current','state_version':3,'tool_status':'ok','available_tools':['finish_with_evidence'],'visible_evidence':[['fresh'],['truth']]}
        original=deepcopy(current)
        self.assertFalse(op.intercept(2,current,eligible=True).fault_applied)
        self.assertFalse(op.intercept(3,current,eligible=False).fault_applied)
        got=op.intercept(3,current,eligible=True)
        want=deepcopy(current);want['visible_evidence']=entry['raw_table']
        self.assertEqual(got.delivered_message,want)
        self.assertTrue(got.fault_applied)
        self.assertEqual(current,original)
        self.assertEqual(op.intercept(3,current,eligible=True).delivered_message,current)
        self.assertFalse(op.intercept(3,current,eligible=True).fault_applied)

    def test_noop_is_ineffective_and_consumes_one_opportunity(self):
        entry=self.entry();op=sut.TableSubstitution('visible_table_substitution_step3',entry)
        result=op.intercept(3,{'visible_evidence':entry['raw_table']},eligible=True)
        self.assertFalse(result.fault_applied)
        self.assertEqual(result.effect,'ineffective_visible_table_substitution')
        self.assertFalse(op.intercept(3,{'visible_evidence':[['different']]},eligible=True).fault_applied)

    def test_step4_changes_only_visible_table_preserving_summary_and_structure(self):
        entry=self.entry();op=sut.TableSubstitution('visible_table_substitution_step4',entry)
        current={'task_id':'4','message_id':'m','payload':{'evidence_result':{'candidate_answer':'current'},'visible_evidence':[['current']],'structured_task_evidence':{'current':True}}}
        before=deepcopy(current);got=op.intercept(4,current,eligible=True)
        want=deepcopy(current);want['payload']['visible_evidence']=entry['target_projected_table']
        self.assertEqual(got.delivered_message,want)
        self.assertEqual(current,before)
        self.assertFalse(op.intercept(4,current,eligible=False).fault_applied)

    def test_same_donor_raw_and_target_projection_across_positions(self):
        entry=self.entry()
        a=sut.TableSubstitution('visible_table_substitution_step3',entry).intercept(3,{'visible_evidence':[['fresh']]},eligible=True)
        b=sut.TableSubstitution('visible_table_substitution_step4',entry).intercept(4,{'payload':{'visible_evidence':[['fresh']]}},eligible=True)
        self.assertEqual(sut.project_visible_evidence({'task_stratum':entry['target']['task_stratum']},a.delivered_message['visible_evidence']), b.delivered_message['payload']['visible_evidence'])
        self.assertEqual(a.parameters['raw_table_sha256'],b.parameters['raw_table_sha256'])

    def test_tampered_manifest_fails_before_injection(self):
        entry=self.entry();entry['raw_table'][1][0]='tampered'
        with self.assertRaisesRegex(ValueError,'hash'):
            sut.TableSubstitution('visible_table_substitution_step3',entry)

    def test_historical_condition_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError,'condition'):
            sut.TableSubstitution('semantic_corruption_step4',self.entry())

    def test_last_observation_mismatch_does_not_select_earlier_matching_table(self):
        r=row(107)
        r['events'].insert(1, {'abstract_step':3,'step_index':6,'original_message':{'visible_evidence':[['Wrong']]},'delivered_messages':[{'visible_evidence':[['Wrong']]}]})
        with self.assertRaisesRegex(ValueError,'projection'):
            sut.reconstruct_donor_table(r)

    @unittest.skipUnless(SOURCE.is_file(), 'requires optional original archive via MAS_HISTORICAL_ADMIN_RECORDS')
    def test_manifest_donors_match_original_historical_selector(self):
        import ast
        import types
        source_code=(ROOT.parent/'run_webarena_admin_main_confirmation.py').read_text()
        tree=ast.parse(source_code)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_select_stale_carrier')
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        module=ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[]))
        namespace={'json':json,'MAIN_TASK_IDS':sut.TASK_ORDER}
        exec(compile(module,'historical_selector','exec'),namespace)
        rows=[json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
        carriers=[]
        for r in rows:
            if r.get('condition')!='clean':continue
            for event in r.get('events',[]):
                if event.get('abstract_step')==4 and event.get('source_agent')=='Evidence Worker' and isinstance(event.get('original_message'),dict):
                    carriers.append({'task_id':r['task_id'],'topology':r['topology'],'repeat_index':r['repeat_index'],'message':event['original_message']})
                    break
        m=sut.build_manifest(SOURCE)
        for entry in m['entries']:
            target=entry['target']
            job=types.SimpleNamespace(task={'task_id':target['task_id']},topology=target['topology'],repeat_index=target['repeat_index'])
            donor=namespace['_select_stale_carrier'](carriers,job,task_ids=sut.TASK_ORDER)
            self.assertEqual(donor['source_session'],entry['donor']['run_id'])

    @unittest.skipUnless(SOURCE.is_file(), 'requires optional original archive via MAS_HISTORICAL_ADMIN_RECORDS')
    def test_real_admin195_has_raw12_and_projected4(self):
        rows=[json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
        r=next(r for r in rows if r['condition']=='clean' and r['task_id']=='195' and r['topology']=='sequential' and r['repeat_index']==2)
        raw,projected,step=sut.reconstruct_donor_table(r)
        self.assertEqual(len(raw[0]),12)
        self.assertEqual(projected[0],['ID','Purchase Date','Grand Total (Base)','Status'])
        self.assertGreater(len(raw),1)
        self.assertIsNotNone(step)

    @unittest.skipUnless(SOURCE.is_file(), 'requires optional original archive via MAS_HISTORICAL_ADMIN_RECORDS')
    def test_full_historical_manifest_is_serializable_and_complete(self):
        m=sut.build_manifest(SOURCE)
        self.assertEqual(len(m['entries']),180)
        self.assertEqual(m['task_order'],[4,107,187,199,288,0,62,185,193,194,195,198,200,208,209,210,211,292,41,42])
        self.assertEqual(json.loads(json.dumps(m)),m)
        self.assertTrue(all(e['donor']['task_id']!=e['target']['task_id'] for e in m['entries']))
        self.assertTrue(all(len(e['source_sha256'])==64 and len(e['raw_table_sha256'])==64 and len(e['target_projected_sha256'])==64 for e in m['entries']))

if __name__=='__main__':unittest.main()
