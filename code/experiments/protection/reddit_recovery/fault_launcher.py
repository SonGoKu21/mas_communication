"""The 108-run fault stage, sharing the clean-stage runtime and accounting."""
import launcher
from launcher import IDS
def select_jobs(design):
 jobs=[j for j in design['jobs'] if j['domain']=='reddit' and j['task_id'] in IDS and j['topology']=='sequential' and j['condition'] in {'valid_partial','semantic_corruption'}]
 if len(design['jobs'])!=720 or len(jobs)!=108 or len({j['job_key'] for j in jobs})!=108:raise ValueError('fixed720_and108_required')
 from collections import Counter
 if Counter(j['task_id'] for j in jobs)!=Counter({i:12 for i in IDS}):raise ValueError('all_nine_candidates_required')
 return jobs

if __name__ == "__main__":
 launcher.select_jobs=select_jobs
 launcher.STAGE="reddit_sequential_fault_108"
 launcher.main()
