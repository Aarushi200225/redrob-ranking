from src.stages.stage1_jd_intelligence import run
from pathlib import Path

print("Running Stage 1...")
obj = run(Path('data/job_description.txt'))

print()
print("=== Stage 1 Output ===")
print(f"Keys: {list(obj.keys())}")
print(f"Query vectors: {list(obj.get('query_vectors', {}).keys())}")
print(f"Vibe clusters: {list(obj.get('vibe_cluster_vecs', {}).keys())}")
print(f"Hard reqs: {obj.get('hard_requirements', [])}")
print(f"Role intent: {obj.get('role_intent', 'MISSING')}")
print(f"Hyde profile length: {len(obj.get('hyde_profile', ''))}")