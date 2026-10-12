"""Check maintained entrypoints and project instruction ownership, offline."""
import hashlib
import json
import os
from pathlib import Path

COMPATIBILITY_SCRIPTS={
    'run_vehicle_dashboard.sh', 'run_autodrive_panel.sh', 'run_vehicle_experiment.sh'}
SKILLS={'vehicle-cloud-thesis','thesis-handoff'}


def check_layout(root):
    root=Path(root)
    if {p.name for p in root.glob('run*.sh')}!=COMPATIBILITY_SCRIPTS:
        raise ValueError('public startup belongs in vehicle; only deployed-path adapters may remain')
    if not (root/'vehicle').is_file() or not os.access(str(root/'vehicle'),os.X_OK):
        raise ValueError('vehicle launcher missing or not executable')
    if (root/'skills').exists() or (root/'skills').is_symlink():
        raise ValueError('project skills must exist only under .agents/skills')
    found_rules=[]
    found_skills=[]
    for base,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if d not in ('.git','outputs','__pycache__','.conda')
                 and not (Path(base)/d).is_symlink()]
        if 'AGENTS.md'in files:found_rules.append((Path(base)/'AGENTS.md').relative_to(root).as_posix())
        if 'SKILL.md'in files:found_skills.append((Path(base)/'SKILL.md').relative_to(root).as_posix())
    if found_rules!=['AGENTS.md']:
        raise ValueError('project collaboration rules must have one root AGENTS.md: '+str(found_rules))
    expected={'.agents/skills/'+name+'/SKILL.md' for name in SKILLS}
    if set(found_skills)!=expected:raise ValueError('project skill discovery is duplicated or incomplete')
    for name in SKILLS:
        folder=root/'.agents/skills'/name
        if folder.is_symlink() or not (folder/'agents/openai.yaml').is_file():
            raise ValueError('skill must have one physical directory and its UI metadata: '+name)
    for row in json.loads((root/'coordination/entrypoint-2026-10-12/rule-migrations.json').read_text()):
        if hashlib.sha256((root/row['archive_path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('original rule archive changed: '+row['archive_path'])
    print('Layout: one public launcher, three deployed-path adapters, one AGENTS.md, two physical skills',flush=True)
