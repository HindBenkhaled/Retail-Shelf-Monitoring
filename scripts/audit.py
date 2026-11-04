"""Basic source secret/large-file/local-Markdown-link audit (not a security certification)."""
import json
import re
import subprocess
from pathlib import Path
from scripts.common import save_json


def main():
    root = Path.cwd()
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'], text=True).splitlines()
    paths = sorted({Path(name) for name in names})
    findings, broken = [], []
    patterns = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                r'\bgh[pousr]_[A-Za-z0-9]{30,}\b', r'\bAKIA[0-9A-Z]{16}\b',
                r'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b']
    for path in paths:
        if path.stat().st_size > 10*1024*1024:
            findings.append({'file':str(path),'issue':'file over 10 MiB'})
        if path.suffix.lower() in ('.pt','.onnx','.pyc','.cache') or path.name == '.env':
            findings.append({'file':str(path),'issue':'private/generated artifact included'})
        if path.suffix.lower() in ('.png','.jpg','.jpeg','.zip'):
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for pattern in patterns:
            if re.search(pattern,text):
                findings.append({'file':str(path),'issue':'possible credential pattern'})
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if target.startswith(('http:','https:','#','mailto:')):
                    continue
                local = target.split('#')[0].strip('<>')
                if not (path.parent/local).exists():
                    broken.append({'file':str(path),'target':target})
    report = {'files_checked':len(paths),'credential_or_artifact_findings':findings,'broken_local_links':broken,
              'scope':'Basic pattern scan; cannot prove absence of every secret. External references researched separately.'}
    save_json('results/repository_audit.json',report)
    print(json.dumps(report,indent=2))
    if findings or broken:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
