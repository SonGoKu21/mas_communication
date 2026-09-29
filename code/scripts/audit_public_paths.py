"""Check tracked text for machine-specific paths and non-example network addresses."""
import ipaddress
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
violations = []
for name in filter(None, paths):
    try:
        text = (root / name).read_text()
    except (UnicodeError, OSError):
        continue
    for number, line in enumerate(text.splitlines(), 1):
        # The sanitizer test uses a synthetic home directory.
        homes = re.findall(r'/(?:Users|home)/([A-Za-z0-9_][^/\s\x22\x27|]+)', line)
        if any(user not in {'example-user'} for user in homes):
            violations.append((name, number, 'personal home path'))
        if re.search(r'/data\d+/', line):
            violations.append((name, number, 'machine storage path'))
        for match in re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', line):
            try:
                address = ipaddress.ip_address(match)
            except ValueError:
                continue
            examples = ('192.0.2.', '198.51.100.', '203.0.113.')
            if not (address.is_loopback or address.is_unspecified or match.startswith(examples)):
                violations.append((name, number, 'non-example IP address'))
for name, number, kind in violations:
    print(f'{name}:{number}: {kind}')
print(f'{len(violations)} public-path findings')
sys.exit(bool(violations))
