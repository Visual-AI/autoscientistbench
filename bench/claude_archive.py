"""Snapshot only Claude's native transcripts; never copy the authentication HOME."""
import json
import os
from pathlib import Path
import shutil
import sys
import time


def snapshot(home, record):
    home, record = Path(home), Path(record)
    source = home/'.claude/projects'
    dest = record/'trace/claude/projects'
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    if source.is_dir() and not source.is_symlink() and not source.parent.is_symlink():
        for directory, dirs, files in os.walk(source, followlinks=False):
            directory = Path(directory)
            dirs[:] = [d for d in dirs if not (directory/d).is_symlink()]
            for name in files:
                path = directory/name
                if path.suffix != '.jsonl' or path.is_symlink(): continue
                out = dest/path.relative_to(source)
                out.parent.mkdir(parents=True, exist_ok=True)
                temp = out.with_suffix('.tmp')
                try: shutil.copyfile(path, temp); temp.replace(out); count += 1
                except FileNotFoundError: pass
    (dest.parent/'manifest.json').write_text(json.dumps({'updated_ts': time.time(),
        'transcript_files': count, 'authentication_files_excluded': True})+'\n')
    if os.geteuid() == 0:
        uid, gid = record.stat().st_uid, record.stat().st_gid
        for path in [dest.parent, *dest.parent.rglob('*')]:
            os.chown(path, uid, gid); path.chmod(0o755 if path.is_dir() else 0o644)
    return count


if __name__ == '__main__':
    print(snapshot(*sys.argv[1:3]))
