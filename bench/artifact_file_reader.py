"""Read one private candidate file without following any symlink or changing it."""
import base64
import json
import os
from pathlib import Path
import stat
import sys


def read_file(workspace, relative):
    root, rel = Path(workspace), Path(relative)
    if not root.is_absolute() or '..' in root.parts or rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Expected an absolute workspace and a contained relative path')
    if rel.parts[0] in ('data', 'bench', 'paper') or '.git' in rel.parts or '__pycache__' in rel.parts:
        raise ValueError('Path is outside the candidate archive scope')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for name in (*root.parts[1:], *rel.parts[:-1]):
            nxt = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd);fd = nxt
        file_fd = os.open(rel.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, 'rb') as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):raise ValueError('Not a regular candidate file')
            while chunk := source.read(65536):
                print(json.dumps({'data': base64.b64encode(chunk).decode()}), flush=True)
            after = os.fstat(source.fileno())
            print(json.dumps({'metadata': {'inode': after.st_ino, 'size': after.st_size,
                'mtime_ns': after.st_mtime_ns, 'mode': stat.S_IMODE(after.st_mode)},
                'changed': (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)}), flush=True)
    except (FileNotFoundError, NotADirectoryError):
        print(json.dumps({'gone': True}), flush=True)
    finally:
        os.close(fd)


if __name__ == '__main__':read_file(*sys.argv[1:3])
