"""Privileged read-only native-session bridge, with fd-relative no-symlink traversal.

The host archive writer stays unprivileged. No credentials or operator sessions
are read: the sole root is one run's isolated HOME/.kimi-code/sessions.
"""
import base64
import json
import os
from pathlib import Path
import stat
import sys


def directory(parent, name):
    return os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent)


def walk(fd, prefix, offsets):
    with os.scandir(fd) as scan:
        names=[e.name for e in scan]
    for name in names:
        rel=prefix+'/'+name if prefix else name
        try:
            st=os.stat(name,dir_fd=fd,follow_symlinks=False)
            if stat.S_ISDIR(st.st_mode):
                child=directory(fd,name)
                try:walk(child,rel,offsets)
                finally:os.close(child)
            elif stat.S_ISREG(st.st_mode) and (name in ('wire.jsonl','state.json') or name.endswith('.log')):
                file_fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=fd)
                with os.fdopen(file_fd,'rb') as src:
                    size=os.fstat(src.fileno()).st_size
                    state=name=='state.json';previous=offsets.get(rel,0);reset=not state and size<previous
                    offset=0 if state or reset else previous
                    src.seek(offset)
                    if state:
                        if size>262144:raise ValueError('native state unexpectedly large')
                        data=src.read()
                        print(json.dumps({'path':rel,'state':True,'offset':0,'data':base64.b64encode(data).decode()}),flush=True)
                    else:
                        remaining=4*1024*1024
                        while remaining>0:
                            data=src.read(min(65536,remaining))
                            if not data:break
                            print(json.dumps({'path':rel,'state':False,'offset':offset,'reset':reset,
                                              'data':base64.b64encode(data).decode()}),flush=True)
                            offset+=len(data);remaining-=len(data);reset=False
        except (FileNotFoundError,NotADirectoryError):
            continue


def main():
    home=Path(sys.argv[1])
    if not home.is_absolute() or '..' in home.parts:raise ValueError('Expected an absolute isolated run HOME')
    offsets=json.loads(sys.stdin.readline())
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in (*home.parts[1:],'.kimi-code','sessions'):
            next_fd=directory(fd,part);os.close(fd);fd=next_fd
        walk(fd,'',offsets)
    except FileNotFoundError:
        pass
    finally:os.close(fd)


if __name__=='__main__':main()
