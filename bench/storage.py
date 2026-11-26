"""Actual scratch-disk headroom, independent of declared task CPU/RAM."""
import os
import shutil


def disk_state(path='/tmp', *, free_bytes=None, free_inodes=None, pause_gib=30,
               launch_gib=60, resume_gib=80):
    if free_bytes is None:
        free_bytes = shutil.disk_usage(path).free
    if free_inodes is None:
        fs = os.statvfs(path)
        free_inodes = fs.f_favail if fs.f_files else None
    return {'free_gib': round(free_bytes / 2**30, 3), 'free_inodes': free_inodes,
            'pause': free_bytes < pause_gib * 2**30 or (free_inodes is not None and free_inodes < 10000),
            'admit': free_bytes >= launch_gib * 2**30 and (free_inodes is None or free_inodes >= 50000),
            'resume': free_bytes >= resume_gib * 2**30 and (free_inodes is None or free_inodes >= 100000)}
