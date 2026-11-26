"""Compile editable CLARITree using bundled headers and the active Python ABI."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

def build(output):
    code = Path(__file__).resolve().parent
    target = Path(output).resolve() / 'native' / 'clari_tree'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(code / 'clari_tree/__init__.py', target / '__init__.py')
    compiler = os.environ.get('CXX', 'c++')
    common = [compiler, '-O2', '-DNDEBUG', '-std=c++17', '-fPIC', '-fvisibility=hidden']
    for path in [code/'include', code/'vendor/eigen3', code/'vendor', Path(sysconfig.get_paths()['include'])]:
        common += ['-I', str(path)]
    objects = []
    for name in ('clari_tree', 'clari_tree_const', 'bindings'):
        obj = target / (name + '.o')
        subprocess.run([*common, '-c', str(code/'src'/(name+'.cpp')), '-o', str(obj)], check=True)
        objects.append(obj)
    extension = target / ('_core' + sysconfig.get_config_var('EXT_SUFFIX'))
    subprocess.run([compiler, '-shared', *map(str, objects), '-o', str(extension)], check=True)
    for obj in objects:
        obj.unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    build(parser.parse_args().out)
