"""Gera environment/requirements.txt e environment/environment.yml com as versões instaladas
no ambiente ativo (sem build strings). Rodar dentro do ambiente usado nos experimentos."""
import sys, os, platform
from importlib.metadata import version, PackageNotFoundError
PKGS = ['torch', 'tensorflow', 'ai-edge-litert', 'numpy', 'pandas', 'scipy', 'scikit-learn',
        'matplotlib', 'jupyter', 'nbconvert', 'ipykernel']
out = os.path.dirname(os.path.abspath(__file__))
lines, notes = [], []
for p in PKGS:
    try:
        v = version(p)
    except PackageNotFoundError:
        notes.append(f'# {p}: not installed in the authors\' environment'); continue
    if '+' in v:                       # e.g. 2.5.1+cu121 -> 2.5.1 (CUDA build noted)
        notes.append(f'# {p} was installed as {v}'); v = v.split('+')[0]
    lines.append(f'{p}=={v}')
py = '.'.join(platform.python_version_tuple()[:2])
hdr = f'# Python {platform.python_version()} on {platform.system()} {platform.machine()}\n'
open(f'{out}/requirements.txt', 'w').write(hdr + '\n'.join(notes + lines) + '\n')
yml = ['name: ecg-acf-afib', 'channels:', '  - conda-forge', 'dependencies:', f'  - python={py}', '  - pip',
       '  - pip:'] + [f'      - {l}' for l in lines]
open(f'{out}/environment.yml', 'w').write('\n'.join(yml) + '\n')
print(open(f'{out}/requirements.txt').read())
