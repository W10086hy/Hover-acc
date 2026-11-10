"""Resolve byte-preserved artifacts relocated by the documented cleanup."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def resolve_artifact(name,*,directory=False):
    path=Path(name)
    root=ROOT.resolve()
    if path.is_absolute():
        resolved=path.resolve()
        if not resolved.is_relative_to(root):raise ValueError('Artifact outside project')
        relative=resolved.relative_to(root).as_posix()
    else:relative=path.as_posix().replace('\\','/')
    candidate=(root/relative).resolve()
    if not candidate.is_relative_to(root):raise ValueError('Artifact outside project')
    exists=lambda p:p.is_dir() if directory else p.is_file()
    if exists(candidate):return candidate
    mapping=json.loads((ROOT/'cleanup/path_map.json').read_text(encoding='utf-8'))
    for old,new in sorted(mapping.items(),key=lambda item:-len(item[0])):
        if relative==old or relative.startswith(old+'/'):
            relocated=(root/new/relative[len(old):].lstrip('/')).resolve()
            if not relocated.is_relative_to(root):raise ValueError('Mapped artifact outside project')
            if exists(relocated):return relocated
    raise FileNotFoundError('Artifact not retained: '+relative)
