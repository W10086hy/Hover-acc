import hashlib,json,os
from pathlib import Path
from io import BytesIO
import torch

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_checkpoint(path,map_location='cpu'):
    """Load and identify the same bytes, even if a training path is replaced."""
    payload=Path(path).read_bytes()
    return (torch.load(BytesIO(payload),map_location=map_location,weights_only=False),
            hashlib.sha256(payload).hexdigest())
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temporary,path)
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary);os.replace(temporary,path)
