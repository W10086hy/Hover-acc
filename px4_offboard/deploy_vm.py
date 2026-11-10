"""Guest-only clean deployment into this task's dedicated runtime directory."""
import argparse,json,shutil,zipfile
from pathlib import Path
import subprocess

def clear_existing(root):
    # rmtree does not traverse child symlinks;top-level links need unlink
    # explicitly,otherwise is_dir() would follow them before rmtree rejects.
    for item in root.iterdir():
        if item.is_symlink() or not item.is_dir():item.unlink()
        else:shutil.rmtree(item)

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
    root=Path('/home/s/hover_bptt_px4')
    if root.is_symlink() or root.resolve()!=root or not root.is_dir():raise ValueError('Unexpected task root')
    if subprocess.run(['pgrep','-x','px4'],capture_output=True).returncode==0:raise RuntimeError('Stop existing PX4 before deployment')
    bundle=Path(a.bundle).resolve()
    if bundle.is_relative_to(root):raise ValueError('Bundle must survive the task cleanup')
    with zipfile.ZipFile(bundle) as z:
        for name in z.namelist():
            if not(root/name).resolve().is_relative_to(root):raise ValueError('Archive path outside task')
        count=0;size=0;links=[]
        for item in root.rglob('*'):
            if item.is_symlink():links.append(str(item.relative_to(root)));continue
            if item.is_file():count+=1;size+=item.stat().st_size
        clear_existing(root)
        z.extractall(root)
    (root/'stable_retest/vm_deployment.json').write_text(json.dumps(dict(old_runtime_files_removed=count,old_runtime_bytes_removed=size,
        scope=str(root),actor='GitHub restored29',new_P3=True,old_Actor35_reused=False,unlinked_entries_without_following=links),indent=2)+'\n')

if __name__=='__main__':main()
