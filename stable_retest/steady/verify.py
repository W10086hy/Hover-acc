"""Final source,model,constant-wind and media integrity check for12 cases."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from stable_retest.prepare import ROOT,ACTOR,RESIDUAL,sha

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--write-manifest',action='store_true');a=p.parse_args()
    e=json.loads((HERE/'evaluation.json').read_text());assert e['passed'] and e['count']==12
    assert e['safe_completed']==12 and e['policy_updates']==e['residual_updates']==0
    assert e['actor_sha256']==sha(ACTOR) and e['residual_sha256']==sha(RESIDUAL)
    assert e['source_sha256']==sha(HERE/'report.py')
    for r in e['cases']:
        folder=ROOT/r['path'];assets=HERE/f"arm{r['attachment_cm']}"/'assets'
        for n,h in r['source_sha256'].items():assert sha(folder/n)==h
        p=json.loads((folder/'protocol.json').read_text())
        assert p['config']['protocol']['wind_profile']=='steady' and p['config']['protocol']['case_specs'][0]['attachment_cm']==r['attachment_cm']
        for n,h in p['asset_sha256'].items():assert sha(assets/n)==h
        with zipfile.ZipFile(folder/'frozen_source.zip') as z:
            for n,h in p['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        if r['backend']=='mujoco':wind=torch.load(folder/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps'].numpy()[:,0]
        else:
            wind=np.load(folder/'trajectory.npz')['wind'];audit=json.loads((folder/'audit.json').read_text());assert audit['passed']
            for n,h in audit['artifact_sha256'].items():assert sha(folder/n)==h
        expected=np.broadcast_to([0.,r['wind_speed_mps'],0.],wind.shape)
        np.testing.assert_allclose(wind,expected,atol=1e-12,rtol=0)
        assert np.ptp(wind,axis=0).max()==0.
        m=json.loads((folder/'replay/manifest.json').read_text());assert sha(folder/'replay/manifest.json')==r['replay_manifest_sha256']
        assert m['GIF_sha256']==sha(folder/'replay/flight.gif')==r['GIF_sha256']
        assert m['render_source_sha256']==sha(ROOT/'px4_offboard/replay.py') and m['attachment_cm']==r['attachment_cm']
        assert m['wind_profile']=='steady' and m['equal_xyz_scale'] and m['frames']==201 and m['end_s']==120 and m['playback_speed']==6
        with Image.open(folder/'replay/flight.gif') as im:assert im.n_frames==201
    out=dict(passed=True,full_duration_cases=12,safe_completed=12,constant_wind_verified_all_samples=True,attachments_cm=[5,10],
        full120s_equal_xyz_GIFs=12,network_and_Adam_unchanged=True,policy_updates=0,residual_updates=0,formal_final_acceptance=False)
    (HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
    target=HERE/'manifest.json'
    if a.write_manifest:
        files=[]
        for base in [HERE,ROOT/'px4_offboard']:
            for f in sorted(base.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts and f!=target:files.append(dict(path=f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
        target.write_text(json.dumps(dict(count=len(files),total_bytes=sum(f['bytes'] for f in files),files=files),indent=2)+'\n')
    if target.exists():
        m=json.loads(target.read_text())
        for r in m['files']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
        out['files_verified']=m['count']
    print(json.dumps(out))

if __name__=='__main__':main()
