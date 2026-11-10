"""Replay selected30s/120s cases without rerunning physical integration."""
import argparse,zipfile
from pathlib import Path
from PIL import Image
from P2.hover_bptt.io import digest,write
from .height_speed import ROOT
from .replay import Dashboard


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--cases',type=int,nargs='+',default=[0]);p.add_argument('--speed',type=float,default=3.);a=p.parse_args()
    out=Path(a.output).resolve();base=Path(a.root).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'track') or a.speed<=0:raise ValueError('Use a new track replay output and positive speed')
    out.mkdir(parents=True);sources={n:digest(ROOT/n) for n in ['track/replay_cases.py','track/replay.py','track/physics.py','P3/replay_attitude_wind.py']}
    with zipfile.ZipFile(out/'render_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,arcname=name)
    rows=[]
    for index in a.cases:
        dash=Dashboard(base,index);indices=list(range(0,len(dash.t),max(1,round(a.speed))));frames=[]
        if indices[-1]!=len(dash.t)-1:indices.append(len(dash.t)-1)
        try:
            for k,i in enumerate(indices):
                frame=dash.render(i,a.speed)
                if k==len(indices)//2:frame.save(out/f'case{index}.png')
                frames.append(frame.convert('P',palette=Image.Palette.ADAPTIVE,colors=192))
            target=out/f'case{index}.gif';frames[0].save(target,save_all=True,append_images=frames[1:],duration=round(100*max(1,round(a.speed))/a.speed),loop=0,optimize=False,disposal=2)
            assert dash.plant.mj_step_calls==0
            rows.append(dict(case_index=index,case=dash.case,metric_horizon_s=dash.duration,start_s=0.,end_s=float(dash.t[-1]),playback_speed=a.speed,
                file=target.name,sha256=digest(target),source_result_sha256=digest(dash.folder/'result.json'),source_trajectory_sha256=digest(dash.folder/'trajectory.npz'),mj_step_calls=0,
                physical_envelope_completed=dash.case.get('physical_envelope_completed',dash.case['completed']),
                failed_tail=('Continue physical motion after performance boundary;freeze only after physical hard failure' if dash.continuation else 'frozen display and explicit failure overlay') if not dash.case['completed'] else None))
            print(dict(stage='replay_complete',case=index,horizon_s=dash.duration),flush=True)
        finally:dash.renderer.close()
    write(out/'manifest.json',dict(source_sha256=sources,replays=rows))


if __name__=='__main__':main()
