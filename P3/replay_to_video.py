"""H264 companion for large scientific replay GIFs using existing Windows codecs."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from P2.hover_bptt.io import read,write,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--replay',required=True);a=p.parse_args();stem=Path(a.replay)
    source=stem.with_suffix('.gif');target=stem.with_suffix('.mp4');metadata=Path(str(stem)+'.video.json')
    if target.exists() or metadata.exists():raise FileExistsError('Preserve existing video')
    archived=read(stem.with_suffix('.json'));source_sha=digest(source)
    assert source_sha==archived['gif_sha256']
    with Image.open(source) as gif:
        width,height=gif.size;fps=archived['fps'];count=gif.n_frames
        writer=cv2.VideoWriter(str(target),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),fps,(width,height))
        if not writer.isOpened():raise RuntimeError('Existing Windows H264 encoder unavailable')
        try:
            for i in range(count):
                gif.seek(i);writer.write(cv2.cvtColor(np.asarray(gif.convert('RGB')),cv2.COLOR_RGB2BGR))
        finally:writer.release()
    reader=cv2.VideoCapture(str(target));decoded=int(reader.get(cv2.CAP_PROP_FRAME_COUNT));video_fps=reader.get(cv2.CAP_PROP_FPS)
    assert decoded==count and abs(video_fps-fps)<1e-6
    assert int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))==width and int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))==height
    errors=[]
    with Image.open(source) as gif:
        for i in [0,count//2,count-1]:
            reader.set(cv2.CAP_PROP_POS_FRAMES,i);okay,frame=reader.read();assert okay
            gif.seek(i);reference=cv2.cvtColor(np.asarray(gif.convert('RGB')),cv2.COLOR_RGB2BGR)
            errors.append(float(np.sqrt(((frame.astype(float)-reference)**2).mean())))
    reader.release();assert digest(source)==source_sha
    write(metadata,dict(codec='H264 via Windows Media Foundation',opencv_version=cv2.__version__,frames=count,fps=fps,
        width=width,height=height,video_duration_s=count/fps,simulation_horizon_s=archived['metric_horizon_s'],playback_speed=archived['playback_speed'],
        source_gif_sha256=source_sha,video_sha256=digest(target),converter_sha256=digest(__file__),sampled_decoded_pixel_rms_error=errors,
        frames_times_and_dimensions_preserved=True,new_simulation=False))
    print(dict(output=str(target),bytes=target.stat().st_size,frames=count),flush=True)


if __name__=='__main__':main()
