"""Summarize paired 120 s MuJoCo validation with uncentered position errors."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from P2.hover_bptt.io import digest, load_checkpoint, read, write
from stable_retest.prepare import ACTOR, CASES, RESIDUAL, ROOT
from track.drift8.validate import baseline_folder


def report(folder, checkpoint=None):
    target = folder / 'evaluation.json'
    if target.exists():
        raise FileExistsError(target)
    saved, checkpoint_sha = load_checkpoint(checkpoint or folder / 'latest.pt')
    cases = [read(folder / 'mujoco' / f'arm{arm}_{case}' / 'comparison.json')
        for arm in (5, 10) for case, _, _ in CASES]
    assert all(row['controller_sha256'] == checkpoint_sha
        and row['residual_sha256'] == digest(RESIDUAL) for row in cases)
    assert digest(ACTOR) == saved['height_speed_continuation']['original_actor_sha256']
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, layout='constrained')
    fig.suptitle('技术路线：BPTT\n8 m 恒风下的水平距离 | MuJoCo 120 秒配对对照')
    for ax, row in zip(axes.flat, cases):
        arm, name = row['attachment_cm'], row['case']
        base = baseline_folder(folder, arm, name)
        candidate = folder / 'mujoco' / f'arm{arm}_{name}'
        for case_folder, label, color in [(base, 'Actor29', '#666666'), (candidate, '续训候选', '#137eb4')]:
            with np.load(case_folder / 'fullrate.npz') as trace:
                radius = np.linalg.norm(trace['tracking_error'][::10, 0, :2], axis=1)
                valid = trace['physical_failure_code'][::10, 0] == 0
                ax.plot(trace['times'][::10], np.where(valid, radius, np.nan), color=color, lw=1.1, label=label)
        ax.axvspan(90, 120, alpha=.12, color='#137eb4')
        ax.set(title=f'{arm} cm | '+{'hover_wind1':'悬停 1 m/s','hover_wind5':'悬停 5 m/s','track4_wind1':'跟踪 4 m/s + 风 1 m/s'}[name],
            xlim=(0,120))
        ax.grid(alpha=.22)
    axes[0,0].set_ylim(bottom=0.)
    axes[0,0].legend()
    fig.supxlabel('仿真时间 (s)')
    fig.supylabel('到目标的水平距离 (m)')
    chart = folder / 'drift_comparison.png'
    fig.savefig(chart, dpi=140)
    plt.close(fig)
    aggregates = {}
    for label in ('baseline','candidate'):
        rows = [case[label] for case in cases]
        all_errors = np.array([row['rmse_3d_h_xy_m'] for row in rows])
        late_errors = np.array([row['late30_xy_rmse_m'] for row in rows])
        aggregates[label] = dict(safe=sum(row['safe'] for row in rows), legacy_safe=sum(row['legacy_safe'] for row in rows),
            rmse_3d_h_xy_m=np.sqrt(np.mean(all_errors**2, axis=0)).tolist(),
            late30_xy_rmse_m=float(np.sqrt(np.mean(late_errors**2))))
    write(target, dict(duration_s=120., backend='native MuJoCo', test_scope='paired development benchmark',
        formal_final_acceptance=False, wind='steady +Y; no gust or turbulence', cases=cases, aggregates=aggregates,
        baseline_checkpoint_sha256=digest(ACTOR), residual_checkpoint_sha256=digest(RESIDUAL),
        candidate_checkpoint_sha256=checkpoint_sha, chart_sha256=digest(chart)))
    print(aggregates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    report(args.folder, args.checkpoint)
