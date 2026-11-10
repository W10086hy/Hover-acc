"""No-update timing and numerical parity check of threaded CUDA BPTT."""
import argparse
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import torch

from .hover_bptt.actor import Actor
from .hover_bptt.flexible_training import ROOT, REPO, source_manifest
from .hover_bptt.io import digest, load_checkpoint, read, write
from .hover_bptt.optimizer import initialize
from .parallel_cohorts import create_cohorts, gradients
from .randomized_training import domains_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=1.)
    parser.add_argument('--forward-seconds', type=float, default=0.)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--concurrent-gpu-jobs', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    steps = round(args.seconds*100)
    if (not 0 < args.seconds <= 30 or args.repeats < 1 or
            abs(args.seconds*100-steps) >= 1e-9 or steps % min(100, max(steps, 1)) or
            args.forward_seconds < 0 or abs(args.forward_seconds*100-round(args.forward_seconds*100)) >= 1e-9):
        raise ValueError('Positive whole control steps and at least one repeat required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = source_manifest(args.protocol)
    for name in ['P2/parallel_cohorts.py', 'P2/benchmark_parallel_cohorts.py']:
        sources[name] = digest(REPO/name)
    with ZipFile(output/'frozen_source.zip', 'x', ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(REPO/name, name)
    protocol = read(args.protocol)
    # Timing isolates gradient execution; causal prefix timing is separate.
    protocol['warmup_s_by_domain'] = [0] * protocol['domain_randomization']['sample_count']
    protocol['late_stationarity_window'] = False
    saved, parent_sha = load_checkpoint(ROOT / protocol['parent_checkpoint'])
    actor = Actor(saved['configuration']['model']).cuda()
    actor.load_state_dict(saved['actor'])
    actor.train()
    optimizer = initialize(actor, saved)
    started = time.perf_counter()
    streams, _, cases, executor = create_cohorts(protocol, saved, domains_for(protocol),
        max(args.seconds, args.forward_seconds), 'cuda', actor)
    snapshots = executor.snapshots(actor, optimizer)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - started
    rows = []
    references = {}
    try:
        kinds = ['gradient'] + (['forward'] if args.forward_seconds else [])
        for kind in kinds:
          for repeat in range(args.repeats):
            for parallel in ([False, True] if repeat % 2 == 0 else [True, False]):
                executor.restore(actor, optimizer, snapshots)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                if kind == 'gradient':
                    results = gradients(executor, actor, steps=steps, credit_steps=steps,
                        recompute_steps=min(100, steps), parallel=parallel)
                else:
                    raw = executor.forward(actor, steps=round(args.forward_seconds/.01), parallel=parallel)
                    results = [dict(score=result[2], commands=result[3]) for result in raw]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                flat = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None]).cpu()
                commands = torch.stack([result['commands'] for result in results]).cpu()
                state = torch.stack([stream.env.state for stream in streams]).cpu()
                if kind not in references:
                    references[kind] = flat, commands, state
                reference_grad, reference_commands, reference_state = references[kind]
                grad_abs = float((flat-reference_grad).abs().max())
                grad_rel = float((flat-reference_grad).norm() / reference_grad.norm().clamp_min(1e-12))
                command_abs = float((commands-reference_commands).abs().max())
                state_abs = float((state-reference_state).abs().max())
                parity = grad_rel <= 1e-5 and command_abs <= 2e-6 and state_abs <= 2e-6
                masks_frozen = bool(torch.equal(actor.fast_head.weight.grad[:, :8],
                    torch.zeros_like(actor.fast_head.weight.grad[:, :8])))
                row = dict(kind=kind, repeat=repeat, threads=len(streams) if parallel else 1, wall_s=elapsed,
                    gradient_max_abs_diff=grad_abs, gradient_relative_l2_diff=grad_rel,
                    command_max_abs_diff=command_abs, state_max_abs_diff=state_abs,
                    gradient_norm=float(flat.norm()), parity=parity, frozen_gradient_mask_preserved=masks_frozen,
                    peak_memory_bytes=torch.cuda.max_memory_allocated(),
                    safe=[sum(result['score']['completed']) for result in results])
                rows.append(row)
                print(row, flush=True)
                if not parity or not masks_frozen:
                    raise RuntimeError('Parallel gradient/state/mask parity failed')
        ratios = {}
        for kind in kinds:
            sequential = [row['wall_s'] for row in rows if row['threads'] == 1 and row['kind'] == kind]
            threaded = [row['wall_s'] for row in rows if row['threads'] > 1 and row['kind'] == kind]
            ratios[kind] = sum(sequential) / sum(threaded)
        speedup = ratios['gradient']
        unchanged = all(torch.equal(tensor.detach().cpu(), saved['actor'][name]) for name, tensor in actor.state_dict().items())
        if not unchanged or digest(ROOT / protocol['parent_checkpoint']) != parent_sha or any(digest(REPO/name) != sha for name, sha in sources.items()):
            raise RuntimeError('Timing benchmark changed weights or sources')
        report = dict(parent_checkpoint_sha256=parent_sha, seconds=args.seconds, cohorts=len(streams),
            cases_per_cohort=len(cases), setup_s=setup_s, measurements=rows, speedup=speedup,
            speedup_by_kind=ratios, forward_seconds=args.forward_seconds,
            faster=speedup > 1.05, parity_passed=True, optimizer_updates=0, weights_unchanged=unchanged,
            torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
            torch_cpu_threads=torch.get_num_threads(),
            includes_90s_warmup=False, concurrent_gpu_jobs=args.concurrent_gpu_jobs,
            source_sha256=sources)
        write(output/'benchmark.json', report)
        print(json.dumps({key: value for key, value in report.items() if key not in ['measurements', 'source_sha256']}, ensure_ascii=False), flush=True)
    finally:
        executor.close()


if __name__ == '__main__':
    main()
