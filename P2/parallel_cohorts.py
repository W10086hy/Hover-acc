"""Independent Actor/plant replicas on private CUDA streams.

Only independent parameter domains run concurrently. Time within each BPTT
trajectory stays causal. Gradient reduction and the single Adam update belong
to the caller and run in deterministic domain order.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import torch


class CohortExecutor:
    def __init__(self, streams, actor, cuda_streams, checkpoint):
        from .hover_bptt.actor import Actor
        from .hover_bptt.optimizer import install_gradient_masks
        if not streams or len(streams) != len(cuda_streams):
            raise ValueError('One private CUDA stream per nonempty cohort required')
        if len({s.cuda_stream for s in cuda_streams}) != len(cuda_streams):
            raise ValueError('CUDA streams must be distinct')
        self.streams = streams
        self.cuda_streams = cuda_streams
        self.actors = []
        current = torch.cuda.current_stream()
        for context in cuda_streams:
            context.wait_stream(current)
            with torch.cuda.stream(context):
                # Tensor deepcopy does not preserve registered gradient hooks.
                # Install the original masks explicitly on each fresh replica.
                replica = Actor(checkpoint['configuration']['model']).to(context.device)
                replica.load_state_dict(actor.state_dict())
                install_gradient_masks(replica)
                replica.rnn.rnn.flatten_parameters()
            self.actors.append(replica)
        torch.cuda.synchronize()
        self.pool = ThreadPoolExecutor(max_workers=len(streams), thread_name_prefix='BPTT-cohort')
        self.lock = Lock()

    def map(self, function, actor, *, parallel=True):
        """Finish every job before propagating an error; never race restores."""
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('A cohort executor cannot be entered twice')
        try:
            current = torch.cuda.current_stream()
            for context, replica in zip(self.cuda_streams, self.actors):
                context.wait_stream(current)
                with torch.cuda.stream(context):
                    replica.load_state_dict(actor.state_dict())
                    replica.train(actor.training)

            def run(index):
                context = self.cuda_streams[index]
                with torch.cuda.device(context.device), torch.cuda.stream(context):
                    result = function(self.streams[index], self.actors[index])
                    context.synchronize()
                    return result

            if not parallel:
                return [run(index) for index in range(len(self.streams))]
            futures = [self.pool.submit(run, index) for index in range(len(self.streams))]
            results, errors = [], []
            for future in futures:
                try:
                    results.append(future.result())
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise errors[0]
            return results
        finally:
            # The root may now consume detached gradients from any worker.
            current = torch.cuda.current_stream()
            for context in self.cuda_streams:
                current.wait_stream(context)
            self.lock.release()

    def close(self):
        self.pool.shutdown(wait=True)

    def snapshots(self, actor, optimizer):
        from .hover_bptt.state import snapshot
        records = []
        current = torch.cuda.current_stream()
        for stream, context in zip(self.streams, self.cuda_streams):
            context.wait_stream(current)
            with torch.cuda.stream(context):
                records.append(snapshot(stream.env, stream, actor, optimizer, updates=0, episode=0, window=0))
            context.synchronize()
        return records

    def restore(self, actor, optimizer, records):
        from .hover_bptt.state import restore_stream
        if len(records) != len(self.streams):
            raise ValueError('One snapshot per cohort required')
        current = torch.cuda.current_stream()
        for stream, context, before in zip(self.streams, self.cuda_streams, records):
            context.wait_stream(current)
            with torch.cuda.stream(context):
                # Global RNG/Actor/Adam restores must remain serialized.
                restore_stream(stream.env, stream, actor, optimizer, before)
            context.synchronize()

    def forward(self, actor, *, steps, parallel=True):
        from .hover_bptt.flexible_training import advance_warmup
        from .hover_bptt.training import trajectory

        def run(stream, replica):
            advance_warmup(stream, replica)
            return trajectory(stream, replica, steps=steps)

        return self.map(run, actor, parallel=parallel)


def create_cohorts(protocol, saved, domains, seconds, device, actor):
    """Capture graphs serially, before launching concurrent replay/backward."""
    from .randomized_training import create_streams

    if torch.device(device).type != 'cuda':
        raise ValueError('Private-stream cohorts require CUDA')
    contexts, streams = [], []
    selected = protocol.get('training_domain_indices', list(range(len(domains))))
    current = torch.cuda.current_stream(device)
    cfg = cases = None
    for index in selected:
        context = torch.cuda.Stream(device=device)
        context.wait_stream(current)
        single = copy.deepcopy(protocol)
        single['training_domain_indices'] = [index]
        with torch.cuda.stream(context):
            group, cfg, cases = create_streams(single, saved, domains, seconds, device)
        context.synchronize()
        streams.extend(group)
        contexts.append(context)
    executor = CohortExecutor(streams, actor, contexts, saved)
    return streams, cfg, cases, executor


def gradients(executor, actor, *, steps, credit_steps, recompute_steps, parallel=True):
    from .hover_bptt.flexible_training import advance_warmup
    from .hover_bptt.training import trajectory

    started = time.perf_counter()
    warmups = executor.map(advance_warmup, actor, parallel=parallel)
    warmup_wall_s = time.perf_counter() - started

    def run(stream, replica):
        loss, diagnostics, score, commands = trajectory(stream, replica,
            gradient=True, steps=steps, credit_steps=credit_steps, recompute_steps=recompute_steps)
        values = {name: p.grad.detach().clone() for name, p in replica.named_parameters() if p.grad is not None}
        return dict(loss=loss, diagnostics=diagnostics, score=score, commands=commands, gradients=values)

    results = executor.map(run, actor, parallel=parallel)
    actor.zero_grad(set_to_none=True)
    for name, parameter in actor.named_parameters():
        contributions = [result['gradients'][name] for result in results if name in result['gradients']]
        if contributions:
            parameter.grad = torch.zeros_like(parameter)
            for value in contributions:
                parameter.grad.add_(value, alpha=1. / len(results))
    for stream, result, warmup_s in zip(executor.streams, results, warmups):
        result['diagnostics'].update(warmup_control_steps=getattr(stream, 'warmup_steps', 0),
            warmup_wall_s=warmup_s, parallel_warmup_wall_s=warmup_wall_s,
            cohort_threads=len(executor.streams) if parallel else 1)
        del result['gradients']
    return results
