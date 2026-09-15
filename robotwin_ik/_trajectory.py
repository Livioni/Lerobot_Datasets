"""CPU graph search and time-aware trajectory metrics, independent of CUDA."""
from __future__ import annotations

import numpy as np


def validate_inputs(q_initial, targets, timestamps, ik_seeds, max_joint_step_rad):
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('timestamps must be nonempty, finite and strictly increasing')
    if np.shape(targets) != (len(times), 4, 4) or not np.isfinite(targets).all():
        raise ValueError('targets must be finite [T,4,4]')
    if np.ndim(q_initial) != 1 or not np.isfinite(q_initial).all():
        raise ValueError('initial joint state must be a finite vector')
    if isinstance(ik_seeds, bool) or int(ik_seeds) != ik_seeds or not 2 <= ik_seeds <= 256:
        raise ValueError('ik_seeds must be an integer between 2 and 256')
    if not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0:
        raise ValueError('max_joint_step_rad must be finite and positive')


def motion_metrics(q, timestamps):
    """Return actual steps, interval velocities, and interior accelerations."""
    q = np.asarray(q, dtype=float)
    dt = np.diff(np.asarray(timestamps, dtype=float))
    steps = np.diff(q, axis=0)
    velocity = steps / dt[:, None]
    acceleration = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) * .5)[:, None]
    return steps, velocity, acceleration


def smoothness_cost(q, timestamps, initial):
    _, velocity, acceleration = motion_metrics(q, timestamps)
    dt = np.diff(timestamps)
    cost = .01 * float(np.sum((q[0] - initial) ** 2))
    if len(dt):
        cost += float(np.sum(velocity ** 2 * dt[:, None]))
    if len(acceleration):
        cost += float(np.median(dt) ** 2 * np.sum(acceleration ** 2 * ((dt[1:] + dt[:-1]) * .5)[:, None]))
    return cost


def select_paths(layers, timestamps, initial, step_limit, count=4, relaxed=False, node_penalties=None):
    """K shortest paths through all frames; no frame is committed greedily.

    Hard edges use actual joint differences. Relaxed edges are exclusively
    optimization initializations, and never constitute an accepted trajectory.
    """
    if not layers or any(len(layer) == 0 for layer in layers):
        return []
    costs = np.full((len(layers[0]), count), np.inf)
    costs[:, 0] = .01 * np.sum((layers[0] - initial) ** 2, axis=1)
    if node_penalties is not None:
        costs[:, 0] += node_penalties[0]
    parents = []
    for t in range(1, len(layers)):
        delta = layers[t][None, :, :] - layers[t-1][:, None, :]
        step = np.max(np.abs(delta), axis=-1)
        edge = np.sum(delta ** 2, axis=-1) / (timestamps[t] - timestamps[t-1])
        if relaxed:
            edge += 1e4 * np.maximum(step / step_limit - 1, 0) ** 2
        else:
            edge[step > step_limit] = np.inf
        if node_penalties is not None:
            edge += node_penalties[t][None, :]
        choices = (costs[:, :, None] + edge[:, None, :]).transpose(2, 0, 1).reshape(len(layers[t]), -1)
        selected = np.argsort(choices, axis=1, kind='stable')[:, :count]
        costs = np.take_along_axis(choices, selected, axis=1)
        parents.append(selected)
    paths = []
    for flat in np.argsort(costs.ravel(), kind='stable')[:count]:
        node, rank = divmod(int(flat), count)
        if not np.isfinite(costs[node, rank]):
            continue
        indices = [node]
        for parent in reversed(parents):
            node, rank = divmod(int(parent[node, rank]), count)
            indices.append(node)
        path = np.asarray([layer[index] for layer, index in zip(layers, reversed(indices))])
        if not any(np.array_equal(path, previous) for previous in paths):
            paths.append(path)
    return paths


def diverse_candidates(values, initial, maximum=256):
    """Deduplicate without identifying different legal windings, then retain diversity."""
    values = np.asarray(values, dtype=float).reshape(-1, len(initial))
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return values
    _, indices = np.unique(np.round(values, 4), axis=0, return_index=True)
    values = values[np.sort(indices)]
    if len(values) <= maximum:
        return values
    first = int(np.argmin(np.sum((values - initial) ** 2, axis=1)))
    chosen = [first]
    distance = np.sum((values - values[first]) ** 2, axis=1)
    for _ in range(maximum - 1):
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(distance, np.sum((values - values[index]) ** 2, axis=1))
    return values[chosen]


def interpolate_candidate_gaps(layers, timestamps, initial, step_limit):
    """Whole-path initialization through missing IK layers; never an accepted result.

    First choose branches jointly at frames with candidates, then interpolate
    their actual joint angles at missing timestamps. No frame is filled from a
    previously accepted output or reported as successful without validation.
    """
    known = [i for i, layer in enumerate(layers) if len(layer)]
    if not known:
        return np.tile(initial, (len(layers), 1))
    sparse = select_paths([layers[i] for i in known], np.asarray(timestamps)[known],
                          initial, step_limit, count=1, relaxed=True)[0]
    return np.column_stack([np.interp(timestamps, np.asarray(timestamps)[known], sparse[:, j])
                            for j in range(len(initial))])
