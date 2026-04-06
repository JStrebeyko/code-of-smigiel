"""Assemble TEST A and TEST B from postprocessed alpha/beta/gamma outputs.

Each source directory must contain three line-synchronized files:
  start.txt   — one text per line
  start.key   — one label per line (0 = human, 1 = generated)
  start.meta  — one TSV metadata row per line (hash\tsource\tstrategy)

Split ratios:
  alpha → halves  → ½ to TEST A, ½ to TEST B
  beta  → thirds  → ⅓ to TEST A, ⅔ to TEST B
  gamma → thirds  → ⅓ to TEST A, ⅔ to TEST B
"""

import heapq
import pathlib


def split_in(n, items):
    """Round-robin distribute items into n buckets.

    Element at index i goes to bucket i % n. Preserves relative order within
    each bucket. Returns a list of n lists.
    """
    buckets = []
    for i, element in enumerate(items):
        bucket_index = i % n
        try:
            buckets[bucket_index].append(element)
        except IndexError:
            buckets.append([element])
    return buckets


def interleave_lists(lists):
    """Interleave elements from multiple lists using a max-heap.

    Guarantees that no two consecutive elements come from the same source list
    (when avoidable). Longer lists are spread as evenly as possible; if one
    list dominates, its tail is appended at the end.

    Args:
        lists: sequence of lists to interleave.
    Returns:
        A single flat list.
    """
    active = [lst for lst in lists if lst]
    if not active:
        return []

    iters = [iter(lst) for lst in active]
    heap = [(-len(lst), i) for i, lst in enumerate(active)]
    heapq.heapify(heap)

    result = []
    prev = None  # (remaining_count, source_index) held aside for one step

    while heap:
        count, i = heapq.heappop(heap)

        try:
            result.append(next(iters[i]))
        except StopIteration:
            continue

        count += 1  # count is negative; +1 means one fewer remaining

        if prev and prev[0] < 0:
            heapq.heappush(heap, prev)

        prev = (count, i)

    # drain any tail from a dominant list
    if prev and prev[0] < 0:
        for x in iters[prev[1]]:
            result.append(x)

    return result


def load_txt_key_meta(directory):
    """Read start.txt / start.key / start.meta and return a list of (text, key, meta) tuples."""
    directory = pathlib.Path(directory)
    text  = (directory / 'start.txt').read_text().splitlines(keepends=True)
    key   = (directory / 'start.key').read_text().splitlines(keepends=True)
    meta  = (directory / 'start.meta').read_text().splitlines(keepends=True)
    return list(zip(text, key, meta))


def write_txt_key_meta(tuples, directory):
    """Write (text, key, meta) tuples to start.txt / start.key / start.meta."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'start.txt').open('w') as tf, \
         (directory / 'start.key').open('w') as kf, \
         (directory / 'start.meta').open('w') as mf:
        for text, key, meta in tuples:
            tf.write(text)
            kf.write(key)
            mf.write(meta)


if __name__ == '__main__':
    DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'

    # Postprocessed source directories (written by postprocess.py)
    alpha = load_txt_key_meta(DATA_PATH / 'postprocessed' / 'test')  # test split = alpha
    beta  = load_txt_key_meta(DATA_PATH / 'postprocessed_beta')
    gamma = load_txt_key_meta(DATA_PATH / 'postprocessed_gamma')

    alpha_a, alpha_b          = split_in(2, alpha)
    beta_a, *beta_b_parts     = split_in(3, beta)
    gamma_a, *gamma_b_parts   = split_in(3, gamma)

    beta_b  = beta_b_parts[0]  + beta_b_parts[1]
    gamma_b = gamma_b_parts[0] + gamma_b_parts[1]

    test_a = interleave_lists([alpha_a, beta_a, gamma_a])
    test_b = interleave_lists([alpha_b, beta_b, gamma_b])

    write_txt_key_meta(test_a, DATA_PATH / 'test_A')
    write_txt_key_meta(test_b, DATA_PATH / 'test_B')

    print(f"TEST A: {len(test_a)} examples")
    print(f"TEST B: {len(test_b)} examples")
