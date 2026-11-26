"""Candidate-side client of the trusted paging replay. Holds no private state.

The scorer owns the trace, the true cache contents and the miss counts. A policy
worker sees accesses only through step(): each call returns every access up to and
including the next miss on a full set, and the next call must carry the victim for
that miss. See io_spec.md.
"""
import json
from private_rpc import call

LINE_BITS = 6          # 64-byte cache lines
NUM_SETS = 2048        # 2 MB / 64 B / 16 ways
WAYS = 16              # k in the paper


def line_of(address):
    """Cache-line number of a byte address; this is the page identity in the paging model."""
    return address >> LINE_BITS


def set_of(line):
    return line & (NUM_SETS - 1)


class Batch:
    """Accesses revealed by one step() call.

    pcs, addresses: parallel lists of ints, in trace order.
    needs_victim:   True when the last access missed on a full set; the next step() must name a victim.
    done:           True when the trace is exhausted and no decision is pending.
    """
    __slots__ = ('pcs', 'addresses', 'needs_victim', 'done')

    def __init__(self, reply):
        self.pcs, self.addresses = reply['pc'], reply['addr']
        self.needs_victim, self.done = reply['evict'], reply['done']


def step(victim=None):
    """Apply the pending eviction, if any, and return the next Batch.

    victim is the line number (line_of(address)) of a resident line of the missing
    access's set. Pass None only when no decision is pending; the first call passes None.
    """
    return Batch(call('step', victim))


def finish():
    """Close the replay once done is True; returns the trusted {'accesses', 'misses'} record."""
    return call('finish')


def write_output(path, record):
    """Write the trusted record to --output; the scorer requires exact agreement with its own count."""
    with open(path, 'w') as f:
        f.write(json.dumps({'accesses': record['accesses'], 'misses': record['misses']}) + '\n')
