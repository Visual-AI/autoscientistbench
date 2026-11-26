"""Trusted replay of one trace; only the scorer constructs this object.

The true cache state and the miss count live here, in the scorer's process. A worker
learns an access only after every earlier decision has been committed, and it cannot
ask for more than the next batch.
"""
from belady import LINE_BITS, NUM_SETS, WAYS

SET_MASK = NUM_SETS - 1


class TraceReplay:
    def __init__(self, pcs, addresses):
        if not pcs or len(pcs) != len(addresses):
            raise ValueError('invalid trace')
        self.pcs, self.addresses = pcs, addresses
        self.lines = [a >> LINE_BITS for a in addresses]
        self.sets = [set() for _ in range(NUM_SETS)]
        self.index = 0
        self.misses = 0
        self.evictions = 0
        self.pending_set = None      # resident set of the access waiting for a victim
        self.pending_line = None
        self.exhausted = False
        self.finished = None
        self.violations = []

    def __call__(self, method, args, kwargs):
        try:
            if kwargs:
                raise ValueError('keyword arguments are unsupported')
            if self.finished is not None:
                raise ValueError('replay is closed')
            if method == 'step' and len(args) == 1:
                return self._step(args[0])
            if method == 'finish' and not args:
                return self._finish()
            raise ValueError('unknown or malformed request')
        except Exception as exc:
            self.violations.append(type(exc).__name__ + ': ' + str(exc))
            raise

    def _step(self, victim):
        if self.exhausted:
            raise ValueError('trace already delivered')
        if self.pending_set is not None:
            # A decision is due: nothing further is revealed until it names a resident line.
            if type(victim) is not int or victim not in self.pending_set:
                raise ValueError('illegal victim')
            self.pending_set.remove(victim)
            self.pending_set.add(self.pending_line)
            self.evictions += 1
            self.pending_set = self.pending_line = None
        elif victim is not None:
            raise ValueError('no eviction is due')
        start = i = self.index
        lines, sets, n = self.lines, self.sets, len(self.lines)
        while i < n:
            line = lines[i]
            resident = sets[line & SET_MASK]
            i += 1
            if line in resident:
                continue
            self.misses += 1
            if len(resident) < WAYS:
                resident.add(line)
                continue
            self.pending_set, self.pending_line = resident, line
            break
        self.index = i
        self.exhausted = i == n and self.pending_set is None
        return {'pc': self.pcs[start:i], 'addr': self.addresses[start:i],
                'evict': self.pending_set is not None, 'done': self.exhausted}

    def _finish(self):
        if not self.exhausted:
            raise ValueError('finish before the trace was delivered')
        self.finished = {'accesses': len(self.lines), 'misses': self.misses}
        return dict(self.finished)

    def accepted_record(self):
        if self.violations or self.finished is None:
            raise ValueError('replay protocol violation or unfinished trace')
        return self.finished
