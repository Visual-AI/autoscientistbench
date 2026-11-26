"""Cache geometry and the offline optimum (Belady's rule), computed by the scorer itself."""
LINE_BITS = 6
NUM_SETS = 2048
WAYS = 16
_NEVER = float('inf')


def lines_of(addresses):
    return [a >> LINE_BITS for a in addresses]


def opt_misses(lines):
    """Misses of Belady's rule run independently in each set, with no bypass (the paper's paging model)."""
    next_use = [0] * len(lines)
    last = {}
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        next_use[i] = last.get(line, _NEVER)
        last[line] = i
    sets = [{} for _ in range(NUM_SETS)]
    mask = NUM_SETS - 1
    misses = 0
    for i, line in enumerate(lines):
        resident = sets[line & mask]
        if line not in resident:
            misses += 1
            if len(resident) >= WAYS:
                del resident[max(resident, key=resident.__getitem__)]
        resident[line] = next_use[i]
    return misses
