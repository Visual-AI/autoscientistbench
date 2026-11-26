"""Trusted temporal transcript. No candidate code runs in this module."""
import csv
import math
import statistics


def read_rows(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


class Replay:
    def __init__(self, input_path, labels_path):
        rows = read_rows(input_path)
        labels = read_rows(labels_path)
        ids = [row['id'] for row in rows]
        if ids != sorted(set(ids)):
            raise ValueError('private history is not chronological')
        requested = [row['id'] for row in rows if row['forecast'] == '1']
        if [row['id'] for row in labels] != requested:
            raise ValueError('private labels and requested weeks differ')
        self.truth = {row['id']: float(row['y']) for row in labels}
        if not self.truth or not all(math.isfinite(y) and y > 0 for y in self.truth.values()):
            raise ValueError('invalid private target')
        self.events = []
        for row in rows:
            if row['forecast'] == '1':
                self.events.append({'op': 'forecast', 'target': {
                    'id': row['id'], 'N': int(row['N']), 'V': float(row['V'])}})
            value = float(row['L']) if row['L'] else self.truth[row['id']]
            self.events.append({'op': 'observe', 'row': {
                'id': row['id'], 'N': int(row['N']), 'L': value, 'V': float(row['V'])}})
        self.index = 0
        self.waiting = False
        self.failed = False
        self.predictions = []
        self.last_observation = None

    def __call__(self, method, args, kwargs):
        try:
            if self.failed or kwargs:
                raise ValueError('invalid temporal request')
            if method == 'next' and not args:
                if self.waiting:
                    raise ValueError('submit the current decision before requesting another observation')
                if self.index == len(self.events):
                    return {'op': 'done'}
                self.waiting = True
                return self.events[self.index]
            if method != 'submit' or len(args) != 1 or not self.waiting:
                raise ValueError('unexpected temporal operation')
            event, reply = self.events[self.index], args[0]
            if not isinstance(reply, dict):
                raise ValueError('reply must be an object')
            if event['op'] == 'observe':
                if reply != {'observed': event['row']['id']}:
                    raise ValueError('observation acknowledgement mismatch')
                self.last_observation = event['row']
            else:
                required = {'id', 'alpha', 'beta'}
                if set(reply) not in (required, required | {'p'}) or reply['id'] != event['target']['id']:
                    raise ValueError('BERM forecast requires id and calibrated Beta alpha/beta')
                if self.last_observation is None:
                    raise ValueError('BERM forecast requires preceding observed topology')
                parameters = []
                for key in ('alpha', 'beta'):
                    v = reply[key]
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                        raise ValueError('Beta parameters must be finite positive numbers')
                    parameters.append(float(v))
                alpha, beta = parameters
                current = self.last_observation
                pairs = current['N'] * (current['N'] - 1) / 2
                if not 0 <= current['L'] <= pairs:
                    raise ValueError('invalid observed BERM link count')
                if pairs <= 0:
                    probability = 0.0
                else:
                    a_post = alpha + current['L']
                    b_post = beta + (pairs - current['L'])
                    denominator = a_post + b_post
                    if math.isfinite(denominator):
                        probability = a_post / denominator
                    else:
                        # Equivalent posterior mean without overflow of a+b.
                        scale = max(a_post, b_post)
                        probability = (a_post / scale) / (a_post / scale + b_post / scale)
                    # Retain the original selected solver's clipping convention.
                    probability = min(1.0 - 1e-9, max(1e-9, probability))
                if 'p' in reply:
                    claimed = reply['p']
                    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed):
                        raise ValueError('claimed probability must be finite numeric')
                    if not math.isclose(claimed, probability, rel_tol=1e-12, abs_tol=1e-14):
                        raise ValueError('claimed probability differs from BERM posterior mean')
                value = float(event['target']['V']) * probability
                if not math.isfinite(value):
                    raise ValueError('nonfinite BERM expected link count')
                self.predictions.append((reply['id'], value))
            self.index += 1
            self.waiting = False
            return {'accepted': True}
        except Exception:
            self.failed = True
            raise

    def result(self):
        if self.failed or self.waiting or self.index != len(self.events):
            raise ValueError('incomplete or invalid temporal transcript')
        if [i for i, _ in self.predictions] != list(self.truth):
            raise ValueError('missing trusted forecasts')
        errors = [abs(value - self.truth[i]) / self.truth[i] for i, value in self.predictions]
        return {'re_l': statistics.fmean(errors), 'n': len(errors),
                'format_guard_ok': True, 'causality_guard_ok': True,
                'causality_guard': 'trusted_sequential_release',
                'research_object_guard': 'trusted_beta_berm_posterior_mean',
                'prior_calibration_provenance': 'requires_source_audit',
                'protocol': 'private-policy-v2'}
