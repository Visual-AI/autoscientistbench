"""Trusted one-panel state machine. Future prices never enter the worker mount."""
import csv
import math

PANELS = ('NASDAQ', 'CSI300', 'TPX100', 'Global30')


def read_panels(path):
    result = {panel: [] for panel in PANELS}
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        expected = ['id', 'panel', 'step', 'date', 'active'] + ['p%02d' % i for i in range(30)]
        if reader.fieldnames != expected:
            raise ValueError('invalid private price header')
        for row in reader:
            result[row['panel']].append(row)
    for panel, rows in result.items():
        if not rows or [int(r['step']) for r in rows] != list(range(len(rows))):
            raise ValueError('private panel is not complete and chronological')
    return result


class Replay:
    def __init__(self, panel, rows):
        self.panel = panel
        self.events = [{'op': 'start', 'panel': panel}]
        self.want = []
        for row in rows:
            prices = [float(row['p%02d' % i]) for i in range(30)]
            if not all(math.isfinite(x) and x > 0 for x in prices):
                raise ValueError('invalid private price')
            active = row['active'] == '1'
            self.events.append({'op': 'step', 'id': row['id'], 'step': int(row['step']),
                                'prices': prices, 'active': active})
            if active:
                self.want.append(row['id'])
        self.index = 0
        self.waiting = False
        self.failed = False
        self.orders = []

    def __call__(self, method, args, kwargs):
        try:
            if self.failed or kwargs:
                raise ValueError('invalid temporal request')
            if method == 'next' and not args:
                if self.waiting:
                    raise ValueError('submit before requesting a later close')
                if self.index == len(self.events):
                    return {'op': 'done'}
                self.waiting = True
                return self.events[self.index]
            if method != 'submit' or len(args) != 1 or not self.waiting:
                raise ValueError('unexpected temporal operation')
            event, reply = self.events[self.index], args[0]
            if not isinstance(reply, dict):
                raise ValueError('reply must be an object')
            if event['op'] == 'start':
                if reply != {'started': self.panel}:
                    raise ValueError('panel acknowledgement mismatch')
            else:
                if set(reply) != {'id', 'orders'} or reply['id'] != event['id']:
                    raise ValueError('order identity mismatch')
                values = reply['orders']
                if not event['active']:
                    if values is not None:
                        raise ValueError('no orders during warm-up or settlement')
                else:
                    if not isinstance(values, list) or len(values) != 30:
                        raise ValueError('expected 30 orders')
                    if not all(not isinstance(x, bool) and isinstance(x, (int, float)) and
                               math.isfinite(float(x)) for x in values):
                        raise ValueError('orders must be finite numbers')
                    self.orders.append((event['id'], [float(x) for x in values]))
            self.index += 1
            self.waiting = False
            return {'accepted': True}
        except Exception:
            self.failed = True
            raise

    def result(self):
        if self.failed or self.waiting or self.index != len(self.events):
            raise ValueError('incomplete or invalid temporal transcript')
        if [i for i, _ in self.orders] != self.want:
            raise ValueError('missing trusted orders')
        return self.orders
