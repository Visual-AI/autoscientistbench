"""Streaming form of the released cross-asset filter/MPC backtest loop."""
import argparse
import json
from pathlib import Path

import numpy as np
from bench_main import PaperMPCFloored, ORDERS, L
from src.backtest import _apply_no_leverage_clip
from src.filter.cokf import CrossAssetCollaborativeKF
from src.filter.single_ckf import irw_state_space
from src.mpc import compute_paper_sigma_l


class Policy:
    def __init__(self, model_dir):
        self.models = json.loads((Path(model_dir) / 'model.json').read_text())['panels']
        self.started = False

    def start(self, panel):
        if self.started:
            raise ValueError('each worker handles one independent panel')
        params = self.models[panel]
        sigma_v = np.asarray(params['sigma_v'])
        self.m = len(sigma_v)
        self.cokf = CrossAssetCollaborativeKF(
            M=self.m, orders=ORDERS,
            sigma_v_per_order=tuple(sigma_v.copy() for _ in ORDERS),
            sigma_w=np.asarray(params['sigma_w']), rho=np.asarray(params['rho']), horizon=L)
        self.f_per_order = [irw_state_space(r)[0] for r in ORDERS]
        self.mpcs = [PaperMPCFloored(horizon=L, lam=params['lam'], b=.5,
                                    sigma_l_mult=params['c']) for _ in range(self.m)]
        self.position = np.zeros(self.m)
        self.cash = np.full(self.m, 1000.)
        self.pending = np.zeros(self.m)
        self.busted = np.zeros(self.m, dtype=bool)
        self.next_step = 0
        self.started = True
        return {'started': panel}

    def handle(self, message):
        if message['op'] == 'start':
            return self.start(message['panel'])
        if message['op'] != 'step' or not self.started:
            raise ValueError('start a panel before sending observations')
        k = int(message['step'])
        if k != self.next_step:
            raise ValueError('observations must be sequential')
        prices = np.asarray(message['prices'], dtype=float)
        if prices.shape != (self.m,) or not np.all(np.isfinite(prices)):
            raise ValueError('price dimensions')
        self.next_step += 1
        if k == 0:
            self.cokf.init(prices)
            if message['active']:
                raise ValueError('first observation is initialization only')
            return {'id': message['id'], 'orders': None}

        # Original order: settle, liquidate, update filters, solve MPC, clip.
        self.position += self.pending
        self.cash -= self.pending * prices
        last_applied = self.pending.copy()
        value = self.position * prices + self.cash
        for i in range(self.m):
            if not self.busted[i] and value[i] < 200.:
                self.cash[i] += self.position[i] * prices[i]
                self.position[i] = 0.
                value[i] = self.cash[i]
                self.busted[i] = True
        out = self.cokf.step(prices)
        if message['active']:
            l_step = self.cokf.aggregate_l_step()
            new_pending = np.zeros(self.m)
            for i in range(self.m):
                if self.busted[i]:
                    continue
                sigma_l = compute_paper_sigma_l(out['weights'][i], self.f_per_order, L)
                plan = self.mpcs[i].solve(
                    u_current=float(last_applied[i]), price_pred=l_step[i],
                    sigma_l_paper=sigma_l, capital=max(float(value[i]), 1e-3))
                new_pending[i] = float(plan[0]) if len(plan) else 0.
            self.pending = _apply_no_leverage_clip(new_pending, self.position, prices, value)
            return {'id': message['id'], 'orders': self.pending.tolist()}
        self.pending = np.zeros(self.m)
        return {'id': message['id'], 'orders': None}


def main():
    from private_rpc import call
    parser = argparse.ArgumentParser()
    for name in ('model', 'input', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    policy = Policy(args.model)
    while True:
        event = call('next')
        if event['op'] == 'done':
            break
        call('submit', policy.handle(event))
    Path(args.output).write_text(json.dumps({'complete': True}) + '\n')


if __name__ == '__main__':
    main()
