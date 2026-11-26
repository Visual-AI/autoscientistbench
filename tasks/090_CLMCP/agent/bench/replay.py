"""Trusted one-observation/one-submission state machine."""
from stream_env import make_episode

class Replay:
    def __init__(self, group):
        self.group = group
        self.episode = make_episode(group['seed'], group['x'])
        self.record = None
        self.failed = False
    def __call__(self, method, args, kwargs):
        try:
            if self.failed or kwargs or self.record is not None:
                raise ValueError('invalid or closed temporal transcript')
            if method == 'observe' and not args:
                return self.episode.observe()
            if method == 'submit' and len(args) == 1:
                if isinstance(args[0], bool) or not isinstance(args[0], (int,float)):
                    raise ValueError('estimate must be numeric')
                self.episode.submit(args[0])
                return {'accepted': True}
            if method == 'finish' and not args:
                self.record = self.episode.finish()
                return self.record
            raise ValueError('unknown temporal request')
        except Exception:
            self.failed = True
            raise
    def rows(self):
        if self.failed or self.record is None:
            raise ValueError('incomplete or invalid temporal transcript')
        return list(zip(self.group['ids'],self.record['estimates']))
