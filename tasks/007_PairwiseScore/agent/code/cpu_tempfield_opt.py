"""CPU inference implementation. Original tempfield.py remains available.

The factorization exp((a-b)/s) = exp(a/s)/exp(b/s) eliminates
per-pair exponentials. Every reference and original clipped weight is retained.
Float64 factors/sums avoid overflow and cancellation; output remains float32.
"""
import ctypes
import os
import numpy as np
import torch
from tempfield import TemperingField as OriginalTemperingField

class TemperingField(OriginalTemperingField):
    def invalidate_cache(self):
        self._reference_cache = None

    def _cached_reference(self):
        key = (self.X.data_ptr(), self.X._version, self.log_p_w_X.data_ptr(),
               self.log_p_w_X._version, self.s,
               tuple(p._version for p in self.r_model.parameters()))
        if getattr(self, '_reference_cache', None) is None or self._reference_cache[0] != key:
            with torch.no_grad():
                ref = self.r_model.logp(self.X).flatten().contiguous()
            w = (1.0 / (self.log_p_w_X.exp() + 1e-8)).flatten()
            w = torch.clamp(w, min=w.quantile(0.01), max=w.quantile(0.9)).contiguous()
            self._reference_cache = (key, ref, w)
        return self._reference_cache[1:]

    def tau(self, x, clamp=True):
        if x.dtype != torch.float32 or x.device.type != 'cpu':
            return super().tau(x, clamp)
        self.r_model.eval()
        ref, weights = self._cached_reference()
        with torch.no_grad():
            query = self.r_model.logp(x).flatten().contiguous()
        lib = getattr(self, '_kernel', None)
        if lib is None:
            lib = ctypes.CDLL(os.environ['REPRO_TAU_LIBRARY'])
            for name in ('tau_factored', 'tau_fused'):
                getattr(lib, name).argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int64]*2 + [ctypes.c_float, ctypes.c_int]
            self._kernel = lib
        out = torch.empty(len(query), dtype=query.dtype)
        # Common centering leaves ratios unchanged; restrict range for safe doubles.
        lo = min(query.min().item(), ref.min().item())
        hi = max(query.max().item(), ref.max().item())
        if np.isfinite(lo+hi) and (hi-lo)/self.s < 600 and not os.getenv('REPRO_TAU_FUSED'):
            offset = (hi+lo)/2
            a = np.exp((query.numpy().astype(np.float64)-offset)/self.s)
            b = np.exp((ref.numpy().astype(np.float64)-offset)/self.s)
            lib.tau_factored(a.ctypes.data, b.ctypes.data, weights.data_ptr(), out.data_ptr(),
                             len(query), len(ref), self.s, torch.get_num_threads())
        else:
            lib.tau_fused(query.data_ptr(), ref.data_ptr(), weights.data_ptr(), out.data_ptr(),
                          len(query), len(ref), self.s, torch.get_num_threads())
        out = torch.nan_to_num(out, nan=1.0, posinf=self.s*1e6, neginf=1.0).clamp(min=1.0)
        if clamp:
            out = out.clamp(max=self.tau_quantile)
        return out
