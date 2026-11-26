"""Opt-in CPU implementation; upstream code remains the reference path."""
import torch


def hybrid_loss(pred_dist, true_dist, mask_2d, alpha=0.7, lambda_=0.1):
    # The upstream MAE is computed then discarded. Preserve MSE/KL arithmetic.
    return alpha * _original.kl_structure_loss(pred_dist, true_dist, mask_2d, lambda_) + (1-alpha) * _original.weighted_mse_loss(pred_dist, true_dist, mask_2d)


def install(module):
    global _original
    _original = module
    module.hybrid_loss = hybrid_loss


class DenseGather(torch.autograd.Function):
    """One-dimensional gather with linear scatter-add instead of embedding sort."""
    @staticmethod
    def forward(ctx, weight, indices):
        ctx.save_for_backward(indices)
        ctx.shape = weight.shape
        return weight[indices]

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        result = grad.new_zeros(ctx.shape)
        result.view(-1).scatter_add_(0, indices.reshape(-1), grad.reshape(-1))
        return result, None


def dense_embedding_forward(self, indices):
    return DenseGather.apply(self.weight, indices)


def cached_getitem(self, idx):
    if self.augment:
        return self._uncached_getitem(idx)
    if not hasattr(self, '_cpu_samples'):
        self._cpu_samples = {}
    if idx not in self._cpu_samples:
        self._cpu_samples[idx] = self._uncached_getitem(idx)
    return self._cpu_samples[idx]


_install_loss = install

def install(module):
    _install_loss(module)
    dataset = module.ChromosomeDataset
    if not hasattr(dataset, '_uncached_getitem'):
        dataset._uncached_getitem = dataset.__getitem__
        dataset.__getitem__ = cached_getitem
    # Only the two scalar edge embeddings benefit from this dense backward.
    from layers import GaussianLayer
    if not hasattr(GaussianLayer, '_original_init'):
        GaussianLayer._original_init = GaussianLayer.__init__
        def init(self, *args, **kwargs):
            import types
            self._original_init(*args, **kwargs)
            self.mul.forward = types.MethodType(dense_embedding_forward, self.mul)
            self.bias.forward = types.MethodType(dense_embedding_forward, self.bias)
        GaussianLayer.__init__ = init


def edge_bias(dist, edge, mul, bias, means, stds, proj_weight, proj_bias):
    x = DenseGather.apply(mul, edge) * dist.unsqueeze(-1) + DenseGather.apply(bias, edge)
    mean = means.float().view(-1)
    std = stds.float().view(-1).abs() + 1e-5
    features = torch.exp(-0.5 * (((x.float() - mean) / std) ** 2)) / ((2 * 3.14159) ** .5 * std)
    return torch.nn.functional.linear(features, proj_weight, proj_bias).permute(0,3,1,2)


def optimized_edge_bias(self, dist, edge):
    d = self.distance_embedding
    return _compiled_edge(dist, edge, d.mul.weight, d.bias.weight, d.means.weight, d.stds.weight, self.bias_proj.weight, self.bias_proj.bias)


def enable_compile(module):
    global _compiled_edge
    import torch._inductor.config
    torch._inductor.config.compile_threads = 1
    _compiled_edge = torch.compile(edge_bias, dynamic=True)
    module.InertialTransformer_Modeling.cpu_edge_bias = optimized_edge_bias


_rank_context = {}


def rank_collate(samples):
    """Reuse sorted per-chromosome ranks; merge only distinct target values.

    The global context retains only the currently yielded batch. Standard CPU,
    num_workers=0 training consumes that batch before requesting the next one.
    """
    from torch.utils.data import default_collate
    records=[]
    for s in samples:
        if '_cpu_ranks' not in s:
            vals,inv,counts=torch.unique(s['true_dist'][s['contact_valid_mask']],sorted=True,return_inverse=True,return_counts=True)
            s['_cpu_ranks']=(vals,inv,counts)
        records.append(s['_cpu_ranks'])
    batch=default_collate([{k:v for k,v in s.items() if k!='_cpu_ranks'} for s in samples])
    _rank_context.clear()
    true=batch['true_dist']
    for i,(vals,inv,counts) in enumerate(records):
        _rank_context[(true[i].data_ptr(),tuple(true[i].shape))]=(len(vals),inv,counts)
    vals=torch.unique(torch.cat([r[0] for r in records]),sorted=True)
    inverses=[torch.searchsorted(vals,v)[inv] for v,inv,c in records]
    inverse=torch.cat(inverses)
    counts=torch.zeros(len(vals),dtype=torch.long)
    for v,inv,c in records:counts.scatter_add_(0,torch.searchsorted(vals,v),c)
    _rank_context[(true.data_ptr(),tuple(true.shape))]=(len(vals),inverse,counts)
    return batch


def ranked_mse(pred_dist,true_dist,mask_2d,eps=1e-8):
    record=_rank_context.get((true_dist.data_ptr(),tuple(true_dist.shape)))
    if record is None:return _original_mse(pred_dist,true_dist,mask_2d,eps)
    n,inv,counts=record
    valid_true=true_dist[~mask_2d];valid_pred=pred_dist[~mask_2d]
    assert inv.numel()==valid_true.numel(), 'rank metadata requires original contact mask'
    if n<2:return torch.mean((valid_pred-valid_true)**2)
    weights=torch.arange(n,0,-1,device=true_dist.device).float()/(n*(n+1)/2+eps)
    errors=torch.zeros(n,device=true_dist.device)
    errors.scatter_add_(0,inv,(valid_pred-valid_true)**2)
    return torch.sum(weights*(errors/(counts.float()+eps)))


def enable_rank_cache(module):
    global _original_mse
    if module.weighted_mse_loss is not ranked_mse:
        _original_mse=module.weighted_mse_loss
    module.weighted_mse_loss=ranked_mse
