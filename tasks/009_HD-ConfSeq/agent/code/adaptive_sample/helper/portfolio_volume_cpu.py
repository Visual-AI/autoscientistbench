"""Balance exact Monte Carlo work using disjoint sections of each RNG stream."""
import numpy as np
from joblib import Parallel, delayed
from adaptive_sample.helper.volume import VolumeCalculator


def _count_block(repeat, predicate, points):
    # Compatible with any point-array tasks serialized before a boundary upgrade.
    return repeat, int(np.count_nonzero(predicate.evaluate_many(points)))


def _count_stream_block(repeat, predicate, state, start, size, D, lower, upper):
    bitgen=np.random.PCG64(0)
    bitgen.state=state
    bitgen.advance(start*D)
    points=np.random.Generator(bitgen).random(size=(size,D))
    points=points*(upper-lower)+lower
    VolumeCalculator._sanity_check_points_in_bbx(points,lower,upper,False)
    return repeat,int(np.count_nonzero(predicate.evaluate_many(points)))


def parallel_portfolio_volume(h, predicate, D, n_samples, num_repeats, lower, upper):
    base_seed=h.get('volume_mc_seed')
    domain_volume=VolumeCalculator.safe_positive_product(upper-lower)
    jobs=[]
    for repeat in range(num_repeats):
        rng=np.random.default_rng(None if base_seed is None else int(base_seed)+repeat)
        assert isinstance(rng.bit_generator,np.random.PCG64), 'Pinned NumPy PCG64 is required.'
        state=rng.bit_generator.state
        VolumeCalculator._sanity_check_points_not_in_bbx(predicate,lower,upper,False,D)
        block_size=(n_samples+3)//4
        for start in range(0,n_samples,block_size):
            jobs.append((repeat,state,start,min(block_size,n_samples-start)))
    print(f'Exact portfolio Monte Carlo: {num_repeats} x {n_samples} samples, '
          f'D={D}, {len(jobs)} RNG blocks on {h.get("volume_workers", 4)} workers',flush=True)
    counts=np.zeros(num_repeats,dtype=np.int64)
    for repeat,count in Parallel(n_jobs=int(h.get('volume_workers', 4)),batch_size=1)(
            delayed(_count_stream_block)(repeat,predicate,state,start,size,D,lower,upper)
            for repeat,state,start,size in jobs):
        counts[repeat]+=count
    volumes=[float(count/n_samples)*domain_volume for count in counts]
    return float(np.mean(volumes)),float(np.std(volumes))
