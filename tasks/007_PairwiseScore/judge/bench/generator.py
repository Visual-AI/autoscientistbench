"""Frozen released generator: exact draw order, including target sampler bugs.
Only the operator materializes instances. Prediction must use supplied observations.
Do not call this generator to synthesize extra data or use target identities.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch, numpy as np
from omegaconf import OmegaConf
from dgp_target import set_up_problem
from dgp_misc import convert_to_ranking, convert_to_ranking_and_change_k
from dgp_transformation2uniform import transform_dataset
from dgp_model import EDMToyModel
CONFIGS=[('experiment2d','onemoon'),('experiment2d','twomoons'),('experiment2d','ring'),
 ('experiment4d_uniform','onegaussian'),('experiment4d_uniform','mixturegaussians'),
 ('experiment6d','stargaussian'),('experiment10d','mixturegaussians'),('experiment16d','onegaussian')]
def config(index,seed):
 name,target=CONFIGS[index]
 cfg=OmegaConf.load(Path(__file__).resolve().parent/'configs'/(name+'.yaml'))
 cfg.exp.target=target;cfg.exp.seed=int(seed)
 return cfg

def generate(cfg):
    ### Device and Precision ###
    torch.set_default_dtype(torch.float64 if cfg.device.precision_double else torch.float32)
    #enable_cuda = True
    #device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    #enable_mps = False #Do not enable as MPS framework when precision is float64
    #device = torch.device('mps' if torch.backends.mps.is_available() and enable_mps else 'cpu')
    device = torch.device(cfg.device.device)

    ### Random seeds ###
    import random
    torch.manual_seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)
    random.seed(cfg.exp.seed)

    ### Target belief density###
    target_name = cfg.exp.target
    D = cfg.exp.d
    target, bounds, uniform, D, normalize = set_up_problem(target_name,D,tightbounds = True if cfg.exp.q_dist=="uniform" else False)

    ### Data generation part 1 ###
    if target_name not in ["llm_prior"]:
        target_sample = target.sample(10000)
        target_mean = target_sample.mean(dim=0)
        target_std = target_sample.std(dim=0)
        if cfg.exp.q_dist=="mixture_gaussian":
            if target_name=="mixturegaussians":
                means, covariance_matrix = target.get_means_and_covariances()
                max_std = torch.sqrt(torch.stack([torch.diag(c) for c in covariance_matrix]).max())
                covariance_matrix = torch.stack([torch.eye(D) * max_std**2 for _ in range(len(means))]) #diagonal Gaussian for each component to make problem harder
            else:
                #Assume always two mixture compenents but with mixture probs (e.g. 0.5 + 0.5 = 1.0) one can make this practically single component when using same means
                means = torch.stack([target_mean, target_mean])
                covariance_matrix = target_std * torch.eye(D).unsqueeze(0).repeat(2, 1, 1) #That is, lambda std is twice higher than the target std
            #How much sampling distribution (lambda) variance is higher than the target?
            var_scale = 3
            covariance_matrix = var_scale * covariance_matrix
        elif cfg.exp.q_dist=="uniform":
            means=None
            covariance_matrix=None

    def sample_alternatives(n,k=2,distribution="uniform",means=None,covariance_matrix=None):
        if distribution=="uniform":
            return uniform.sample(torch.tensor([k*n])).to(device)
        elif distribution=="target":
            return target.sample(k*n).to(device)
        elif distribution=="mixture_gaussian":
            component_distribution = torch.distributions.MultivariateNormal(means, covariance_matrix)
            mixing_probs = torch.tensor(cfg.exp.mixture_probs)
            mixing_distribution = torch.distributions.Categorical(mixing_probs)
            target_gaussian = torch.distributions.MixtureSameFamily(mixing_distribution, component_distribution)
            return target_gaussian.sample((k*n,))
        
    def expert_feedback_ranking(alternatives,rum_noise_dist,s):
        k = alternatives.shape[0]
        if rum_noise_dist == "exponential":
            noise = torch.distributions.Exponential(s).sample((k,)).to(device)
        elif rum_noise_dist == "gumbel":
            noise = torch.distributions.Gumbel(torch.tensor([0.0]), torch.tensor([s])).sample((k,)).to(device).squeeze(1)
        elif rum_noise_dist == "normal":
            noise = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([s])).sample((k,)).to(device).squeeze(1)
        logprobs = target.log_prob(alternatives).to(device) + noise
        _, ranking_inds = torch.sort(logprobs, descending=True)
        return ranking_inds.view(k).to(device)

    def generate_dataset_ranking(n,k,distribution="uniform",means=None,covariance_matrix=None,rum_noise_dist="exponential",s=1.0):
        X = sample_alternatives(1,k,distribution,means,covariance_matrix)
        Y = expert_feedback_ranking(X,rum_noise_dist,s).view(1,k)
        X = X.unsqueeze(2) #add new dimension, which indicates sample index
        if n > 1:
            for i in range(0,n-1):
                alternatives = sample_alternatives(1,k,distribution,means,covariance_matrix)
                X = torch.cat((X,alternatives.unsqueeze(2)),2)
                Y = torch.cat((Y,expert_feedback_ranking(alternatives,rum_noise_dist,s).view(1,k)),0)
        Xdata = convert_to_ranking(X.numpy(),Y.numpy())
        #return X,Y #X.shape = (k,D,N) = (alternatives,space dimensions, number of rankings)
        return torch.from_numpy(Xdata).view(k,-1,n)

    ### Data generation part 2 ###
    n = cfg.data.n
    if target_name in ["llm_prior"]:
        Xdata1 = np.load("data/llm_prior/california_data_set_1_21-04-2024_dataX.npy") #207 rankings
        Ydata1 = np.load("data/llm_prior/california_data_set_1_21-04-2024_dataY.npy", allow_pickle=True)
        Xdata2 = np.load("data/llm_prior/california_data_set_2_22-04-2024_dataX.npy") #13 rankings
        Ydata2 = np.load("data/llm_prior/california_data_set_2_22-04-2024_dataY.npy", allow_pickle=True)
        if cfg.data.k==5: #Basic scenario, matches to k that was used in creating the dataset
            Xdata = convert_to_ranking(np.concatenate((Xdata1,Xdata2), axis=2),np.concatenate((Ydata1,Ydata2), axis=0))
        else:
            Xdata = convert_to_ranking_and_change_k(np.concatenate((Xdata1,Xdata2), axis=2),np.concatenate((Ydata1,Ydata2), axis=0),k=cfg.data.k)
        Xdata = normalize(torch.from_numpy(Xdata)).to(torch.float64 if cfg.device.precision_double else torch.float32)
        variable_names = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
        n = 220
        dataset = Xdata
        ranking = True
    else:
        ranking = True if cfg.data.k > 2 else False
        k = cfg.data.k
        dataset = generate_dataset_ranking(n,k,distribution=cfg.exp.q_dist,means=means,covariance_matrix=covariance_matrix,rum_noise_dist=cfg.exp.rum_noise_dist,s=getattr(cfg.exp, "s_true", cfg.exp.s)) #s=cfg.exp.s)
        raw_dataset = dataset.clone()
        if cfg.data.transform2unitcube:
            dataset = transform_dataset(dataset, cfg, bounds, means, covariance_matrix, torch.tensor(cfg.exp.mixture_probs))
            target_std_transformed = dataset.permute(0, 2, 1).reshape(-1, D).std(dim=0).mean()

    # Advance the exact released model-initialization draws before its evaluation target.
    before_model = torch.get_rng_state().clone()
    sigma_data=target_std_transformed if cfg.data.transform2unitcube else target_std.mean()
    model=EDMToyModel(in_dim=2*D,num_layers=cfg.model.num_layers,hidden_dim=cfg.model.hidden_dim,
                      sigma_data=sigma_data,use_temp=False).to(device).train().requires_grad_(True)
    target_eval=target.sample(cfg.plot.nsamples)
    return dict(dataset=dataset,raw_dataset=raw_dataset,bounds=np.asarray(bounds),means=means,
                covariance_matrix=covariance_matrix,sigma_data=sigma_data,target=target_eval,
                before_model=before_model,after_target=torch.get_rng_state().clone(),
                model_state=model.state_dict())
