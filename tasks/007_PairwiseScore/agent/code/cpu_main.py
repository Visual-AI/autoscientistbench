import os
import time
import copy

import torch
import numpy as np
from tqdm import tqdm

from diffusion import EDM
import phema #EDM type EMA
from likelihood import loglik
from model import EDMToyModel
from tempfield import TemperingField
from transformation2uniform import transform, inversetransform, transform_dataset


#run e.g. by command: python main.py --config-name=experiment2d --multirun exp.target=onemoon,ring exp.seed=1,2
method="SCORE-TAU(X)"   #score-based method with full tempring field tau(x)
#method="SCORE-TAUSTAR" #score-based method with constant tempring field


def run(cfg, output_folder, original=False, small_reference=None):

    from cpu_run_support import Stage, finish_run
    stage = Stage(output_folder)
    stage.start("setup")
    if not original:
        from cpu_tempfield_opt import TemperingField
    else:
        from tempfield import TemperingField
 

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

    # Offline ranked observations and known proposal transform; no target access.
    target_name = "observed"
    D = cfg.exp.d
    with np.load(os.environ['BENCH_INSTANCE'], allow_pickle=False) as obs:
        dataset = torch.from_numpy(obs['dataset']).to(device)
        input_sigma_data = torch.as_tensor(obs['sigma_data'])
        bounds = obs['bounds'].tolist()
        means = torch.from_numpy(obs['means']) if obs['means'].size else None
        covariance_matrix = torch.from_numpy(obs['covariance_matrix']) if obs['covariance_matrix'].size else None
    n = dataset.shape[2]
    ranking = cfg.data.k > 2
    normalize = None
    target_std_transformed = dataset.permute(0, 2, 1).reshape(-1, D).std(dim=0).mean()

    def minibatch(dataset,batch_size,ranking=False):
        indices = torch.randperm(n)[:batch_size]
        batch = dataset[:,:,indices]
        return batch

    #Initialize diffusion model
    if target_name not in ["llm_prior"]:
        if cfg.data.transform2unitcube:
            sigma_data = target_std_transformed #e.g. 0.28 in twogaussian 4D
        else:
            sigma_data = input_sigma_data
    else:
        sigma_data = 0.333
    if cfg.method.name == "EDM":
        if cfg.model.name == "EDMtoy":
            model = EDMToyModel(in_dim=2*D,num_layers=cfg.model.num_layers,hidden_dim=cfg.model.hidden_dim,sigma_data=sigma_data,use_temp=False).to(device).train().requires_grad_(True)
        if cfg.model.name == "Dit":
            raise NotImplementedError
        sigma_min = cfg.method.sigma_min
        sigma_max = cfg.method.sigma_max
        L = cfg.method.langevin_L
        #Noise schedule corresponding to the EDM time step discretization would be a natural option, but cosine noise schedule gives better results (and mode mixing?)
        def cosine_noise_schedule(sigma_min, sigma_max, L):
            t = torch.linspace(0, 1, steps=L)
            sigmas = sigma_min + 0.5*(sigma_max - sigma_min)*(1 + torch.cos(t * np.pi))
            return sigmas
        ALD_sigmas = cosine_noise_schedule(sigma_min, sigma_max, L)
        diffusion = EDM(2*D,sigma_min,sigma_max,ALD_sigmas=ALD_sigmas,L=L)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        ema_std = cfg.method.ema_std

    #Initialize optimizer
    loss_hist = np.array([])
    batch_size = cfg.optimization.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimization.lr_ref, betas=(0.9, 0.99))

    #Convenience sampler method
    def sample_diffusion(ema,diffusion=diffusion,temp=0):
        with torch.no_grad():
            #diffusionsample = diffusion.sample_stochastic(ema,cfg.plot.nsamples,joint=0,model_temp=temp).cpu().numpy()
            diffusionsample = diffusion.sample_deterministic(ema,cfg.plot.nsamples,joint=0,temp=temp).cpu().numpy()
            diffusionsample = diffusionsample[:,:D] #keep only winner samples
            if target_name in ["llm_prior"]:
                diffusionsample = normalize(diffusionsample,reverse=True)
            return diffusionsample
    
    #Initial sampling
    if target_name not in ["llm_prior"]:
        targetsample = None  # target draws are held in frozen labels
    initial_diffusionsample = sample_diffusion(ema)[:,:D] if target_name != "llm_prior" else (None,None)
    if cfg.data.transform2unitcube:
        initial_diffusionsample = inversetransform(initial_diffusionsample, cfg, bounds, means, covariance_matrix, torch.tensor(cfg.exp.mixture_probs))

    stage.end()
    stage.start("score_training")
    #Main training loop: learn the joint distribution p(x_winner,x_loser)
    start = time.time()
    time_training = start
    for it in tqdm(range(cfg.optimization.max_iter),disable=not cfg.plot.progressbar_show):
        
        #Use EDM type learning rate decay schedule?
        if cfg.optimization.lr_iter is not None:
            optimizer.param_groups[0]['lr'] = cfg.optimization.lr_ref / np.sqrt(max(it / cfg.optimization.lr_iter, 1)) #Below Eq. (15) in Karras et al, (NeurIPS2024)
        optimizer.zero_grad()

        batch = minibatch(dataset,batch_size) #(k, D, batch_size)
        winners =  batch.clone()[0,:,:].transpose(0,1) #(batch_size,D)
        losers =  batch.clone()[1,:,:].transpose(0,1) #(batch_size,D)

        if cfg.method.name == "EDM":
        
            x0 = torch.cat([winners, losers], dim=1)
            if cfg.method.sigma_dist=="uniform":
                log_sigma_t = torch.rand((batch_size,)) * (torch.tensor(sigma_max).log() - torch.tensor(sigma_min).log()) + torch.tensor(sigma_min).log()
            if cfg.method.sigma_dist=="lognormal":
                log_sigma_t = cfg.method.P_mean + cfg.method.P_std*torch.randn((batch_size,)) #Karras et al., 2024, Appendix C
            if random.random() < cfg.method.phi: #phi of time: specify first sigmas based on geometric series
                idx = torch.randperm(L)[:batch_size] if L >= batch_size else torch.randint(0, L, (batch_size,))
                log_sigma_t = ALD_sigmas.log()[idx]
            sigma_t = torch.clamp(log_sigma_t.exp().reshape(-1, 1), min=sigma_min, max=sigma_max)
            tilde_x = x0 + sigma_t * torch.randn_like(x0)
            temp = torch.zeros(batch_size,1) #no tempering
            if random.randint(0, 1) == 0: #Coin flip: train marginal p(x_winner)
                joint = torch.zeros(batch_size,1) #0=consider marginal p(x_winner)
                score = diffusion.score_function(tilde_x[:,:D], x0[:,:D], sigma_t)
                tilde_x[:, D:] = sigma_t * torch.randn_like(tilde_x[:, D:]) #explicit masking with noise
                predicted_score = model(tilde_x, sigma_t, joint, temp)[:,:D]
            else: #Coin flip: train joint p(x_winner,x_loser)
                joint = torch.ones(batch_size,1) #1=consider joint p(x_winner,x_loser)
                score = diffusion.score_function(tilde_x, x0, sigma_t)
                predicted_score = model(tilde_x, sigma_t, joint, temp)
            
            #Score-matching loss and weighting
            weight = (sigma_t**2) #Karras et al., 2024, Eq. (15)
            #weight = (sigma_t ** 2 + sigma_data ** 2) / (sigma_t * sigma_data) ** 2 #Karras et al., 2022, leads too high weights
            loss = torch.mean(weight * ((predicted_score - score) ** 2))
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                #Update EMA
                beta = phema.power_function_beta(std=ema_std, t_next=it+1, t_delta=1)
                for p_net, p_ema in zip(model.parameters(), ema.parameters()):
                    p_ema.lerp_(p_net.detach(), 1 - beta)

                loss_hist = np.append(loss_hist, loss.to('cpu').detach().numpy())
            
        if (it + 1) % cfg.plot.show_iter == 0:
            end = time.time()
            mean_loss = np.mean(loss_hist[-cfg.plot.show_iter:])  # Compute mean of last cfg.plot.show_iter iters
            print(f"{it+1}: loss {mean_loss.item():0.7f} time {(end - start):0.2f}")
            start = end

    time_training = time_training - time.time()
    
    stage.end()
    stage.start("tempering_training")
    tempfield = TemperingField(D=D, diffusion=diffusion, MWD=ema, minibatch=minibatch,
        loglik=loglik, s=cfg.exp.s, MWD_sample_size=small_reference or 2000*D,
        r_hidden_dim=cfg.model.hidden_dim)
    tempfield.train_r(dataset, cfg.optimization.tau_batch_size, cfg.optimization.tau_lr,
                      cfg.optimization.tau_max_iter, cfg.optimization.tau_max_weight_decay)
    stage.end()
    stage.start("reference_monte_carlo")
    tempfield.sample_MWD()
    tempfield.tau_quantile = torch.quantile(tempfield.tau(tempfield.X, clamp=False), tempfield.quantile)
    tempfield.tau_mean = tempfield.tau(tempfield.X, clamp=True).mean()
    stage.end()
    return finish_run(cfg, output_folder, stage, model, ema, diffusion, tempfield,
                      locals().get("targetsample"), normalize, bounds,
                      locals().get("means"), locals().get("covariance_matrix"), loss_hist)
