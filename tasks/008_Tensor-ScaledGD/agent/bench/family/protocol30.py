"""Figure 1 generator. Regression batches independent RHS against identical As.

All tensor reconstruction and factor update blocks are extracted verbatim from
cpu_opt/protocol.py's portable upstream source. There is no RNG in the updates.
"""
from pathlib import Path
import sys
import re
P = Path(__file__).resolve().parent
sys.path.insert(0, str(P/'cpu_opt'))
from protocol import generate, METHODS, KAPPAS, CODE


def small(base, transform, out, index, records=None, smoke=False):
    s = generate(base, transform, out, index=index, records=records, smoke=smoke)
    # Remove Figure 2 timing only. No arithmetic in the estimator is changed.
    s = '\n'.join(l for l in s.splitlines() if not any(x in l for x in
        ['repro_shared_clock', 'repro_method_clock', 'times_ScaledGD', 'times_GD', 'shared_init_seconds']))
    return s+'\n'


def regression(out, records=1000, smoke=False, mode='batch'):
    out = Path(out)
    s = generate('TR','DFT',out,records=records,smoke=smoke)
    prefix, cases = s.split('for i_kappa = 1:length(kappa_list)', 1)
    case_init, methods = cases.split('    %% Scaled GD', 1)
    scaled, gd = methods.split('    %% GD', 1)
    factor_init, scaled_loop = scaled.split('    for t = 1:T',1)
    _, gd_loop = gd.split('    for t = 1:T',1)
    reconstruct = scaled_loop.split('        error = ')[0]
    gradient = "Z=reshape(As*(As'*X(:)-y),n,n,n3);"
    update_scaled = scaled_loop.split(gradient)[1].rsplit('    end',1)[0]
    update_gd = gd_loop.split(gradient)[1].split('\n    end\nend')[0]
    # Initial U/V seed and As are identical for DFT and DCT: tsvd consumes no RNG.
    pre_as, _ = prefix.split('As=zeros(',1)
    seed_code = pre_as.split('U_seed = ',1)[1]
    seed_code = 'U_seed = '+seed_code
    dct = "q=(0:n3-1)'; j=0:n3-1; C=sqrt(2/n3)*cos(pi*(j+0.5).*q/n3); C(1,:)=C(1,:)/sqrt(2); transform.L=C; transform.l=1; transform.inverseL=C';"
    preamble = prefix.replace('As=zeros(n*n*n3,m);', 'operator_rng=rng;\nAs=zeros(n*n*n3,m);')
    # Store original operator RNG state; resetting for DCT is explicitly checked.
    body = "batch_clock=tic;\n"+preamble.replace("clearvars;", "")+"\nstates={};\nfor transform_index=1:2\n"
    body += "if transform_index==2\nrng(20261022,'twister');\n"+dct+'\n'+seed_code+"\nassert(isequal(rng,operator_rng));\nend\n"
    body += 'for i_kappa=1:length(kappa_list)\n'+case_init+factor_init
    body += """
for method_index=1:2
    st=struct('L',L,'R',R,'L_plus',L_plus,'R_plus',R_plus,'Xtrans',Xtrans,
              'X_star',X_star,'y',y,'sigma_star',sigma_star,'transform',transform,
              'transform_index',transform_index,'i_kappa',i_kappa,
              'method_index',method_index,'errors',zeros(1,T),'active',true);
    states{end+1}=st;
end
end
end
initialization_s=toc(batch_clock);\noptimization_clock=tic;\nfprintf('INITIALIZED %d trajectories in %.3f s\\n',numel(states),initialization_s); fflush(stdout);
for t=1:T
    ids=[]; XX=[]; YY=[];
    for j=1:numel(states)
        st=states{j};
        if ~st.active; continue; end
        L=st.L; R=st.R; Xtrans=st.Xtrans; transform=st.transform;
"""+reconstruct+"""
        error=tensor_fro(X-st.X_star)/tensor_fro(st.X_star);
        st.errors(t)=error;
        if ~isfinite(error) || error>thresh_up || error<thresh_low
            st.active=false;
        else
            ids(end+1)=j; XX(:,end+1)=X(:); YY(:,end+1)=st.y;
        end
        states{j}=st;
    end
    if isempty(ids); break; end
"""
    if mode=='batch':
        body += "    ZZ=As*(As'*XX-YY);\n"
    elif mode=='separate':
        body += "    ZZ=zeros(size(XX)); for jj=1:size(XX,2); ZZ(:,jj)=As*(As'*XX(:,jj)-YY(:,jj)); end\n"
    else:
        raise ValueError(mode)
    body += """
    for jj=1:numel(ids)
        j=ids(jj); st=states{j};
        L=st.L; R=st.R; L_plus=st.L_plus; R_plus=st.R_plus;
        transform=st.transform; sigma_star=st.sigma_star;
        Z=reshape(ZZ(:,jj),n,n,n3);
        if st.method_index==1
"""+update_scaled+'\nelse\n'+update_gd+"""
        end
        st.L=L; st.R=R; st.L_plus=L_plus; st.R_plus=R_plus;
        states{j}=st;
    end
end
"""
    body += f"optimization_s=toc(optimization_clock); csvwrite('{out}/TR_stage_seconds.csv',[initialization_s,optimization_s]); out='{out}'; transforms={{'DFT','DCT'}}; methods={{'ScaledGD','GD'}};\n"
    body += """
for j=1:numel(states)
    st=states{j};
    dest=sprintf('%s/TR_%s_kappa%d',out,transforms{st.transform_index},kappa_list(st.i_kappa));
    if ~exist(dest,'dir'); mkdir(dest); end
    csvwrite(sprintf('%s/errors_%s.csv',dest,methods{st.method_index}),st.errors);
end
"""
    return body
