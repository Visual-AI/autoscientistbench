"""Generate reviewable Octave protocols from unchanged upstream scripts.

No prior outputs are inputs. Seed and Octave portability match the earlier
reproduction. 'dense=False' retains the original regression cell operator.
"""
from pathlib import Path
import re

CODE = Path(__file__).resolve().parents[1]
KAPPAS = [1, 5, 10, 20]
METHODS = ['ScaledGD', 'GD']


def generate(base, transform, out, *, index=None, parameter=None,
             smoke=False, dense=True, records=None):
    out = Path(out).resolve()
    s = (CODE / (base + '.m')).read_text().split('h1 = figure(1);')[0]
    s = s.replace("rng('shuffle');", "rng(20261022, 'twister');")
    # Octave has a signal thread. Sequential-BLAS small jobs use <=3 FFT
    # threads; regression uses a single FFT thread and parallel BLAS.
    fft_threads = "1" if base == 'TR' else "min(3,str2double(getenv('REPRO_WORKER_THREADS')))"
    s = s.replace("addpath('tproduct');", f"addpath('{CODE / 'tproduct'}'); addpath('{CODE / 'cpu_opt'}');\nfftw('threads', {fft_threads});")
    s = s.replace("norm(X - X_star, 'fro')", 'tensor_fro(X - X_star)').replace("norm(X_star, 'fro')", 'tensor_fro(X_star)')
    s = s.replace('randsample(n*n*n3, round(alpha*n*n*n3), false)', 'randperm(n*n*n3, round(alpha*n*n*n3))')
    if transform == 'DCT':
        s = s.replace('transform.L = @fft; transform.l = n3; transform.inverseL = @ifft;',
            "q=(0:n3-1)'; j=0:n3-1; C=sqrt(2/n3)*cos(pi*(j+0.5).*q/n3); C(1,:)=C(1,:)/sqrt(2); transform.L=C; transform.l=1; transform.inverseL=C';")
    if smoke:
        s = re.sub(r'^n = \d+;', 'n = 12;', s, flags=re.M)
        s = re.sub(r'^n3 = \d+;', 'n3 = 6;', s, flags=re.M)
        s = re.sub(r'^r = \d+;', 'r = 2;', s, flags=re.M)
        records = records or 8
    if parameter:
        param, value = parameter
        s = s.replace(f'{param} = 0.5;', f'{param} = {value:.1f};')
        s = s.replace('thresh_up = 1e3;', 'thresh_up = 1e2;')
        records = records or 301
        for method in METHODS:
            marker = f'errors_{method}(i_kappa, t) = error;'
            s = s.replace(marker, marker + '\n        if t == T; break; end')
    if records:
        s = s.replace('T = 1000;', f'T = {records};')
    if dense and base == 'TR':
        s = s.replace('As = cell(m, 1);\nfor k = 1:m\n    As{k} = randn(n, n, n3)/sqrt(m);\nend',
                      'As=zeros(n*n*n3,m);\nfor k=1:m; As(:,k)=reshape(randn(n,n,n3),[],1)/sqrt(m); end')
        s = s.replace("y = zeros(m, 1);\n    for k = 1:m\n        y(k) = As{k}(:)'*X_star(:);\n    end", "y=As'*X_star(:);")
        s = s.replace('Y = zeros(n, n, n3);\n    for k = 1:m\n        Y = Y + y(k)*As{k};\n    end', 'Y=reshape(As*y,n,n,n3);')
        s = s.replace("Z = zeros(n, n, n3);\n        for k = 1:m\n            Z = Z + (As{k}(:)'*X(:) - y(k))*As{k};\n        end", "Z=reshape(As*(As'*X(:)-y),n,n,n3);")
        assert 'As{' not in s
    if index is not None:
        # Preserve preceding draws when separating condition numbers.
        s = s.replace('    %% Spectral initialization',
            f'    if i_kappa ~= {index}; continue; end\n    repro_shared_clock=tic;\n    %% Spectral initialization')
        s = s.replace('    %% Scaled GD', '    shared_init_seconds=toc(repro_shared_clock);\n    repro_method_clock=tic;\n    %% Scaled GD')
        s = s.replace('    %% GD', '    repro_method_clock=tic;\n    %% GD')
        s = s.replace('errors_GD = zeros(length(kappa_list), T);',
            'errors_GD = zeros(length(kappa_list), T);\ntimes_ScaledGD=zeros(length(kappa_list),T);\ntimes_GD=zeros(length(kappa_list),T);')
        for method in METHODS:
            marker = f'errors_{method}(i_kappa, t) = error;'
            s = s.replace(marker, marker + f'\n        times_{method}(i_kappa,t)=toc(repro_method_clock);')
    for method in METHODS:
        row = f'({index},:)' if index else ''
        s += f"\ncsvwrite('{out}/errors_{method}.csv', errors_{method}{row});\n"
        if index:
            s += f"csvwrite('{out}/times_{method}.csv', times_{method}{row});\n"
    if index:
        s += f"csvwrite('{out}/initialization.csv', shared_init_seconds);\n"
    return s
