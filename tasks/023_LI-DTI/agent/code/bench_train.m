args = argv();
assert(numel(args) == 2);
bench_setup();
[R, folds, sideInfoRow_no_diag, sideInfoCol_no_diag] = bench_prepare(args{1});
rng(str2double(getenv('BENCH_SEED')));
bound = 0;
options = optimset('Algorithm','sqp','GradObj','on','MaxIter',10000,'MaxFunEvals',10000,'Display','off');
lambda1 = 1;
lambda2 = 0.1;
[nruns,nfolds] = size(folds);
weights = cell(nruns,nfolds);
for i = 1:nruns
    for j = 1:nfolds
            local_R = R;
            %remove testing DTIs from training
            test_fold = folds{i,j};
            idx = sub2ind(size(R),test_fold(:,1),test_fold(:,2));
            local_R(idx) = 0;
            % Matlab Implementation
            %SIC_tmp = sideInfoCol_no_diag;
            %SIR_tmp = sideInfoRow_no_diag;

            SIR_tmp = sideInfoRow_no_diag;
            SIC_tmp = sideInfoCol_no_diag;

            %initialise weights for optimisation
            initRange = [ones(1,length(SIC_tmp)+length(SIR_tmp)+3)*0.05]';
            params = rand(length(initRange),1) .* initRange;

            %set lower bound of weights for fmincon
            low_bound = bound * ones(length(params),1);
            
            %normalise DTI matrix by row
            Rr = normaliseY(local_R,1); % by row

            %compute row covariance similarity
            cov_no_diag = (Rr*Rr' - (Rr*Rr').*eye(size(Rr*Rr',1)));

            %add covariance to drug similarites
            %SIR_tmp = [cov_no_diag SIR_tmp(:)'];
            SIR_tmp = [{cov_no_diag}, SIR_tmp(:)'];
                        
            %normalise DTI matrix by column
            Rc = normaliseY(local_R,0); % by column   
            cov_no_diag = (Rc'*Rc - (Rc'*Rc) .* eye(size(Rc'*Rc,1)));
           
            %add covariance to drug similarites
            %SIC_tmp = [cov_no_diag SIC_tmp(:)'];
            SIC_tmp = [{cov_no_diag}, SIC_tmp(:)'];

            %get optimal values for gamma, alphas and betas with our method
            [local_X,~] =  fmincon(@(t)(our_method_gradients(t,local_R,Rr,Rc,SIC_tmp,SIR_tmp,lambda1,lambda2)),params,[],[],[],[],low_bound,[],[],options);
            
            weights{i,j} = local_X;
            fprintf('LI-DTI fitted run %d fold %d on %dx%d matrix\n',i,j,rows(R),columns(R));
    end
end
save('-mat7-binary', args{2}, 'R','folds','sideInfoRow_no_diag','sideInfoCol_no_diag','weights');
