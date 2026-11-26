args = argv();
assert(numel(args) == 3);
bench_setup();
load(args{1});
queries = dlmread(args{2}, ',', 1, 0);
fid = fopen(args{3}, 'w');
assert(fid > 0);
fprintf(fid, 'run,fold,sample,row,col,pred\n');
for i = 1:rows(folds)
  for j = 1:columns(folds)
    selected = queries(:,1)==i & queries(:,2)==j;
    if ~any(selected), continue; end
    local_R = R;
    test_fold = folds{i,j};
    idx = sub2ind(size(R),test_fold(:,1),test_fold(:,2));
    local_R(idx) = 0;
    Rr = normaliseY(local_R,1);
    cov_no_diag = (Rr*Rr' - (Rr*Rr').*eye(size(Rr*Rr',1)));
    SIR_tmp = [{cov_no_diag}, sideInfoRow_no_diag(:)'];
    Rc = normaliseY(local_R,0);
    cov_no_diag = (Rc'*Rc - (Rc'*Rc).*eye(size(Rc'*Rc,1)));
    SIC_tmp = [{cov_no_diag}, sideInfoCol_no_diag(:)'];
    Res = get_predictions(weights{i,j},local_R,SIR_tmp,SIC_tmp);
    q = queries(selected,:);
    values = Res(sub2ind(size(Res),q(:,4),q(:,5)));
    for n = 1:rows(q)
      fprintf(fid,'%d,%d,%d,%d,%d,%.17g\n',q(n,1),q(n,2),q(n,3),q(n,4),q(n,5),values(n));
    end
  end
end
fclose(fid);
