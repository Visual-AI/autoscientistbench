function [AUPR] = AUPR_perfcurve(labels, scores)
    %% Area under the precision-recall curve with the same convention as MATLAB's
    %% perfcurve(labels, scores, 1, 'xCrit','reca','yCrit','prec'):
    %%   * the curve starts at the highest-scoring sample (no artificial (recall=0,
    %%     precision=1) point is prepended),
    %%   * every distinct score threshold is a point,
    %%   * the area is the trapezoidal integral of precision over recall.
    %% The Octave port shipped with the paper (code (Octave)/AUCDownSample.m) prepends the
    %% (0,1) point, which inflates AUPR by ~1%.  On the authors' own shipped Octave
    %% predictions this function returns 0.7562 +- 0.0056, i.e. exactly the value reported
    %% in the paper (Fig. 2a), while AUCDownSample.m returns 0.7643 +- 0.0056.
    labels = labels(:) > 0;
    scores = scores(:);
    [scores_sorted, order] = sort(scores, 'descend');
    labels_sorted = labels(order);
    P = sum(labels_sorted == 1);
    if P == 0
        AUPR = NaN;
        return;
    end
    TP = cumsum(labels_sorted == 1);
    FP = cumsum(labels_sorted == 0);
    % one point per distinct threshold (last index of each tied group, as in perfcurve)
    [~, last_idx] = unique(scores_sorted, 'last');
    last_idx = last_idx(end:-1:1);   % unique() sorts by value; we need decreasing score
    TP = TP(last_idx);
    FP = FP(last_idx);
    recall = TP / P;
    precision = TP ./ (TP + FP);
    AUPR = trapz(recall, precision);
end
