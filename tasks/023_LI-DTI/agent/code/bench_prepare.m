function [R, folds, sideInfoRow_no_diag, sideInfoCol_no_diag] = bench_prepare(path)
load(path);
chemSim = chemSim - eye(size(chemSim));
protSeqSimN = protSeqSim./max(protSeqSim);
protSeqSimN = protSeqSimN - eye(size(protSeqSimN));
sideEffect = sideEffect(:,sum(sideEffect,1) > 0);
%% 
% Compute the cosine similarity for Side Effects, drug-disease associations,DDI, 
% target-Disease associations and PPI

sideEffectSim = 1 - squareform(pdist(sideEffect,'cosine'));
drugDiseaseSim = 1 - squareform(pdist(drugDisease,'cosine'));
protDiseaseSim = 1 - squareform(pdist(protDisease,'cosine'));
PPISim = 1 - squareform(pdist(PPI,'cosine'));
PPISim(isnan(PPISim)) = 0;
drugDrugSim = 1 - squareform(pdist(drugDrug,'cosine'));
drugDrugSim(isnan(drugDrugSim)) = 0;
%% 
% Remove the Main Diagonal

sideEffectSim = sideEffectSim - eye(size(sideEffectSim));
drugDiseaseSim = drugDiseaseSim - eye(size(drugDiseaseSim));
protDiseaseSim = protDiseaseSim - eye(size(protDiseaseSim));
PPISim = PPISim - eye(size(PPISim));
drugDrugSim = drugDrugSim - eye(size(drugDrugSim));
optimal_threshold = [0.6 ... Chemical similarity
                     0.5 ... Side effect similarity
                     0.7 ... Drug drug interaction
                     0 ...   Drug disease associations
                     0.3 ... Sequence similarity
                     0.4 ... Protein protein interaction similarity
                     0];     % Disease similarity

%% 
% Remove values below threshold value

chemSim(chemSim <optimal_threshold(1)) = 0;
sideEffectSim(sideEffectSim < optimal_threshold(2)) = 0;
drugDrugSim(drugDrugSim < optimal_threshold(3)) = 0;
drugDiseaseSim(drugDiseaseSim<optimal_threshold(4)) = 0;
protSeqSimN(protSeqSimN< optimal_threshold(5)) = 0;
PPISim(PPISim<optimal_threshold(6)) = 0;
protDiseaseSim(protDiseaseSim<optimal_threshold(7)) = 0;
%% 
% Group similarities 

sideInfoRow_no_diag = {chemSim,sideEffectSim,drugDrugSim,drugDiseaseSim};
sideInfoCol_no_diag = {protSeqSimN,PPISim,protDiseaseSim};

end
