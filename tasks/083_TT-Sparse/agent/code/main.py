"""Fixed reproduced TT-Sparse configuration with file-based benchmark entry points."""
import argparse, dataclasses, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tt_sparse import TabularEncoder,TTSparseModel,train,prune,explain,predict_rules
sys.path.insert(0,os.environ['BENCH_BENCH_DIR'])
from certificate import export_encoder,unpack,write_prediction,FEATURES

def development(data):
    tr=pd.read_csv(data/'train/train.csv',float_precision='round_trip')
    va=pd.read_csv(data/'val/input.csv',float_precision='round_trip')
    labels=pd.read_csv(data/'val/labels.csv',float_precision='round_trip').set_index('id')
    va['target']=['tested_positive' if labels.loc[i,'y'] else 'tested_negative' for i in va.id]
    order=json.loads((data/'train/development_order.json').read_text())
    full=pd.concat([tr,va]).set_index('id').loc[order]
    index={id_:i for i,id_ in enumerate(order)}
    return full.reset_index(drop=True), (np.array([index[i] for i in tr.id]),np.array([index[i] for i in va.id]))

def build_model(enc):
    return TTSparseModel(input_size=enc.n_ltt_features,num_nodes=30,num_classes=1,n_bits=4,tau=.05,task_type='binary',skip_size=enc.n_skip_features,dropout=.1)

def fit(out):
    out.mkdir(parents=True,exist_ok=True)
    frame,indices=development(Path(os.environ['BENCH_DATA_DIR']))
    training_frame=frame.iloc[indices[0]]
    outer=int(os.environ.get('BENCH_SEED','1'))
    cards=[]
    for replica in range(5):
        seed=(5*(outer-1)+replica)%(2**32)
        torch.manual_seed(seed);np.random.seed(seed)
        enc=TabularEncoder(target='target',task_type='binary',num_bits=9)
        # Validation can select checkpoints/pruning decisions, but must not fit
        # normalization, thresholds, vocabularies, or rule coverage.
        enc.fit(training_frame)
        data=enc.transform(frame);data['split_indices']=indices
        model=build_model(enc)
        train(model,data,epochs=200,batch_size=1024,lr=.005,val_split=.2,patience=25,device='cpu',seed=seed,verbose=False)
        prune(model,data,max_drop_pct=2.,finetune_epochs=30,max_iterations=80,max_fanin=16,device='cpu',seed=seed,verbose=False)
        rules=explain(model,enc,training_data=training_frame)
        cards.append({'rules':rules.rules,'encoder':export_encoder(enc)})
        torch.save({'state_dict':model.state_dict(),'encoder':enc,'seed':seed},out/f'model_{replica}.pt')
        print(f'replicate {replica} seed {seed} complexity {rules.complexity}',flush=True)
    (out/'rules.json').write_text(json.dumps({'format':'tt-sparse-rules-v1','models':cards})+'\n')

def predict(model,input_,output):
    frame=pd.read_csv(input_,float_precision='round_trip')
    card=json.loads((model/'rules.json').read_text())
    models=unpack(card)
    # Actual prediction calls the authors' predict_rules function.
    pred=np.column_stack([predict_rules(r,frame[FEATURES],enc) for r,enc in models])
    write_prediction(output,frame.id.tolist(),pred,card)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict']);p.add_argument('--out',type=Path);p.add_argument('--model',type=Path);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    # Two threads per sequential instance, bounded by the allocated budget, as in the reproduction.
    torch.set_num_threads(min(2,int(os.environ.get('BENCH_CPUS','8'))))
    if a.stage=='train':fit(a.out)
    else:predict(a.model,a.input,a.output)
