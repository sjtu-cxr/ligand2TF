"""Final gate protocol over independently reconstructed fold-local evidence."""
from dataclasses import asdict
import numpy as np
import torch
from src.gate_training import (
    CandidateGateConfig, active_backbone_rank_data, fit_candidate_gate,
    frozen_candidate_gate_grid, grouped_oof_predictions, mean_rank_ensemble,
    predict_candidate_gate, ranking_metrics, select_candidate_gate_config,
)
from src.candidate_gated_top10 import CandidateGatedCorrection
from src.unified_bscd_residual import FEATURE_NAMES
from src.release_protocol import PROTOCOL


def arrays_for(inputs):
    return {key:getattr(inputs,key) for key in ('features','backbone','labels','eligible_mask')}


def evaluate_scores(inputs,scores):
    return ranking_metrics(scores,candidate_hashes=inputs.candidate_hashes,
                           relevant_hashes=inputs.relevant_hashes,eligible_mask=inputs.eligible_mask,
                           query_ids=inputs.query_ids)


def score_gate(inputs,payload,*,beta,device='cpu'):
    if (payload['split'],payload['fold'])!=(inputs.split,inputs.fold) or payload['beta']!=beta:
        raise ValueError('Checkpoint/bundle identity or transfer beta mismatch')
    if payload['feature_names']!=list(FEATURE_NAMES) or payload['training_seeds']!=PROTOCOL['seeds']:
        raise ValueError('Checkpoint feature order or seed mismatch')
    if payload['selected_config'] is None:
        if payload['state_dicts']:raise ValueError('Fallback checkpoint must contain no learned heads')
        return active_backbone_rank_data(inputs.backbone,inputs.eligible_mask,inputs.candidate_hashes)[0]
    if len(payload['state_dicts'])!=3:raise ValueError('Exactly three heads required')
    config=CandidateGateConfig(**payload['selected_config']);device=torch.device(device)
    predictions=[]
    for state in payload['state_dicts']:
        model=CandidateGatedCorrection(feature_count=32,width=config.width,gamma=config.gamma).to(device)
        model.load_state_dict(state,strict=True)
        predictions.append(predict_candidate_gate(model,arrays_for(inputs),
                           indexes=np.arange(len(inputs.query_ids)),candidate_hashes=inputs.candidate_hashes,device=device))
    return mean_rank_ensemble(predictions,inputs.candidate_hashes,inputs.eligible_mask)


def fit_gate(validation,*,beta,epochs=30,device='cpu'):
    if validation.stage!='validation':
        raise ValueError('Gate selection/fitting requires a validation bundle, never outer-test labels')
    if not validation.labels.any(axis=1).all():raise ValueError('Every training query requires a positive')
    if epochs<=0:raise ValueError('epochs must be positive')
    arrays=arrays_for(validation);device=torch.device(device)
    base=active_backbone_rank_data(validation.backbone,validation.eligible_mask,validation.candidate_hashes)[0]
    baseline,_=evaluate_scores(validation,base);candidate_metrics={};proofs={}
    for config in frozen_candidate_gate_grid():
        def train_predict(train,holdout,seed):
            model=fit_candidate_gate(arrays,indexes=train,query_ids=validation.query_ids,
                    candidate_hashes=validation.candidate_hashes,config=config,training_seed=seed,epochs=epochs,device=device)
            return predict_candidate_gate(model,arrays,indexes=holdout,candidate_hashes=validation.candidate_hashes,device=device)
        oof,proof=grouped_oof_predictions(query_ids=validation.query_ids,split_units=validation.split_units,
                      candidate_hashes=validation.candidate_hashes,training_seeds=PROTOCOL['seeds'],train_predict=train_predict)
        candidate_metrics[config]=evaluate_scores(validation,oof)[0];proofs[config.config_id]=proof
    selected,metrics=select_candidate_gate_config(candidate_metrics,baseline)
    states=[]
    if selected is not None:
        for seed in PROTOCOL['seeds']:
            model=fit_candidate_gate(arrays,indexes=np.arange(len(validation.query_ids)),query_ids=validation.query_ids,
                    candidate_hashes=validation.candidate_hashes,config=selected,training_seed=seed,epochs=epochs,device=device)
            states.append({name:value.detach().cpu() for name,value in model.state_dict().items()})
    payload={'split':validation.split,'fold':validation.fold,'beta':beta,
             'selected_config':None if selected is None else asdict(selected),
             'feature_names':list(FEATURE_NAMES),'training_seeds':PROTOCOL['seeds'],'state_dicts':states,
             'epochs':epochs,'protocol_training':epochs==PROTOCOL['gate_epochs']}
    report={'baseline':asdict(baseline),'selected':asdict(metrics),
            'candidates':{c.config_id:asdict(m) for c,m in candidate_metrics.items()},'oof_proofs':proofs}
    return payload,report
