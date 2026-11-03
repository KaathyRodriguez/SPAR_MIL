"""
模型评估主程序
用于评估训练好的ViLa-MIL模型在测试集上的性能
"""
from __future__ import print_function

import numpy as np
import argparse
import torch
import os
import pandas as pd
from utils.utils import *
from math import floor
from datasets.dataset_generic import Generic_MIL_Dataset
from utils.eval_utils import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text_prompts_csv(path: str):
    """Load text prompts from a CSV into a `List[str]`."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 1:
        prompts = []
        for x in df.iloc[:, 0].tolist():
            if pd.isna(x):
                prompts.append("")
            else:
                prompts.append(str(x).strip())
        return prompts

    prompts = []
    for row in df.itertuples(index=False, name=None):
        parts = [str(x).strip() for x in row if pd.notna(x) and str(x).strip()]
        prompt = " ".join(parts).strip()
        prompts.append(prompt)
    return prompts

# Evaluation settings 
parser = argparse.ArgumentParser(description='Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--data_folder_s', type=str, default=None, help='dir under data directory' )
parser.add_argument('--data_folder_l', type=str, default=None, help='dir under data directory' )
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL','ViLa_MIL_Optimized','ViLa_MIL_Semantic_Prototype_Guided','DSP_ViLa_MIL','DSPG_ViLa_MIL','RAG_DSPNet','ViLa_MIL_Mammoth_Optimized','ViLa_MIL_Mammoth_LinearReplace','ViLa_MIL_Mammoth_DynaMoE','ViLa_MIL_TextGammaBeta','maple','FOCUS','TOP_MIL','clam_sb','clam_mb','maxpooling','meanpooling'], default='ViLa_MIL')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--drop_out', action='store_true', default=False, help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str)
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default=None)
parser.add_argument("--prototype_prompt", type=str, default=None, help="semantic prototype prompt csv for the semantic prototype guided model")
parser.add_argument("--prototype_prompt_csv", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--text_backend", type=str, choices=["open_clip", "conch"], default="open_clip", help="text encoder backend")
parser.add_argument("--conch_model_cfg", type=str, default="conch_ViT-B-16", help="CONCH model config name")
parser.add_argument("--conch_checkpoint_path", type=str, default=None, help="path to the CONCH checkpoint")
parser.add_argument('--prototype_number', type=int, default=8)
parser.add_argument('--prototype_contrastive_weight', type=float, default=0.05)
parser.add_argument('--prototype_contrastive_temperature', type=float, default=0.1)
parser.add_argument('--aux_bag_loss_weight', type=float, default=0.2)
parser.add_argument('--rag_llm_backend', type=str, choices=['auto', 'ollama', 'transformers'], default='ollama')
parser.add_argument('--rag_llm_model_name', type=str, default=None)
parser.add_argument('--rag_prototype_llm_model_name', type=str, default=None)
parser.add_argument('--rag_diagnosis_llm_model_name', type=str, default=None)
parser.add_argument('--rag_ollama_base_url', type=str, default='http://127.0.0.1:11434')
parser.add_argument('--rag_ollama_timeout', type=float, default=120.0)
parser.add_argument('--rag_ollama_num_predict', type=int, default=256)
parser.add_argument('--rag_prototype_use_llm', action='store_true', default=False)
parser.add_argument('--rag_diagnosis_use_llm', action='store_true', default=False)
parser.add_argument('--rag_feature_dim', type=int, default=512)
parser.add_argument('--rag_hidden_dim', type=int, default=256)
parser.add_argument('--rag_navigation_max_steps', type=int, default=4)
parser.add_argument('--rag_filter_target_keep_ratio', type=float, default=0.6)
parser.add_argument('--rag_filter_min_keep', type=int, default=1)
parser.add_argument('--rag_prototype_max_count', type=int, default=8)
parser.add_argument('--rag_prototype_min_count', type=int, default=4)
parser.add_argument('--rag_nav_entropy_weight', type=float, default=0.0)
parser.add_argument('--rag_filter_entropy_weight', type=float, default=0.0)
parser.add_argument('--rag_filter_aux_weight', type=float, default=0.0)
parser.add_argument('--rag_bag_dropout', type=float, default=0.0)


# MAPLE 评估相关参数（需与训练时保持一致）
parser.add_argument('--maple_base_model', type=str, default='CLIP', choices=['CLIP', 'PLIP'])
parser.add_argument('--maple_text_path', type=str, default=None)
parser.add_argument('--maple_weight', type=float, default=0.5)
parser.add_argument('--maple_pos_ratio', type=float, default=0.25)
parser.add_argument('--maple_neg_ratio', type=float, default=0.25)
parser.add_argument('--maple_attr_n_ctx', type=int, default=4)
parser.add_argument('--maple_bag_n_ctx', type=int, default=4)
parser.add_argument('--maple_attr_edge_topk', type=int, default=3)
parser.add_argument('--maple_all_ctx_trainable', action='store_true', default=True)
parser.add_argument('--maple_csc', action='store_true', default=True)
parser.add_argument('--maple_p_drop_out', type=float, default=0.0)
parser.add_argument('--maple_p_bag_drop_out', type=float, default=0.0)

# MAMMOTH 评估相关参数（需与训练时保持一致）
parser.add_argument('--mammoth_auto_rank', action='store_true', default=False)
parser.add_argument('--mammoth_lora_rank', type=int, default=4)
parser.add_argument('--mammoth_rank_mode', type=str, choices=['formula', 'search'], default='formula')
parser.add_argument('--mammoth_num_experts', type=int, default=16)
parser.add_argument('--mammoth_num_heads', type=int, default=8)
parser.add_argument('--mammoth_share_lora_weights', action='store_true', default=True)
parser.add_argument('--mammoth_use_layernorm', action='store_true', default=True)
parser.add_argument('--mammoth_routing_temperature', type=float, default=1.0)
parser.add_argument('--mammoth_routing_balance_weight', type=float, default=1.0)
parser.add_argument('--mammoth_routing_entropy_weight', type=float, default=1.0)
parser.add_argument('--mammoth_routing_aux_weight', type=float, default=0.01)
parser.add_argument('--mammoth_low_num_slots', type=int, default=8)
parser.add_argument('--mammoth_low_slot_dim', type=int, default=256)
parser.add_argument('--mammoth_low_lora_rank', type=int, default=4)
parser.add_argument('--mammoth_low_dropout', type=float, default=0.1)
parser.add_argument('--mammoth_low_slot_dropout', type=float, default=0.0)
parser.add_argument('--mammoth_low_rank_mode', type=str, choices=['formula', 'search'], default=None)
parser.add_argument('--mammoth_low_rank_candidates', type=str, default='4,8,16,32')
parser.add_argument('--mammoth_low_num_experts', type=int, default=16)
parser.add_argument('--mammoth_low_num_heads', type=int, default=8)
parser.add_argument('--mammoth_low_keep_slots', action='store_true', default=False)
parser.add_argument('--mammoth_low_residual_dropout', type=float, default=0.1)
parser.add_argument('--mammoth_low_routing_temperature', type=float, default=1.0)
parser.add_argument('--mammoth_low_routing_balance_weight', type=float, default=1.0)
parser.add_argument('--mammoth_low_routing_entropy_weight', type=float, default=1.0)
parser.add_argument('--mammoth_high_num_slots', type=int, default=8)
parser.add_argument('--mammoth_high_slot_dim', type=int, default=256)
parser.add_argument('--mammoth_high_lora_rank', type=int, default=4)
parser.add_argument('--mammoth_high_dropout', type=float, default=0.1)
parser.add_argument('--mammoth_high_slot_dropout', type=float, default=0.0)
parser.add_argument('--mammoth_high_rank_mode', type=str, choices=['formula', 'search'], default=None)
parser.add_argument('--mammoth_high_rank_candidates', type=str, default='4,8,16,32')
parser.add_argument('--mammoth_high_num_experts', type=int, default=16)
parser.add_argument('--mammoth_high_num_heads', type=int, default=8)
parser.add_argument('--mammoth_high_keep_slots', action='store_true', default=False)
parser.add_argument('--mammoth_high_residual_dropout', type=float, default=0.1)
parser.add_argument('--mammoth_high_routing_temperature', type=float, default=1.0)
parser.add_argument('--mammoth_high_routing_balance_weight', type=float, default=1.0)
parser.add_argument('--mammoth_high_routing_entropy_weight', type=float, default=1.0)
# MacNet Grassmann model related parameters
parser.add_argument('--macnet_num_clusters', type=int, default=3)
parser.add_argument('--macnet_cluster_iters', type=int, default=5)
parser.add_argument('--macnet_cluster_temperature', type=float, default=1.0)
parser.add_argument('--macnet_gfk_dim', type=int, default=20)
parser.add_argument('--macnet_target_ratio', type=float, default=0.2)

args = parser.parse_args()
if args.model_type == 'RAG_DSPNet':
    args.mode = 'transformer'
# 只有 ViLa 系列和 FOCUS 需要文本提示
if args.prototype_prompt is None:
    args.prototype_prompt = args.prototype_prompt_csv
args.prototype_prompt_csv = args.prototype_prompt
if args.model_type in ['ViLa_MIL', 'ViLa_MIL_Semantic_Prototype_Guided', 'DSP_ViLa_MIL', 'DSPG_ViLa_MIL', 'ViLa_MIL_Mammoth_Optimized', 'ViLa_MIL_Mammoth_LinearReplace', 'ViLa_MIL_Mammoth_DynaMoE', 'ViLa_MIL_TextGammaBeta', 'FOCUS'] and args.text_prompt_path is not None:
    args.text_prompt = load_text_prompts_csv(args.text_prompt_path)
else:
    args.text_prompt = None
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'mode': args.mode,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_RCC_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2},
                                  patient_strat= False,
                                  ignore=[])

elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_Lung_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'LUAD':0, 'LUSC':1},
                                  patient_strat= False,
                                  ignore=[])
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_f1 = []
    all_true = []
    all_pred = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, test_f1, df, _ = eval(args.mode, split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        all_f1.append(test_f1)

        all_true += list(df['Y'])
        all_pred += list(df['Y_hat'])

        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc,'test_acc': all_acc, 'test_f1': all_f1})
    result_df = pd.DataFrame({'metric': ['mean', 'var'],
                              'test_auc': [np.mean(all_auc), np.std(all_auc)],
                              'test_acc': [np.mean(all_acc), np.std(all_acc)],
                              'test_f1': [np.mean(all_f1), np.std(all_f1)],
                              })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.save_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.save_dir, save_name))
