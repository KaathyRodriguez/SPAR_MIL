"""
SPAR-MIL training main program.
For training the dual-scale vision-language multiple instance learning model
for whole-slide image classification.
"""
from __future__ import print_function
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch

from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text_prompts_csv(path: str):
    """Load text prompts from a CSV into a List[str].

    Supports both single-column CSVs (one prompt per row) and multi-column CSVs
    (joins non-empty cells per row into one prompt string).
    """
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


# ============================================================
# Task configuration
# ============================================================
TASK_CONFIG = {
    'task_pcr_vs_nonpcr': {
        'csv_path': 'H:/dataset/SYSUFH_ESCC_Dataset/label.csv',
        'label_dict': {'non-pCR': 0, 'pCR': 1},
        'n_classes': 2,
    },
    'task_tcga_brca_subtyping': {
        'csv_path': 'H:/dataset/TCGA-BRCA/label.csv',
        'label_dict': {'ILC': 0, 'IDC': 1},
        'n_classes': 2,
    },
    'task_camelyon16_subtyping': {
        'csv_path': 'H:/dataset/CAMELYON16/label.csv',
        'label_dict': {'Normal': 0, 'Tumor': 1},
        'n_classes': 2,
    },
}


# ============================================================
# Generic training settings
# ============================================================
parser = argparse.ArgumentParser(description='Configurations for WSI Training')

# Data
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--data_folder_s', type=str, default=None, help='low-scale feature folder under data directory')
parser.add_argument('--data_folder_l', type=str, default=None, help='high-scale feature folder under data directory')
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--text_prompt_path', type=str, default=None)
parser.add_argument('--prototype_prompt', type=str, default=None,
                    help='semantic prototype prompt csv for SPAR-MIL')
parser.add_argument('--prototype_prompt_csv', type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument('--prototype_number', type=int, default=8)
parser.add_argument('--text_backend', type=str, choices=['open_clip', 'conch'], default='open_clip',
                    help='text encoder backend')
parser.add_argument('--conch_model_cfg', type=str, default='conch_ViT-B-16',
                    help='CONCH model config name')
parser.add_argument('--conch_checkpoint_path', type=str, default=None,
                    help='path to the CONCH checkpoint')

# Training
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results/SPAR_MIL',
                    help='results directory (default: ./results/SPAR_MIL)')
parser.add_argument('--log_data', action='store_true', default=False,
                    help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False,
                    help='enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False,
                    help='enable dropout (p=0.25)')
parser.add_argument('--model_type', type=str, choices=['SPAR_MIL'], default='SPAR_MIL',
                    help='type of model')
parser.add_argument('--mode', type=str, choices=['transformer', 'mil'], default='transformer')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False,
                    help='enable weighted sampling')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce')
parser.add_argument('--task', type=str, required=True)

# Model dimensions
parser.add_argument('--input_size', type=int, default=512, help='input feature dimension')
parser.add_argument('--hidden_size', type=int, default=192, help='hidden dimension for attention')

# SPAR-MIL specific hyper-parameters
parser.add_argument('--gnn_num_layers', type=int, default=2,
                    help='number of PrototypeGNN transformer encoder layers')
parser.add_argument('--gnn_num_heads', type=int, default=4,
                    help='number of attention heads in PrototypeGNN')
parser.add_argument('--gnn_ffn_factor', type=int, default=2,
                    help='FFN expansion factor in PrototypeGNN')
parser.add_argument('--gnn_dropout', type=float, default=0.1,
                    help='dropout rate in PrototypeGNN')
parser.add_argument('--router_temperature', type=float, default=1.0,
                    help='temperature for the scale router')
parser.add_argument('--router_mode', type=str,
                    choices=['gumbel', 'softmax', 'static', 'mean'],
                    default='gumbel',
                    help='scale-router mode; gumbel for training, softmax for inference')
parser.add_argument('--static_w_low', type=float, default=0.5,
                    help='fixed low-scale weight used when router_mode=static')
parser.add_argument('--topk_ratio', type=float, default=0.7,
                    help='Top-k ratio for prototype aggregation')
parser.add_argument('--loss_weight_contrastive', type=float, default=0.1,
                    help='weight for cross-scale prototype contrastive loss')
parser.add_argument('--loss_weight_diversity', type=float, default=0.05,
                    help='weight for prototype diversity loss')
parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                    help='temperature for cross-scale prototype contrastive loss')
parser.add_argument('--margin_loss_weight', type=float, default=0.1,
                    help='weight for prototype margin loss')
parser.add_argument('--margin_value', type=float, default=0.3,
                    help='margin value for prototype margin loss')

args = parser.parse_args()

# Resolve prototype prompt aliases
if args.prototype_prompt is None:
    args.prototype_prompt = args.prototype_prompt_csv
args.prototype_prompt_csv = args.prototype_prompt

# Load class text prompts
if args.text_prompt_path is not None:
    args.text_prompt = load_text_prompts_csv(args.text_prompt_path)
else:
    args.text_prompt = None


def seed_torch(seed=7):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

settings = {
    'num_splits': args.k,
    'k_start': args.k_start,
    'k_end': args.k_end,
    'task': args.task,
    'max_epochs': args.max_epochs,
    'results_dir': args.results_dir,
    'lr': args.lr,
    'experiment': args.exp_code,
    'label_frac': args.label_frac,
    'seed': args.seed,
    'model_type': args.model_type,
    'mode': args.mode,
    'use_drop_out': args.drop_out,
    'weighted_sample': args.weighted_sample,
    'opt': args.opt,
}

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(args.label_frac * 100)))

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


def main(args):
    """
    Main training function.
    Runs k-fold cross-validation training for SPAR-MIL.
    """
    task_name = args.task.strip().lower()
    if task_name not in TASK_CONFIG:
        raise NotImplementedError(f"Unsupported task: {args.task}")

    task_cfg = TASK_CONFIG[task_name]
    args.n_classes = task_cfg['n_classes']

    # SPAR-MIL is a dual-scale model: load both low- and high-scale features.
    dataset = Generic_MIL_Dataset(
        csv_path=task_cfg['csv_path'],
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict=task_cfg['label_dict'],
        patient_strat=False,
        ignore=[],
    )

    # Determine fold range
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []

    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
        )
        datasets = (train_dataset, val_dataset, test_dataset)

        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    # Aggregate results
    final_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'test_acc': all_test_acc,
        'test_f1': all_test_f1,
    })
    result_df = pd.DataFrame({
        'metric': ['mean', 'var'],
        'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
        'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)],
        'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
    })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.results_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)


if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")