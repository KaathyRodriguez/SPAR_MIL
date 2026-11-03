# SPAR-MIL

**SPAR-MIL: Semantic Prototype-guided Adaptive Dual-scale Routing Multiple Instance Learning for Whole Slide Image Classification**, *Bioinformatics*, 2026 (Original Paper).

Xiwen Lin

School of Computer Science, South China Normal University, 55 Zhongshan Avenue West, 510631, Tianhe District, Guangzhou, Guangdong, China

**Abstract**: The gigapixel scale and hierarchical structure of whole-slide images (WSIs) have made multiple instance learning (MIL) the mainstream paradigm for weakly supervised pathology diagnosis. However, most existing MIL methods rely on bag-level annotations, while diagnostic information in WSIs is distributed across multiple scales; the sparsity of slide-level annotations further increases the difficulty of robust feature aggregation. To address these issues, we propose **SPAR-MIL**, a dual-scale MIL framework based on semantic prototypes and dynamic routing. Specifically, semantic prototypes and a prototype relation graph (**PrototypeGNN**) are introduced at both low and high scales to guide visual feature aggregation. A context-guided text enhancer enriches the semantic space of class text prompts. A **Per-Sample dynamic scale router** adaptively fuses dual-scale representations. Experiments on TCGA-BRCA, CAMELYON16, and TCGA-LUNG under 1-shot, 4-shot, and 16-shot settings show average AUCs of **83.7%**, **77.7%**, and **88.2%** under 16-shot, respectively. Component ablations further analyze the contribution of each design. Code is available at `https://github.com/KaathyRodriguez/SPAR_MIL` (to be updated).

https://image/readme/framework.png

![](image/readme/fig_architecture.png)

**Keywords**: few-shot learning, dual-scale multiple instance learning, semantic prototype, prototype relation graph, dynamic scale routing

## 1. Pre-requisites

Python 3.8+ is recommended. Main dependencies:

text

```
torch, torchvision
h5py, matplotlib, numpy, pandas
opencv-python, pillow
openslide-python, openslide
scikit-learn, scipy
tensorboardx
open_clip        # default text backend
clip             # optional, CLIP tokenizer
CONCH            # optional, pathology vision-language text backend
```



Notes:

- The cross-scale prototype contrastive loss uses `scipy.optimize.linear_sum_assignment` for bipartite matching by default. If `scipy` is unavailable, `model_SPAR_MIL.py` falls back to greedy matching.
- The text backend is controlled by `config.text_backend`. The default is `open_clip` (`ViT-B-32` + `laion2b_s34b_b79k`). The optional `conch` backend requires `config.conch_model_cfg` and `config.conch_checkpoint_path`.

## 2. Download Datasets

Three public WSI datasets are used:

- **TCGA-BRCA**: 908 WSIs, including 713 IDC and 195 ILC cases.
- **CAMELYON16**: 399 WSIs, including 239 normal and 160 tumor cases.
- **TCGA-LUNG**: 365 WSIs, including 177 LUAD and 188 LUSC cases.

TCGA-BRCA and TCGA-LUNG can be downloaded from the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/). CAMELYON16 can be downloaded from the [CAMELYON16 website](https://camelyon16.grand-challenge.org/Download/). For download tools, see [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/).

## 3. Prepare Dataset CSV Files

For each dataset, prepare a `.csv` file under `dataset_csv/` in the following format:

csv

```
case_id,slide_id,label
4398	TCGA-05-4398-01Z-00-DX1	LUAD
5489	TCGA-22-5489-01Z-00-DX1	LUSC
```

Label mappings:

- TCGA-BRCA: `IDC` / `ILC`
- CAMELYON16: `normal` / `tumor`
- TCGA-LUNG: `LUAD` / `LUSC`

> Note: SPAR-MIL targets **binary** WSI classification. `SPAR_MIL_Model` explicitly requires `num_classes == 2`. The class text prompt file must contain `num_classes × 2 = 4` prompts (two classes × low/high resolution); otherwise, model initialization raises a `ValueError`.

## 4. Preprocess Data

### 4.1 Generate Patch Coordinates

Low scale (5×, 2048×2048):

bash

```
python create_patches_fp.py \
	--source SOURCE_FOLDER \ 
	--save_dir SAVE_FOLDER \ 
	--patch_size 2048 
	--step_size 2048 
	--seg 
	--patch 
	--stitch
```

High scale (10×, 1024×1024):

bash

```
python create_patches_fp.py \
	--source SOURCE_FOLDER \ 
	--save_dir SAVE_FOLDER \ 
	--patch_size 1024 
	--step_size 1024 
	--seg 
	--patch 
	--stitch
```

Parameter descriptions:

- `source`: folder containing the original WSIs downloaded in Step 2.
- `slide_name_file`: CSV dataset file prepared in Step 3.
- `save_dir`: output folder for patch coordinates; use separate folders for low and high scales.
- `uuid_name_file`: file storing the mapping between uuid, slide name, and label.
- `patch_size` / `step_size`: use 2048 for low scale and 1024 for high scale.

### 4.2 Crop Patches

bash

```
cd feature_extraction
python patch_generation.py
```

Note: paths between lines 12–17 in `patch_generation.py` must be configured separately for low-scale and high-scale patches. Parameter meanings:

- `slide_folder`: folder containing the original WSIs.
- `root_folder`: same as `save_dir`.
- `define_path_size`: if patches of size 256 are needed at 10×, set this parameter to 1024.
- `save_folder`: output folder for cropped patches.

### 4.3 Extract Patch Features

Extract features for low-scale and high-scale patches separately:

bash

```
python patch_extraction.py \
  --patches_path 'PATCHES_PATH_LOW' \
  --library_path 'LIBRARY_PATH_LOW' \
  --model_name resnet50_trunc \
  --batch_size 64

python patch_extraction.py \
  --patches_path 'PATCHES_PATH_HIGH' \
  --library_path 'LIBRARY_PATH_HIGH' \
  --model_name resnet50_trunc \
  --batch_size 64
```

Parameter descriptions:

- `patches_path`: same as `save_folder` in Step 4.2.
- `library_path`: output folder for patch features.
- `model_name`: `'clip_RN50'` uses CLIP-pretrained ResNet50; `'resnet50_trunc'` uses ImageNet-pretrained ResNet50.
- The paper uses a frozen ResNet50 to extract visual features at both scales.

## 5. Split Datasets

### 5.1 Generate k-fold Splits

bash

```
python create_splits_seq.py \
  --label_frac 1 \
  --k 5 \
  --task 'TASK' \
  --val_frac VAL_FRAC \
  --test_frac TEST_FRAC \
  --dataset DATASET
```

Parameter descriptions:

- `task`: e.g., `task_tcga_brca_subtyping`, `task_camelyon16`, `task_tcga_lung_subtyping`.
- `val_frac` / `test_frac`: validation and test proportions. The paper uses a **3:1:1** training/validation/test split.
- `dataset`: dataset name, e.g., `TCGA_BRCA`, `CAMELYON16`, `TCGA_LUNG`.

### 5.2 Build Few-shot Splits

bash

```
python create_splits_fewshot.py
```

Note: parameters between lines 5–8 in `create_splits_fewshot.py` must be configured:

- `N`: number of randomly selected samples per class in the training set (1, 4, or 16).
- `data_folder`: split folder generated in Step 5.1.
- `all_data`: same as `uuid_name_file`.
- `save_folder`: output folder for few-shot split files.

Few-shot split logic in the paper: for each fold, validation and test sets are independently randomly sampled from the full dataset, and the remainder is used as the training set. **Only the training set** is subsampled with N samples per class; validation and test sets remain unchanged and do not participate in few-shot sampling.

## 6. Prepare Text Prompts

SPAR-MIL uses **two types** of text prompts. Both are generated by a frozen LLM (GPT-5.3 in the paper) using predefined templates.

### 6.1 Class Text Prompts (`text_prompt/`)

Template:

> Write a low-resolution and high-resolution WSI class text prompt for `{class name}`. Please use the following format: "A `{low/high}`-resolution WSI image of `{class name}`, what are its salient visual characteristics?"

For binary classification, there are **4 prompts** in total (two classes × low/high resolution), stored as CSV files, e.g.:

- `text_prompt/TCGA_BRCA_two_scale_text_prompt.csv`
- `text_prompt/CAMELYON16_two_scale_text_prompt.csv`
- `text_prompt/TCGA_LUNG_two_scale_text_prompt.csv`

### 6.2 Semantic Prototype Text Prompts (`prototype_prompt/`)

Template:

> Generate `{low/high}`-resolution visual prototype prompts for `{class name}`. For each `{semantic prototype}`, please use the following format: "A pathology image of `{low/high}`× magnified tissue shows ..."

Semantic prototypes are defined by pathology experts, e.g., "dense tumor cell region", "fibrotic stroma". Prototype counts are dataset-specific:

| Dataset    | Low-scale prototypes | High-scale prototypes |
| :--------- | :------------------- | :-------------------- |
| TCGA-BRCA  | 6                    | 6                     |
| CAMELYON16 | 5                    | 5                     |
| TCGA-LUNG  | 8                    | 8                     |

The prototype prompt CSV must contain `scale` (`low` / `high`) and `prompt` columns, e.g.:

- `prototype_prompt/TCGA_BRCA_Semantic_Prototype_text_prompt.csv`
- `prototype_prompt/CAMELYON16_Semantic_Prototype_text_prompt.csv`
- `prototype_prompt/TCGA_LUNG_Semantic_Prototype_text_prompt.csv`

> Note: `SPAR_MIL_Model` automatically reads the number of `scale == low` and `scale == high` entries from the prototype prompt CSV as `low_proto_num` / `high_proto_num`. Therefore, `--prototype_number` on the command line is mainly for compatibility; the actual prototype counts are determined by the CSV.

## 7. Train the Model

Example for TCGA-BRCA 1-shot:

bash

```
export CUDA_VISIBLE_DEVICES=0
python main.py \
  --seed 1 \
  --drop_out \
  --early_stopping \
  --lr 1e-4 \
  --k 5 \
  --label_frac 1 \
  --bag_loss ce \
  --task 'task_tcga_brca_subtyping' \
  --results_dir 'results' \
  --exp_code 'spar_mil_brca_1shot' \
  --model_type SPAR_MIL \
  --mode transformer \
  --log_data \
  --data_root_dir 'DATA_ROOT_DIR' \
  --data_folder_s 'features_2048' \
  --data_folder_l 'features_1024' \
  --split_dir 'fewshot_splits/TCGA_BRCA/1-shot' \
  --text_prompt_path 'text_prompt/TCGA_BRCA_two_scale_text_prompt.csv' \
  --prototype_prompt 'prototype_prompt/TCGA_BRCA_Semantic_Prototype_text_prompt.csv' \
  --prototype_number 6
```



Parameter descriptions:

- `--model_type`: use `SPAR_MIL`.
- `--task`: `task_tcga_brca_subtyping` / `task_camelyon16` / `task_tcga_lung_subtyping`.
- `--data_folder_s`: low-scale (5×, 2048×2048) feature folder, e.g., `features_2048`.
- `--data_folder_l`: high-scale (10×, 1024×1024) feature folder, e.g., `features_1024`.
- `--split_dir`: same as `save_folder` in Step 5.2.
- `--text_prompt_path`: class text prompt CSV from Step 6.1.
- `--prototype_prompt`: semantic prototype prompt CSV from Step 6.2; `--prototype_prompt_csv` is also accepted.
- `--prototype_number`: recommended to match the prototype prompt CSV (BRCA 6, CAMELYON16 5, LUNG 8).

For CAMELYON16 and TCGA-LUNG, adjust `--task`, `--split_dir`, `--text_prompt_path`, `--prototype_prompt`, and `--prototype_number` accordingly.

### Key Training Hyperparameters 

- `λ1 = 0.1` (`--loss_weight_contrastive`)
- `λ2 = 0.05` (`--loss_weight_diversity`)
- `λ3 = 0.1` (`--margin_loss_weight`)
- `m = 0.3` (`--margin_value`)
- Router temperature `τ = 1.0` (`--router_temperature`)
- Top-k ratio `k/P = 0.7` (`--topk_ratio`)
- Cross-scale contrastive temperature `0.1` (`--contrastive_temperature`)
- GNN layers: default 2; TCGA-BRCA uses 5, CAMELYON16 and TCGA-LUNG use 2 (`--gnn_num_layers`)
- GNN attention heads: default 4 (`--gnn_num_heads`)

### Ablation Switches

`model_SPAR_MIL.py` supports the following ablation options for reproducing Tables 4 and 5 in the paper:

- Disable `--use_norm_anchor`: `w/o Norm Anchor`
- Disable `--use_gated_residual`: `w/o Gated Residual`
- Disable both: `w/o Anchor & Gated`
- `--router_mode softmax`: `Softmax` fusion
- `--router_mode mean`: simple average of low/high scales
- `--router_mode static --static_w_low 0.0`: `High-scale`
- `--router_mode static --static_w_low 1.0`: `Low-scale`
- Disable `--use_prototype_gnn`: `w/o PrototypeGNN`
- Disable `--use_contrastive_loss` / `--use_diversity_loss` / `--use_margin_loss` separately: corresponding auxiliary loss ablations

## 8. Evaluate the Model

bash

```
export CUDA_VISIBLE_DEVICES=0
python eval.py \
  --drop_out \
  --k 5 \
  --k_start 0 \
  --k_end 5 \
  --task 'task_tcga_brca_subtyping' \
  --results_dir 'results' \
  --models_exp_code 'spar_mil_brca_1shot_s1' \
  --save_exp_code 'eval_spar_mil_brca_1shot' \
  --model_type SPAR_MIL \
  --mode transformer \
  --splits_dir 'fewshot_splits/TCGA_BRCA/1-shot' \
  --data_root_dir 'DATA_ROOT_DIR' \
  --data_folder_s 'features_2048' \
  --data_folder_l 'features_1024' \
  --text_prompt_path 'text_prompt/TCGA_BRCA_two_scale_text_prompt.csv' \
  --prototype_prompt 'prototype_prompt/TCGA_BRCA_Semantic_Prototype_text_prompt.csv' \
  --prototype_number 6
```



Parameter descriptions:

- `--models_exp_code`: training directory corresponding to `--exp_code` (with seed suffix).
- `--save_exp_code`: output directory name for evaluation results.
- Other parameters are the same as in training.

## 9. Core Algorithm Details

### 9.1 Dual-scale Features

- **Low scale**: 5× magnification, 2048×2048 patches, capturing global tissue architecture, growth patterns, and stromal reaction.
- **High scale**: 10× magnification, 1024×1024 patches, capturing cell morphology, nuclear features, and cell arrangement.
- Both scales use a frozen ResNet50 for visual feature extraction (`--model_name clip_RN50`).

### 9.2 Semantic Prototype-guided Cross-attention Aggregator

Key implementation in `SPAR_MIL_Model`:

- `learnable_low_query` / `learnable_high_query`: learnable query vectors introduced at each scale.
- `_norm_anchor()`: anchors the norm of learnable prototypes to that of text prototypes while preserving directional freedom (Eq. 1).
- `_gated_mix()`: Sigmoid-gated fusion of text prototypes and learnable prototypes (Eqs. 2–3).
- `cross_attention_1`: uses `Q_mixed` as query and WSI patch features `M` as key/value to obtain prototype features `C` (Eq. 4).

### 9.3 PrototypeGNN and Top-k Aggregation

- `PrototypeGNN`: built from `nn.TransformerEncoder`, with default `num_layers=2`, `num_heads=4`, `ffn_factor=2`, `dropout=0.1`. Layer count is configurable per dataset (BRCA 5 layers; CAMELYON16 / LUNG 2 layers).
- `_topk_aggregation()`: applies Top-k filtering on prototype attention weights (default `topk_ratio=0.7`), renormalizes, and performs weighted aggregation (Eqs. 5–8).
- Aggregation produces WSI-level image features `image_features_low` / `image_features_high`.

### 9.4 Context-guided Text Enhancer

- `cross_attention_2`: uses class text features as query and image context `C_ctx = [C; M]` as key/value, followed by a residual connection (Eq. 9).
- Establishes a reverse information channel from visual context to class text, allowing text embeddings to perceive the morphological context of the current WSI.

### 9.5 Dynamic Scale Router

- `ScaleRouter`: a small MLP that takes `[f_low; f_high]` as input and outputs unnormalized logits for the two scales (Eq. 10).
- During training, Gumbel-Softmax is used (`--use_gumbel`, enabled by default; temperature `τ=1.0`) to keep differentiability; during inference, Softmax is used (Eqs. 11–12).
- `z_fused = w_low · z_low + w_high · z_high`.
- Ablation modes: `router_mode = gumbel / softmax / static / mean`.

### 9.6 Joint Training Objective

text

```
L_total = L_CE + λ1 · L_InfoNCE + λ2 · L_div + λ3 · L_margin
```



- `L_InfoNCE` (`prototype_contrastive_loss_v2`): cross-scale prototype contrastive loss. It finds optimal low-high prototype pairs via bipartite matching (`_bipartite_match_smart`, preferring scipy Hungarian and falling back to greedy) and applies symmetric InfoNCE between paired prototypes (Eq. 14).
- `L_div` (`prototype_diversity_loss`): prototype diversity loss. It penalizes cosine similarity between different prototypes within the same scale to prevent prototype collapse (Eq. 15).
- `L_margin` (`prototype_margin_loss`): prototype margin loss. It enforces "same-class attraction, different-class repulsion" between WSI-level image features and class text features, with `m = 0.3` (Eq. 13).

## 10. Output Files

### Training Outputs

- `results/{exp_code}_s{seed}/`
  - `s_{fold}_checkpoint.pt`: model checkpoint
  - `split_{fold}_results.pkl`: detailed results per fold
  - `summary.csv`: summary over all folds
  - `result.csv`: mean results and standard deviations

### Evaluation Outputs

- `eval_results/EVAL_{save_exp_code}/`
  - `fold_{fold}.csv`: predictions per fold
  - `summary.csv`: summary results

## 11. Notes

1. **Binary classification constraint**: `SPAR_MIL_Model` explicitly requires `num_classes == 2`. Extending to multi-class classification requires modifying the text prompts, margin loss, and logits construction.
2. **Text prompt length**: `config.text_prompt` must contain `num_classes × 2 = 4` prompts; otherwise, model initialization fails.
3. **Prototype counts**: automatically read from the `prototype_prompt` CSV. `--prototype_number` is recommended to match the CSV to avoid confusion.
4. **Prototype prompt CSV columns**: must contain `scale` (`low` / `high`) and `prompt`.
5. **Memory and GPU**: 64GB+ RAM is recommended for WSI processing; at least 8GB GPU memory is recommended for training. The paper uses 24GB or 8GB devices.
6. **Feature folder naming**: the low-scale feature folder corresponds to `--data_folder_s` (e.g., `features_2048`), and the high-scale feature folder corresponds to `--data_folder_l` (e.g., `features_1024`).
7. **Dynamic routing**: use Gumbel-Softmax during training and Softmax during inference. Do not replace inference fusion with simple addition or averaging.
8. **GNN layers**: default is 2. TCGA-BRCA uses 5 layers; CAMELYON16 and TCGA-LUNG use 2 layers. Too many layers (≥4) tend to overfit under 1-shot / 4-shot settings.
9. **Text backend**: default is `open_clip` (`ViT-B-32`). To use CONCH as the pathology text backend, provide `--text_backend conch --conch_model_cfg ... --conch_checkpoint_path ...`.

## 12. Citation

If you find this work useful, please cite:

bibtex

```
@article{lin2026sparmil,
  title   = {SPAR-MIL: Semantic Prototype-guided Adaptive Dual-scale Routing Multiple Instance Learning for Whole Slide Image Classification},
  author  = {Lin, Xiwen and Author, Second},
  journal = {Bioinformatics},
  year    = {2026},
  note    = {Original Paper}
}
```

## 13. Acknowledgement

The code structure of this project is inspired by [CLAM](https://github.com/mahmoodlab/CLAM), [CoOp](https://github.com/KaiyangZhou/CoOp), and [ViLa-MIL](https://github.com/jiangbo-shi/ViLa-MIL). We thank these excellent open-source projects.

## 14. Contact

For questions, please contact: `calvin_0820@qq.com` (corresponding author email, to be updated).
