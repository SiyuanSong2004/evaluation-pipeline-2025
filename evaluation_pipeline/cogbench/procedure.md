# CogBench fMRI Evaluation Procedure

This document describes the complete procedure for evaluating human-model neural alignment using fMRI data. The goal is to measure how well a language model's internal representations align with human brain activity recorded while reading the same text.

## Overview

The evaluation pipeline consists of two main stages:
1. **Inference**: Extract hidden state representations from a language model when processing story text
2. **Evaluation**: Compare model representations with human fMRI data using ridge regression to compute alignment correlations

## Data Structure

The fMRI dataset is organized as follows:

```
evaluation_data/cogbench-fmri/train/
├── story/                  # Text stories (e.g., story_10.txt)
├── node_count_bu/          # Reference TR (Time Repetition) lengths per story
│   └── story_X.mat         # Contains 'word_feature' with shape (1, n_TRs)
├── notPU/                  # Valid word indices (excludes punctuation)
│   └── story_X.mat         # Contains 'isvalid' array marking valid words
└── fmri_dim128/            # fMRI responses per ROI and subject
    ├── Cognition/          # ROI (Region of Interest) type
    │   ├── 01/             # Subject ID
    │   │   └── story_X.mat # Contains 'fmri_response' with shape (128, n_TRs)
    │   └── 02/
    ├── Language/
    ├── Manipulation/
    ├── Memory/
    ├── Reward/
    └── Vision/
```

### Data Details

- **Story Text**: Chinese text files containing discourse-level stories (e.g., 733 words in story_10.txt)
- **fMRI Response**: 128-dimensional reduced fMRI data (n_voxels = 128 after dimensionality reduction)
- **TR (Repetition Time)**: 0.71 seconds - the sampling rate of fMRI data
- **Valid Words**: The `notPU` files mark which words are valid (excludes punctuation words)

## Stage 1: Inference (Feature Extraction)

**File**: `inference/infer_sentence.py`

### Step 1.1: Story Collection
Collect all story files from `story/` directory, parsing story IDs from filenames (e.g., `story_10.txt` → story_id=10).

### Step 1.2: Text Processing
For each story:
1. Read words line by line
2. Split long sequences to fit within model's max position embeddings
3. Tokenize using the model's tokenizer

### Step 1.3: Hidden State Extraction
For each chunk of words:
1. Tokenize with `is_split_into_words=True` to track word boundaries
2. Run forward pass through the model
3. Extract hidden states from specified layer (default: last layer, `layer_index=-1`)
4. **Mean Pooling**: For words split into multiple subword tokens, average the token representations to get a single vector per word

The output is a matrix of shape `(n_words, hidden_size)` where:
- `n_words` = number of valid words in the story
- `hidden_size` = model's hidden dimension (e.g., 768 for BERT-base)

### Step 1.4: Save Features
Save extracted features as `.mat` files: `sentence_feature_story_{id}.mat` with variable `data` containing the word-level representations.

## Stage 2: Evaluation (Alignment Computation)

**File**: `evaluation/eval_discourse.py`

### Step 2.1: Hemodynamic Response Function (HRF) Convolution

The fMRI signal is delayed and dispersed in time relative to neural activity due to the slow BOLD (Blood-Oxygen-Level-Dependent) response. To align model features with fMRI:

1. **Generate HRF**: Use SPM's canonical HRF with parameters:
   - TR = 0.71s
   - Oversampling = 71 (for fine-grained convolution)
   - HRF peak at ~6s, undershoot at ~16s

2. **Create Time Series** (`_postprocess_story_feature`):
   - Create a high-resolution time series at 100Hz (10ms bins)
   - Place each word's feature vector at its end time (from `word_time_features_postprocess`)
   - Filter out invalid words (punctuation) using `notPU` masks

3. **Convolve and Downsample**:
   - Convolve each feature dimension with the HRF
   - Downsample by taking every 71st sample (matching TR = 0.71s)
   - Apply offset of 19 TRs to account for HRF delay
   - Z-score normalize the features

Result: Feature matrix of shape `(n_TRs, feature_dim)` temporally aligned with fMRI data.

### Step 2.2: Load fMRI Data

For each ROI (e.g., Cognition, Language, Vision) and subject:
- Load fMRI responses from `fmri_dim128/{roi}/{subject}/story_X.mat`
- Concatenate across stories
- fMRI data shape: `(n_TRs, 128)` where 128 is the reduced voxel dimension

### Step 2.3: Ridge Regression Alignment

**File**: `utils/data_utils.py`

The alignment between model features and fMRI is measured by how well model features can predict fMRI activity using ridge regression.

#### Training Modes

1. **Nested Cross-Validation** (`ridge_nested_cv`):
   - Used when no explicit test split exists
   - Outer loop: 5-fold cross-validation
   - Inner loop: 5-fold to select best alpha
   - Alphas tested: `np.logspace(-3, 3, 10)` = [0.001, 0.003, ..., 1000]

2. **Train/Dev/Test Split** (`ridge_train_dev_test`):
   - Train on train set, select alpha on dev set, evaluate on test set
   - Best alpha selected by mean correlation on dev set

#### Ridge Regression Math

For a given regularization parameter α:

1. Compute SVD of feature matrix: `X = U · S · V^T`
2. Compute weights: `w = V · diag(S / (S² + α²)) · U^T · y`
   where y is the fMRI activity
3. Predict: `ŷ = X_test · w`
4. Compute correlation: `corr = mean(zscore(ŷ) * zscore(y_test))`

The correlation is computed per voxel and then averaged across voxels to get the alignment score.

### Step 2.4: Result Storage

Results are saved as `.mat` files containing:
- `test_corrs`: Correlation values for each of the 128 dimensions
- `best_alpha`: The optimal regularization parameter (for train/dev/test mode)

## Summary of Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| TR | 0.71s | fMRI sampling rate |
| HRF Oversampling | 71 | For temporal alignment |
| HRF Offset | 19 TRs | Accounts for hemodynamic delay |
| fMRI Dimensions | 128 | Reduced voxel dimension per ROI |
| Ridge Alphas | logspace(-3, 3, 10) | Regularization strengths tested |
| ROI Types | 6 categories | Cognition, Language, Manipulation, Memory, Reward, Vision |

## Alignment Metric Interpretation

The final metric is the **Pearson correlation** between predicted and actual fMRI activity:
- **Higher correlation** = better alignment between model and human brain activity
- Correlations are computed per voxel (dimension) and averaged
- The metric indicates how well the model's representations capture the information encoded in human neural responses during natural reading

## Pipeline Flow

```
Story Text (Chinese)
       ↓
Tokenization + Model Forward Pass
       ↓
Extract Hidden States (layer -1)
       ↓
Mean Pool Subword Tokens → Word Features
       ↓
HRF Convolution + Downsampling (to TR rate)
       ↓
Ridge Regression (features → fMRI)
       ↓
Correlation between predicted and actual fMRI
       ↓
Per-ROI, Per-Subject Alignment Score
```

## Notes

- The model features are extracted at the **word level**, then convolved with HRF to match the **slow fMRI sampling rate**
- Multiple ROIs are tested separately to assess alignment in different functional brain networks
- Ridge regularization prevents overfitting when mapping high-dimensional model features to fMRI voxels
- The 128-dimensional fMRI data is pre-reduced (likely using PCA) from the original voxel space
