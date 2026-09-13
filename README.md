![MIMYR Method](mimyr.jpg)



# Overview of MIMYR

MIMYR is a generative framework for spatial transcriptomics data reconstruction and imputation. The framework leverages deep learning to predict cellular locations, cell type classifications, and gene expression patterns in spatial transcriptomics data.

The framework consists of three key components:

1. **Location Model**: A diffusion-based model (DDPM) that generates spatial coordinates for cells, with optional KDE-based biological priors for spatially-aware sampling.

2. **Cell Type Model**: A neural network classifier that predicts cell type labels (clusters) based on spatial features and context.

3. **Expression Model**: A model for predicting gene expression patterns conditioned on spatial location and cell type information.

This integrated approach enables robust reconstruction of spatial transcriptomics data by jointly modeling spatial organization, cellular identity, and molecular profiles.

---

# Running MIMYR

## Installation

### Step 1: Create a Conda Environment

We recommend using **Anaconda** to manage your environment. If you haven't already, refer to the [Anaconda webpage](https://www.anaconda.com/) for installation instructions.

Create a Python environment using the following command:
```bash
conda create --name mimyr python=3.10
```

Activate the environment:
```bash
conda activate mimyr
```

### Step 2: Install Dependencies

#### Install PyTorch with CUDA
If you have an NVIDIA GPU and want to use CUDA for acceleration, install PyTorch with the desired CUDA version:
```bash
pip install torch
```

#### Install Remaining Dependencies
Install the remaining required packages:
```bash
pip install -r requirements.txt
```

---

## Running the Code

The pipeline consists of three components: location modeling, cell type classification, and gene expression prediction.

The entry point is main.py, which supports two modes: inference and training, selected via a command-line argument.

### Inference mode
Use inference mode to reproduce the results in the paper. To run using inference mode, use the following command:
```bash
python main.py
```

This will:
1. Automatically download the necessary data and model checkpoints if not present.
2. Load and prepare the spatial transcriptomics data.
3. Run inference using the pretrained models.
4. Evaluate predictions and save results to CSV and artifact directories.

### Training Mode

Training mode fine-tunes the model on a new tissue sample. It requires a directory of `.h5ad` files containing:

- `obsm["aligned_spatial"]`: spatial (CCF) coordinates  
- `obs["cluster"]`: cell-type or cluster labels  
- `X`: gene expression matrix  

After training, rerun `main.py` in inference mode using the newly saved checkpoints to generate predictions.


### Other command Line Arguments

You can customize the pipeline behavior using various command-line arguments:
```bash
python main.py \
  --data_mode rq1 \
  --data_label cluster \
  --location_model_checkpoint model_checkpoints/smoothtune_conditional_ddpm_2d_checkpoint_400.pt \
  --cluster_model_checkpoint model_checkpoints/best_model_rq1.pt \
  --expression_model_checkpoint model_checkpoints/TG-base4_epoch4_model.pt \
  --batch_size 1024 \
  --device cuda \
  --out_csv results/output.csv
```

#### Key Arguments:
- `--data_mode`: Dataset mode (default: 'rq1')
- `--data_label`: Label type for classification (default: 'cluster')
- `--location_inference_type`: Type of location inference ('model' or 'closest_slice' or 'skip')
- `--cluster_inference_type`: Type of celltype inference ('model' or 'majority_baseline' or 'skip')
- `--expression_inference_type`: Type of expression inference ('model' or 'lookup' or 'skip')
- `--kde_bandwidth`: Bandwidth for KDE-based location model (default: 0.01)
- `--guidance_signal`: Guidance strength for conditional generation (default: 0.01)
- `--metrics`: Comma-separated list of evaluation metrics (default: 'soft_accuracy')
- `--metric_sampling`: Percentage of samples for metric computation (default: 100)
- `--device`: Computing device ('cuda' or 'cpu')

### Output

The pipeline generates:
- **CSV file**: Evaluation metrics for each test slice
- **Artifact directory**: Contains per-slice results including:
  - `config.json`: Configuration parameters used
  - `results.json`: Evaluation metrics
  - `pred.pkl`: Prediction outputs

---

## Project Structure
```
MIMYR/
├── models/
│   ├── diffusion_model.py       # DDPM location model
│   ├── celltype_model.py        # Cell type classifier
│   ├── biological_model.py      # KDE-based spatial prior
│   └── gene_exp_model.py        # Gene expression model
├── data_loader.py               # Data loading and preprocessing
├── inference.py                 # Inference pipeline
├── evaluator.py                 # Evaluation metrics
├── main.py                      # Main execution script
└── model_checkpoints/           # Pretrained model weights
```

