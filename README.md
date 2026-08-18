# StrInt
The software implementation of the method in 
[Deciphering more accurate cell-cell interactions by modeling cells and their interactions]().

<img src="https://github.com/deepomicslab/StrInt/blob/rev/main.jpg" alt="StrInt-Main">

# Pre-requirements
* python 3.9.2
* numpy==1.22.3, pandas==1.5.2
* seaborn==0.11.0
* scipy==1.9.3, scanpy==1.9.8
* smurf-imputation

  
# Installation
## 1. Main Package Installation (Python)
```shell
conda create --name strint python=3.9
conda activate strint
python -m pip install --user pyStrint
```
## 2. Analysis Reproduction (R)
```shell
conda install -c conda-forge r-base=4.2.0
```
```R
install.packages(pkgs = 'devtools')
devtools::install_github('linxihui/NNLM')
devtools::install_github('ZJUFanLab/SpaTalk')

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("clusterProfiler")
BiocManager::install(c("org.Hs.eg.db", "org.Mm.eg.db"))
```

# Input file format
## 1. DataFrame format
<details><summary>Expand section</summary>

* **Spatial Transcriptomics (ST) Count Data**
  * `st_exp` dataframe with spots as rows and genes as columns
 
* **Spatial coordinates**
  * `st_coord` dataframe with spot as rows, axis x and y as columns 

* **Cell-type deconvoluted spatial matrix**
  * `st_decon` dataframe with spot as rows and cell-type as columns

* **Single-cell RNA-seq Count Data**
  * `sc_exp` dataframe with cells as rows and genes as columns

* **Single-cell RNA-seq Metadata**
  * `sc_meta` dataframe with cells as rows and cell types as columns
  * `cell_type_key` column name of the celltype identity in `sc_meta`

* **Single-cell RNA-seq distribution Data**
  * `sc_distribution` dataframe with cells as rows and genes as columns
    
* **Ligand and Receptor Data (optional)**
  * `lr_df` user provided dataframe with ligand-receptor pairs as rows, ligand, receptor and its weight as columns

***
Convert to adata format
```python
sc_adata, st_adata, sc_distribution, lr_df = pp.prep_adata(sc_exp = sc_exp, st_exp = st_exp, sc_distribution = sc_smurf, 
                            sc_meta = sc_meta, st_coord = st_coord, SP = species)
```
</details>

## 2. Adata format
<details><summary>Expand section</summary>
  
* **Spatial Transcriptomics (ST) Count Data**
  * `st_adata` adata.X with spots as rows and genes as columns
  * `st_adata.obs`  dataframe with spot as rows, spot coordinates x and y as columns 
 
* **Cell-type deconvoluted spatial matrix**
  * `st_decon` dataframe with spot as rows and cell-type as columns

* **Single-cell RNA-seq Count Data**
  * `sc_adata` adata.X dataframe with cells as rows and genes as columns
  * `sc_adata.obs` dataframe with cells as rows and cell types as columns

* **Single-cell RNA-seq distribution Data**
  * `sc_distribution` dataframe with cells as rows and genes as columns
</details>

# Usages
## Prep object
<details><summary>Expand section</summary>

```python
obj = spamint.spaMint(save_path = outDir, st_adata = st_adata, weight = st_decon, 
                 sc_distribution = sc_distribution, sc_adata = sc_adata, cell_type_key = 'celltype', 
                 st_tp = st_tp)
obj.prep()
```
### Parameters
* `save_path` Output Dir to save results
  
* `st_adata` adata.X Spatial Transcriptomics (ST) Count Data with spots as rows and genes as columns
  * `st_adata.obs`  dataframe with spot as rows, spot coordinates x and y as columns
    
* `weight` Cell-type deconvoluted spatial dataframe with spot as rows and cell-type as columns
    
* `sc_distribution` Single-cell RNA-seq distribution dataframe with cells as rows and genes as columns
    
* `sc_adata` adata.X Single-cell RNA-seq Count dataframe with cells as rows and genes as columns
  * `sc_adata.obs` dataframe with cells as rows and cell types as columns
 
* `cell_type_key` cell type colname in sc_adata.obs

* `st_tp` ST sequencing platform choose from st (ST legacy), visium (10X Visium), or slide-seq (Any single-cell resolution data)


</details>

## Initial process (Cell selection)
<details><summary>Expand section</summary>
  
  ```python
sc_agg_meta = select_cells(self, p = 0.1, mean_num_per_spot = 10,  max_rep = 3, repeat_penalty = 10)
```
* `p` percentage of the interface similarity during cell selection
* `mean_num_per_spot` Average number of cells per spot.
* `max_rep` Maximum number of repetitions for cell selection.
* `repeat_penalty` When one cell has been picked for [THIS] many times, its probability of being picked again decreases by half.
                    Recommanded to be near   (st_exp.shape[0]*num_per_spot/sc_exp.shape[0]) * 10
</details>

## Refinement process (Gradient descent)
<details><summary>Expand section</summary>

 ```python 
refine_sc_exp, sc_agg_meta = gradient_descent(self, alpha = 1, beta = 0.001, gamma = 0.001, 
                 delta = 0.1, eta = 0.0005, 
                init_sc_embed = None,
                iteration = 20, k = 2, W_HVG = 2,
                left_range = 0, right_range = 8, steps = 1, dim = 2)
```

* `alpha, beta, gamma, delta`
  Hyperparameters for the loss function.

  alpha: the weight of the term that maintains the expression similarity between cells and their respective gamma distribution models, default: 1.
  
  beta: the weight of adjusting cell locations based on cell-cell affinity.
  
  gamma: the weight of optimizing interface profile similarity between pseudo-spots and their corresponding ST spots, default: 0.001.
  
  delta: the weight of the regularization term.
  
* `eta` float, default: 0.0005
  
    Learning rate for gradient descent.
  
* `init_sc_embed` DataFrame, optional, default: None
  
    Initial embedding for single-cell data.
  
* `iteration` int, optional, default: 20
  
    The number of iterations for optimization.
  
* `k` int, optional, default: 2
  
    The number of neighbors in each adjacent spot.
  
* `W_HVG` int, optional, default: 2
  
    Weight for highly variable genes.
  
* `left_range` int, optional, default: 0
* `right_range` int, optional, default: 8
  
    The index range for the neighbor number in the embedding process, the actual neighbor number is (i+1)*10
    
* `steps` int, optional, default: 1
  
    The iteration number for each neighbor
  
  
* `dim` int, optional, default: 2
  
    The embedding dimension of the reconstruction
</details>

More details in demo_tutorial.ipynb </br>
tutorial file can be downloaded at: https://drive.google.com/drive/folders/1FYa4hzg3vVo6y2BOzlJbXhPTmdEcjD4O?usp=sharing
