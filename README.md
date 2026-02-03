# L1DeepMETv2

## Introduction

L1DeepMETv2 is a **GNN** based algorithm for **online** MET regression for the Phase-2 CMS Level-1 Trigger, using L1 PUPPI candidates as input. It regresses to the generator-level MET by assigning a per-particle weight that is applied to the L1 PUPPI candidates to compute the L1 MET sum.

This work extends the following efforts for MET reconstruction algorithms:
- **Fully-Connected Neural Network (FCNN)** based algorithm in `keras` for **offline** MET reconstruction: [DeepMETv1](https://github.com/DeepMETv2/DeepMETv1)
- **Graph Neural Network (GNN)** based algorithm in `torch` for **offline** MET reconstruction: [DeepMETv2](https://github.com/DeepMETv2/DeepMETv2)
- **FCNN** based algorithm in `keras` for **online** MET reconstruction: [L1MET](https://github.com/jmduarte/L1METML) 

For the L1DeepMETv2 algorithm itself, there are several versions, each relying on different model architectures or training samples.
- **The main version** can be found here: [L1DeepMETv2](https://github.com/DeepMETv2/L1DeepMETv2). Trained with central CMS samples of 200 PU tt-bar events that are produced with the [FastPUPPI framework](https://github.com/p2l1pfp/FastPUPPI/blob/11_1_X/NtupleProducer/python/runPerformanceNTuple.py#L588-L601) in CMSSW_11_1. These are the same samples as originally used for the online FCNN. They are preprocessed for training the L1DeepMETv2 as described in the [README](https://github.com/DeepMETv2/L1DeepMETv2/tree/master?tab=readme-ov-file). 
- **The Delphes version** can be found here: [Delphes version](https://github.com/Pmeiring/L1DeepMETv2/tree/dev_delphes_oldmodel). The model architecture is largely the same, but it relies on Delphes simulated samples of 200 PU tt-bar events to provide the L1 PUPPI input candidates. These were produced by Eric Anton Moreno et al. in the context of the [NGT project](https://indico.cern.ch/event/1594908/contributions/6721724/attachments/3154497/5602747/L1T%20weekly%20WP3.7.pdf) (see also this [sample-bank](https://huggingface.co/datasets/fastmachinelearning/collide-1m/tree/main)). This Delphes model version is used for the FlowGNN+EdgeConv publication with GaTech.
- **The HGQ version** can be found here: [HGQ version](https://github.com/Pmeiring/L1DeepMETv2/tree/dev_hgq). This is the product of Leo Yao and Jocelyn Yu (originally pushed to a [seperate repo](https://github.com/LeoY20/L1DeepMETv2-HGQ)), who worked on providing a high-granularity quantization (HGQ) version of L1DeepMETv2, which required for the first time also implementing the quantization for the EdgeConv layer. 
- **The Golden-C version** can be found here: [Golden-C version](https://github.com/Pmeiring/L1DeeptMetv2-GoldenC/tree/main). This is the C++ implementation of the Pytorch version of the L1DeepMETv2 model. It is a fork of the [repo of Davendra Maharaj'](https://github.com/davendramaharaj1/L1DeeptMetv2-GoldenC/tree/main), who originally implemented this based on the model "main version". However, for the FlowGNN+EdgeConv publication with GaTech, Peter updated the C++ version repo to point to the newer Delphes model. Tu Pham worked on this repo as well to address some mismatches in the PyTorch and C++ model predictions - so there may be a more up-to-date fork.
- **Miscellaneous versions**: Several other versions - newer than the "main version" - were trained with other architectures (eg. sigmoid vs. Relu activation), but these did not seem to improve performance a lot, hence the above versions are probably the most important ones.

## Prerequisites 

```
conda install cudatoolkit=10.2
conda install -c pytorch pytorch=1.12.0
export CUDA="cu102"
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.0+${CUDA}.html
pip install torch-geometric
pip install coffea
pip install mplhep
```
If running on one of the rogue machines (rogue01, rogue02), use micromamba instead of conda. That is, instead of above commands: 

```
# 1. Create and activate new environment with python 
micromamba create -n deepmet python=3.9
micromamba activate deepmet

# 2. Install PyTorch + CUDA 
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0 --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric extensions 
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 4. Install torch-geometric meta package 
pip install torch-geometric

# 5. Other Python packages
pip install coffea mplhep

```



## Produce Input Data

For producing training input data, we use _TTbar process_ simulation data available in [this link](https://cernbox.cern.ch/files/link/public/JK2InUjatHFxFbf?tiles-size=1&items-per-page=100&view-mode=resource-table). The data files are in `.root` format and they contain _L1_ information; the full list of variables available in the `.root` files can be found in [`./data_ttbar/branch_list_L1_TTbar.csv`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/data_ttbar/branch_list_L1_TTbar.csv). From the list of variables, we extract the ones that will be used for our training and save it to `.npz` format under `./data_ttbar/raw/`, using [`./data_ttbar/generate_npz.py`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/data_ttbar/generate_npz.py). 

```
L1PuppiCands_pt, L1PuppiCands_eta, L1PuppiCands_phi, L1PuppiCands_puppiWeight, L1PuppiCands_pdgId, L1PuppiCands_charge 
```

From these six variables, we make the following training inputs into `.pt` files under `./data_ttbar/processed/` using [`./model/data_loader.py`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/model/data_loader.py):

```
pt, px (= pt*cos(phi)), py (= pt*sin(phi)), eta, phi, puppiWeight, pdgId, charge 
```


## Get Input Data and Train 

Training script [`train.py`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/train.py) will use input data using torch dataloader. Training input in `.npz` format is available in [this link](https://cernbox.cern.ch/s/RETpE7fzw4g0lnF) under `/raw/`. Follow these steps to get the input data and train the algorithm.

1. From [this link](https://cernbox.cern.ch/s/RETpE7fzw4g0lnF), download the `/raw/` folder containing `.npz` files and place the folder under `./data_ttbar/`.
2. When you run the training script right after step 1, the training script will fetch the `.npz` files from `./data_ttbar/raw` and produce dataloader from these files, using the functions defined in [`./model/data_loader.py`](https://github.com/DeepMETv2/L1DeepMETv2/blob/master/model/data_loader.py). This dataloader-producing takes some time, so feel free to use the dataloader already produced and saved in [this link](https://cernbox.cern.ch/s/RETpE7fzw4g0lnF). From the link, download `processed.tar.gz` that contains dataloader saved as `.pt` files: un-compress the file and place the `.pt` files under `./data_ttbar/processed/`.
3. Run the training script `train.py` using the following command:
```
python train.py --data data_ttbar --ckpts ckpts_ttbar
```
If you have done the step 2, the training script will directly fetch the input dataloader from `./data_ttbar/processed/` and save the training & evaluation output to `./ckpts_ttbar`.


## Access the train dataloader and test dataloader

dataloader saved as `.pt` files in `data_ttbar` is not split into training and test set. What is done at the training/evaluating code is that it splits the full dataloader into 8:2 training-test set on the fly. 

The training dataloader and test dataloader with 8:2 split that can be directly used can be downloaded [here](https://cernbox.cern.ch/s/oNs7GNCOi7ZX7ak) 
Once you download them, you can use them in your training/evaluating code as the following:
```
test_dl = torch.load('dataloader/test_dataloader.pth')
```

## Plot performance plots (response and resolution)

```
python3 plt.py --ckpts [ckpts directory]
```
