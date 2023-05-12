# Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs

Official code implementation for WSDM 23 paper Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs.

[Paper](https://dl.acm.org/doi/10.1145/3539597.3570465)   
[Proof and supplementary file](./upplementary_file.pdf)    
[Slides](./WSDM-23-GSNOP-Slides.pdf)   
[Poster](./WSDM-23-GSNOP-Poster.pdf)

Source code: `code`
## Environment
* python 3.8
* ubuntu 20.04
* RTX2080
* Anaconda
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c dglteam dgl-cuda11.3
conda install pandas numpy pyyaml tqdm pybind11 psutil scikit-learn
python setup.py build_ext --inplace
python setup.py install
pip install torch-scatter torchdiffeq
```

## Datasets
[All datasets](https://drive.google.com/file/d/1c4_lwUI0DHAPS2ym_p39Fu99QERTVh2z/view?usp=share_link)

## Train and evaluate
```
python train_np.py --data WIKI_0.3 --config config/DySAT.yml --base_model origin --eval_neg_samples 50
python train_np.py --data WIKI_0.3 --config config/DySAT.yml --base_model snp --ode --eval_neg_samples 50
python train_np.py --data WIKI_0.3 --config config/TGN.yml --base_model origin --eval_neg_samples 50
python train_np.py --data WIKI_0.3 --config config/TGN.yml --base_model snp --ode --eval_neg_samples 50
python train_np.py --data WIKI_0.3 --config config/TGAT.yml --base_model origin --eval_neg_samples 50
python train_np.py --data WIKI_0.3 --config config/TGAT.yml --base_model snp --ode --eval_neg_samples 50
python train_np.py --data WIKI_0.3 --config config/APAN.yml --base_model origin --eval_neg_samples 50
python train_np.py --data WIKI_0.3 --config config/APAN.yml --base_model snp --ode --eval_neg_samples 50
python train_np.py --data WIKI_0.3 --config config/JODIE.yml --base_model origin --eval_neg_samples 50
python train_np.py --data WIKI_0.3 --config config/JODIE.yml --base_model snp --ode --eval_neg_samples 50
```
## Bibfile
```
@inproceedings{luo2022gsnop,
author = {Luo, Linhao and Haffari, Gholamreza and Pan, Shirui},
title = {Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs},
year = {2023},
isbn = {9781450394079},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539597.3570465},
doi = {10.1145/3539597.3570465},
booktitle = {Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
pages = {778–786},
numpages = {9},
keywords = {neural process, link prediction, graph neural networks, dynamic graphs},
location = {Singapore, Singapore},
series = {WSDM '23}
}
```
## Acknowlement
This repo is mainly based on [amazon-science/tgl](https://github.com/amazon-science/tgl). We thank the authors for their great works.
