# Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs

Official code implementation for WSDM 23 paper Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs.

Proof supplementary file: [supplementary_file.pdf](./upplementary_file.pdf)

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