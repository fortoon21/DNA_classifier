# DNA_classifier 사용방법

<hr/>

- 목차
  * [1. Installation](#installation)
  * [2. Data](#data)
  * [3. Folder Structure](#folder-structure)

<hr/>

## Installation
가상환경을 설정하기 전에 설치환경은 GPU가 있는 PC(Windows 사용)에서 CUDA 11을 사용하는 것을 전제로 한다. GPU의 종류마다 CUDA 설치 가능 여부가 다르기 때문에 스펙을 잘 확인해야 한다.

각 os마다 CUDA, cuDNN을 설치하는 방법이 다르기에 구글링을 통해 찾아보는 것을 추천하며 예시 url을 같이 두었다.

Windows : https://seong6496.tistory.com/23

Windows에서 가상환경은 Anaconda3를 이용했으며 python버전은 3.8이다.

먼저 Anaconda3를 설치하기 위해 아래의 사이트를 들어가서 os에 맞는 64-bit 프로그램을 다운로드 한다.
> https://www.anaconda.com/distribution/

구체적인 방법은 예시 url을 참고하면 된다. 
Windows : https://needjarvis.tistory.com/563

그 뒤에 가상환경을 만들기 위해서 Anaconda prompt를 키고 아래와 같은 명령어를 입력해주면 DNA_classifier를 실행시키기 위한 Library들을 모두 설치할 수 있다.
```
conda create -n env_name python=3.8
# env_name으로 가상환경 설치

pip uninstall pywin32
conda install pywin32
# pywin32 에러 처리

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
# pytorch 설치

conda install autopep8
pip install -r requirements.txt
# 필요 pkg 설치(pandas, jupyter 등)

ipython kernel install --user --name=env_name
# jupyter notebook에서 env_name으로 kernel을 이용할 수 있도록 해준다.
```
## Data
synapse.org에 회원가입을 한 뒤에 아래의 url을 들어가서 20531개의 RNA sequence의 발현정도를 수치화한 unc.edu_PANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv를 다운로드한다.

https://www.synapse.org/#!Synapse:syn4301332

또한 https://gdc.cancer.gov/about-data/publications/pancanatlas 사이트에서 각 환자들의 id와 dataset의 종류를 알려주고 환자의 암종을 알려주는 merged_sample_quality_annotations.tsv를 다운로드한다.

그리고 다운로드 된 tsv 파일들을 github를 clone한 디렉토리에 위치시켜준다.

## Folder Structure
  ```
  DNA_classifier/
  │
  ├── preprocess.ipynb - TCGA Data preprocessing
  ├── tsne_genedata.ipynb - t-SNE plotting
  ├── visualize_dataset.ipynb - Fake dataset 3D plot
  ├── 
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── feature-selector/ - Use feature-selector
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

2021.01.06. Jeoung Chanyoung 작성