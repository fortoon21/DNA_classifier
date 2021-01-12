# DNA_classifier 사용방법

<hr/>

- 목차
  * [1. Installation](#installation)
  * [2. Data](#data)
  * [3. Train and Test](#train-and-test)
  * [4. Make DNA sequences](#make-dna-sequences)
  * [5. Folder Structure](#folder-structure)

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

preprocess.ipynb을 실행시켜 tiny_data.df.tsv 파일을 만든다.

그 후, feature-selector/Feature Selector Usage.ipynb를 실행시켜 유의미한 feature(gene)만 남기고 tiny_data로 feature의 수를 10, 20, 30, 40, 50 등 다양하게 하여 tiny_data_10.tsv, tiny_label_10.tsv 파일로 만들어준다. 그리고 각 결과를 tsne_genedata.ipynb 파일을 이용해 t-SNE처리한 뒤 저장해준다.(2D, 3D, 다른 visualization 방법을 이용할 수도 있다.)

fake data 같은 경우 sklearn을 이용해서 만들었고 데이터 분포를 보기 위해 visualize_dataset.ipynb를 이용하여 3D로 plot, 회전하는 animation을 만들었다.


## Train and Test
pytorch-template에 들어가 각 훈련하고 싶은 feature의 갯수의 config.json file을 만들어준 뒤에 아래와 같은 명령어를 입력한다.

```
> python train.py -c config.json
```

 각 model의 weight.pth를 실험하기 위해서 test.py를 실행시키려면 아래와 같은 명령어를 입력한다. 명령어에서 saved/models/는 .pth가 위치하는 relative path이다.
 ```
 > python test.py -c config.json -r saved/models/weight.pth
 ```
 실제 제한된 조건(각 weight 값이 0.X꼴, bias 모두 0이하)이 잘 구동되는지 확인하기 위해 pytorch-template에 Check results.ipynb를 작성하여 각 과정을 fully-connected layer를 직접 행렬연산으로 구현했을 때, 같은 결과를 내놓는지 확인하고 SVM(Support Vector Machine)으로 구현했을 때의 성능을 비교한다.

 ## Make DNA sequences
pytorch-template 내에 Making_Seq.ipynb를 이용해서 DNA를 만들 수 있으며 만약 model이 커서 sequence 참조에 오류가 나면 seesaw compiler를 이용해서 서로 orthogonal한 DNA sequence들을 충분히 만들어서 4번째 cell의 sequences에 붙여넣기하면 된다.

결과는 jupyter notebook에 print가 되며, 따로 5'->3'로 바꾸는 함수를 작성했으나 확인의 편의를 위해 한 번에 result를 내놓지 않도록 했다.

## Folder Structure
  ```
  DNA_classifier/
  │
  ├── README.md
  ├── .gitignore 
  ├── requirements.txt
  ├── preprocess.ipynb - TCGA Data preprocessing
  ├── tsne_genedata.ipynb - t-SNE plotting
  ├── visualize_dataset.ipynb - Fake dataset 3D plot
  │
  ├── feature-selector/ Use feature-selector(check README in this directory for detail)
  │
  └── pytorch-template/ - Use pytorch-template for training models(check README in this directory for detail)


2021.01.06. Jeoung Chanyoung 작성