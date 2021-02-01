## [DAY 11] 딥러닝 기초
### [AI Math 8강] 베이즈 통계학 맛보기
#### 1. 조건부 확률이란?
+ 사건 B가 일어난 상황에서 사건 A가 발생할 확률을 의미
+ 사건 B가 일어난 상황을 분모로, 교집합의 사건을 분자로 넣었을 때 구할 수 있음

![gif](https://user-images.githubusercontent.com/44515744/106469746-04740c00-64e3-11eb-8e77-e1e1af4d55cf.gif)


![CodeCogsEqn (12)](https://user-images.githubusercontent.com/44515744/106405143-8713b180-6478-11eb-8a3a-a6961e302cfb.gif)

#### 2. 베이즈 정리
+ 조건부 확률을 이용해서 정보를 갱신하는 방법을 알려줌
+ 사건 A를 주어졌을 때 B가 일어날 확률을 아래의 방식으로 구할 수 있음

![CodeCogsEqn (13)](https://user-images.githubusercontent.com/44515744/106405359-4cf6df80-6479-11eb-938e-45e9b5790bbf.gif)

#### 3. 베이즈 정리 예제
+ 아래 이미지를 구체적으로 정리하자면 아래와 같음
    + D = 관찰하는 데이터
    + θ = 모델링하는 이벤트, 모델에서 계산하고 싶어하는 모수(파라미터)
    + 사후 확률(posterior)은 데이터가 주어졌을 때 이 파라미터가 설립할 확률
    + 사전 확률(prior)은 데이터가 주어지기 이전에 사전에 주어진 확률, 데이터를 분석하기 전에 모델링하고자 하는 타겟에 대해 가정을 깔아두고 시작하는 확률 분포
    + 가능도(likelihood)는 현재 주어진 파라미터 모수에서 데이터가 관찰될 확률
    + Evidence는 데이터 전체의 분포

![캡처](https://user-images.githubusercontent.com/44515744/106405546-e6be8c80-6479-11eb-9cb5-3c123bf8e356.JPG)

+ 코로나 발병률 10%, 검진율 99%, 오진율 1% - 양성 판정시 감염되지 않았을 확률
    + 사전확률, 민감도(Recall), 오탐율(False alarm)을 갖고 정밀도(Precision)를 계산하는 문제
    + θ를 발생 사건으로 정의하고, D를 테스트 결과로 정의함
    + ㄱ자를 수식에 표시를 하면 부정을 의미함 (네게이션을 걸었다라고 함)
    + 이때 Evidence는 가능도를 활용해서 계산이 가능함(시그마의 ㄱθ를 활용)
    + P(D)는 질병에 걸렸다고 검사결과가 나올 확률, 아래는 D가 일어날 수 있는 모든 확률을 구해서 더한 것
        + P(D|ㄱθ) 오탐율(False alarm)이 오르면 테스트의 정밀도(Precision)이 떨어짐

![CodeCogsEqn (14)](https://user-images.githubusercontent.com/44515744/106406452-a8769c80-647c-11eb-8c7d-d4611f5f0795.gif)

#### 3. 조건부 확률의 시각화
+ P(θ) = 0.1 사전확률
+ P(ㄱθ) = 0.9 사전확률
+ P(D|θ) = 0.99 민감도(Recall)
+ P(D|ㄱθ) = 0.01 오탐(False alarm)
+ P(ㄱD|ㄱθ) = 0.9 특이도(specitifcity)
+ 양성이 나왔을 때 질병이 걸렸을 확률 - True Positive
+ 양성이 나왔을 때 질병이 걸리지 않았을 확률 - False Positive (1종 오류)
+ 음성이 나왔을 때 질병에 걸렸을 확률 - False Negative (2종 오류)
+ 음성이 나왔을 때 질병이 아닐 확률 - True Negative
+ TP / (TP + FP) = 정밀도 - 오탐율이 오르면 정밀도가 떨어진다.

#### 4. 베이즈 정리를 통한 정보의 갱신
+ 베이즈 정리를 통해 새로운 데이터가 들어왔을 때 앞서 계산한 사후확률을 사전확률로 사용하여 갱신된 사후확률을 계산할 수 있다.
+ 사전확률이 바뀌면 evidence 값도 바뀌게 된다.

#### 5. 조건부 확률 -> 인과관계??
+ 데이터가 많아져도 조건부 확률만 가지고 인과관계를 추론하는 것은 불가능
+ 인과관계는 데이터 분포의 변화에 강건한 예측모형을 만들 때 필요
    + 단, 인과관계만으로는 높은 예측 정확도를 담보하기 어렵다.
    + 조건부확률만 활용해서 예측모형을 만들때는 시나리오마다 정확도 차이가 크다.
+ 인과관계를 알아내기 위해서는 중첩요인의 효과를 제거하고 원인에 해당하는 변수만의 인과관계를 계산해야함
    + 지능 지수를 구할 때 키가 크면 지능이 높다는 결과가 나올 수 있음(나이라는 중첩요인을 제거해야함)
+ 중첩효과를 제거한 인과관계를 고려한 데이터 분석을 하면 안정적인 예측모형 설계가 가능함
+ 단순히 조건부확률을 사용하는 것이 아닌 실제로 변수들간의 인과관계를 파악하여 데이터모형을 설계해야함


### [DLBasic] 딥러닝 기본 용어 설명 - Historical Review
#### 1. 좋은 딥러너가 되는 방법
+ Implementation Skills
+ Math Skills(Linear Algebra, Probability)
+ Knowing a lot of recent Papers

#### 2. 딥러닝에 대한 정의
+ 인공지능 : 사람의 지능을 모방하는 것 (Mimic human intelligence)
    + 인공지능 안에 머신러닝이라는 분야가 존재 (Data-driven approach)
        + 머신러닝 안에 Deep Learning가 있음 Neural networks를 활용하여 데이터를 학습하는 분야

+ 딥러닝에 있어서 필요한 것
    + 모델이 배울 수 있는 데이터
    + 데이터를 변형할 수 있는 모델
    + 모델을 학습시키기 위한 loss 함수가 필요
    + loss를 최소화하기 위한 모수에 적용할 알고리즘

#### 3. 데이터 유형
+ Classification : 분류하는 것 ( 개와 고양이를 구분 )
+ Semantic Segmentation : Fixel을 기준으로 사물을 분류하는 것
+ Detection : 이미지 안의 물체에 대한 바운딩 값을 찾는 것(바운딩 박스를 뱉어줌)
+ Pose Estimation : 이미지 안의 사람의 스켈레톤 정보를 알아냄
+ Visual QnA : 이미지에 대한 질문을 주어졌을 때 그 답을 말하는 것

#### 4. Model
같은 데이터가 주어졌더라도 모델의 성질에 따라 다른 결과가 나옴, 결과를 잘 만들기 위한 테크닉들이 존재
+ alexNet : 컨볼루션 신경망, 2개의 GPU로 병렬 연산을 수행
+ GoogLeNet : 22층으로 구성되어 있으며, 1 x 1 사이즈의 필터로 컨볼루션을 한다. ( 특성맵의 갯수를 줄이는 목적)
+ ResNet : 152개의 층을 가진다. Residual Block을 통해서 입력값을 출력값에 더해줄 수 있도록 지름길을 만들었다.
+ DenseNet : DenseNet은 feature map끼리 Concatenation을 시키는 것이 특징
+ 기타 : LSTM, Deep AutoEncoders, GAN 등등

#### 5. Lost Function
+ 모델이 정해져있고 데이터가 정해져있을 때 이 모델을 어떻게 학습할지에 대한 것
+ 문제를 풀기 위한 근사치, 원하는 결과를 얻는다는 보장이 없음
+ Regression Task : 뉴럴 네트워크의 출력값과 내가 맞추려는 타겟점 사이의 제곱을 최소화하는 것, MSE(Mean Squared Error)
+ Classification Task : 뉴럴 네트워크의 출력값과 라벨값과의 차이를 최소화, CE(Classification Error)
+ Probabilistic Task : 어떤 값이 아니라 값에 대한 평균과 분산으로 MLE(Maximum Likelihood Estimation) 관점으로 문제를 풀 수 있음

#### 6. 최적화 방법
+ 데이터가 정해져있고 모델이 정해져있고 loss Function이 정해져 있을때, 네트워크를 어떻게 줄일지에 대한 얘기
    + 뉴럴 네트워크의 파라미터를 loss function에 대해서 1차 미분한 정보를 활용
    + 모델이 학습하지 않은 데이터에 대해 잘동작하는 것이 목적
+ 여러 알고리즘이 존재 
    + Dropout
    + Early stopping
    + k-fold validataion
    + Weight decay
    + Bathc normalization
    + MixUp
    + Ensemble
    + Bayesian Opimization

#### 7. Historical Review
+ 역사적인 논문에 대한 내용 : Deep Learning's Most Important Ideas = A Brief Historical Review ( Denny Britz)
+ AlexNet
    + Convolution Net에 대한 것, 224 by 224 이미지를 분류하는 것이 목적 
    + 2012년에 AlexNet이 우승한 뒤로부터 딥러닝이 이 분야에서 계속 1등을 하게됨(기계학습의 판도가 딥러닝으로 바뀜)
+ DQN
    + 딥마인드가 알타리라는 게임을 강화학습을 이용해서 풀어낸 것이 DQN
    + DQN을 보고 구글에서 인수
+ Encoder/Decoder
    + NMT를 풀기위한 방법, 단어의 연속이 주어졌을 때 다른 언어의 연속으로 뱉어주는 방법
+ Adam Optimizer
    + 학습시키고자 할 때 여러 Optimizer가 있는데 주로 Adam을 씀
    + 왠만하면 잘된다라는 방법론, 낮은 성능의 하드웨어 사양에서도 잘 동작함
+ Generative Adversarial Network
    + 이미지, 텍스트를 만들어 내는 것에 대한 방법
    + generator와 discriminator 두 개를 만들어서 학습을 시키는 것
+ ResNet
    + 네트워크를 깊게 쌓기때문에 딥러닝이라고 부름
    + 기존에는 네트워크를 깊게 쌓으면 성능이 안좋다는 얘기가 있었음, ResNet이 나온 이후로는 레이어를 어느 정도 깊게 쌓아도 성능이 보장된다는 내용이 전파됨
+ Transformer
    + 웬만한 구조들을 대체하고 비전까지 넘보고 있는 구조, 다른 기존의 방법론과 비교해 장점과 좋은 선능을 가지고 있음
+ Bert(fine-tuned NLP Models)
    + 자연어 처리에서 일반적인 문장들을 학습하고, 풀고자 하는 문제에 fine Tuning을 함 
+ Big Language Models
    + Fine Tuning의 끝판왕 다양한 많은 파라미터(175억개)의 파라미터를 갖고 있는 모델
+ Self-Supervised Learning
    + 한정된 학습데이터 외에 라벨을 모르는 unsupervised 데이터를 활용
    + 라벨이 없는 이미지를 벡터로 잘바꿀 수 있는지에 대한 내용
    + 이미지에 대한 좋은 Representation을 학습 데이터 외에 얻어서 문제를 풀겠다는 것

### [DLBasic] PyTorch 시작하기
> PyTorch와 TF2.0(TensorFlow+Keras)를 많이 사용함
#### 1. 프레임워크 장단점
+ Keras : TensorFlow를 쉽게 쓰기 위해서 만든 언어, 컴퓨터션 그래프(체인 미분용도) Static Graphs를 선언해서 주입
+ TensorFlow : 구글에서 만든 프레임워크, 컴퓨터션 그래프(체인 미분용도) Static Graphs를 선언해서 주입
+ PyTorch
    + Facebook에서 Torch 기반으로 만든 언어, Numpy를 쓰듯이 불러오고 싶을때 불러와서 쓸 수 있음
    + Numpy + AutoGrad + Function
    + Numpy 구조를 가지는 Tensor 객체로 Array 표현
    + 자동미분을 지원하여 DL 연산을 지원
    + 다양한 형태의 DL을 지원하는 함수와 모델을 지원함

#### 2. Colab과 VS Code 연결하기
+ 참고링크 : https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/pytorch/00_utils/colab%20on%20VSCode.md
+ 아래 명령어를 Colab에서 실행 후 출력 값을 VS Code에 설정
``` python
!pip install colab_ssh --upgrade
from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared
launch_ssh_cloudflared(password="upstage")
```

+ PyTorch 관련 함수
``` python
import numpy as np
n_array = np.arange(10).reshape(2,5)

import torch
import torch.nn.functional as F
t_array = torch.FloatTensor(n_adday) # Tensor 생성

print(t_array.shape) # 모양 출력 ([2, 5])
print(t_array.ndim) # 차원수 출력 2
t_array[:2, :3] # 슬라이싱 가능
t1.matmul(t2) # product, np의 dot 기능
t1.mul(t2) # 요소끼리 곱셈
t1.mean() # 평균
t1.view(-1, 2) # np의 reshape와 동일, -1은 자동 지정을 의미
t1.view(-1, 10).squeeze() # 차원 줄이기

y = torch.randint(5, (10,5))
y_label = y.argmax(dim=1) # 최대값이 있는 index 반환
torch.nn.functional.one_hot(y_label) # one_hot 벡터로 변환

w = torch.tensor(2.0, requires_grad=True) # w 값과, 미분여부 입력
y = w**2 # 공식 생성1
z = 2*y + 5 # 공식 생성2

z.backward() # z 미분하기
w.grad # w의 미분값 구하기

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.]) # DQ를 1로 해서 연산하는 것(간단하게 넘어감)
Q.backward(gradient=external_grad)

a.grad # 미분값 반환
b.grad # 미분값 반환
```

### [DLBasic] 뉴럴 네트워크 - MLP
#### 1. 인공 신경망(Neural Networks)란?
+ 포유류의 신경망에서 영감을 받은 시스템, 역전파라는 개념은 실제 생명체의 신경망에는 없는 개념
+ 함수를 모방하는 Function approximators다.

#### 2. 실습
+ MLP 구축 실습 파일
https://colab.research.google.com/drive/1V4m7Hs4qYgnhvHAc1k_QXX2wSH9t2V-f?usp=sharing

### [DLBasic] 데이터셋 다루기
+ config, main, util 등으로 객체지향 형태로 코드 구현함
+ Mnist 데이터셋 생성 코드
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
