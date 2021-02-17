## [DAY 17] NLP - 2차
### (3강) Recurrent Neural Network and Language Modeling
#### 1. Recurrent Neural Network
+ RNN은 시퀀스 데이터가 입력으로 주어질 때 각 타임스텝에서 들어오는 입력벡터 x_t와 이전 타임스텝의 모듈에서 계산한 히든 state 벡터 h_t를 받아서 현재 타임스텝에서의 h_t를 출력으로 내보내는 구조
+ 타임스텝마다 각 단어가 입력되는 구조

![캡처](https://user-images.githubusercontent.com/44515744/108007049-56b74000-7040-11eb-9445-270daaa63aa4.PNG)

#### 2. RNN의 구성 요소
+ h_{t-1}은 이전의 히든 스테이트 벡터
+ x_t는 스텝마다 들어오는 입력 벡터
+ h_t는 현재 스텝에서 만들어지는 히든 스테이트 벡터
+ f_w는 가중치를 갖고있는 RNN 함수
+ y_t는 출력값으로 필요한 스텝에서 결과를 출력한다. (긍,부정일 경우 마지막 스텝에서 추출)

![캡처](https://user-images.githubusercontent.com/44515744/108007426-523f5700-7041-11eb-910f-56d3c2b1ac0d.PNG)

#### 3. Types of RNNs
+ One-to-one : Standard Neural Networks - 시퀀스 데이터가 입력되지 않은 일반적인 구조
+ One-to-many : Image Captioning - 하나의 이미지를 입력으로 받고, 여러번의 타임스텝으로 출력(첫번째 타임스텝에만 값을 입력, 다른 타임스텝은 0으로 채워진 행렬을 입력)
+ many-to-one : Sentiment Classfication - 여러번의 타입스텝으로 입력을 받고 최종적으로 마지막 타임스텝에서 결과를 출력
+ many-to-many : Machine Translation - 여러번의 타임스텝으로 입력을 받고, 이를 기준으로 여러번의 출력이 발생하는 구조(입력이 끝난 후에 여러개의 타임스텝으로 출력)
+ many-to-many(2) : Video classification on frame level - 입력값이 들어올 때마다 문장 성분이나 성분을 예측, 영상 프레임별로 예측을 수행

![캡처](https://user-images.githubusercontent.com/44515744/108009182-7ac95000-7045-11eb-8ea9-cfeb266083d2.PNG)

#### 4. Character-level Language Model
+ 문자열이나 단어들의 순서를 바탕으로 다음에 오는 단어를 예측
+ hidden-layer의 첫번째 입력 값(h0)은 영벡터를 입력으로 준다.
+ 각 단계별로 다음에 나올 글자를 유추하는 로직으로, 이를 활용하면 문장을 학습해서 새로운 문장을 작성할 수 있음
+ RNN을 활용하여 논문, 코드 등을 작성할 수 있음

![캡처](https://user-images.githubusercontent.com/44515744/108010332-f9bf8800-7047-11eb-9b58-3a403770a516.PNG)

#### 5. Backpropagation through time (BPTT) 
+ 타임스텝이 지나면서 누적된 정보를 역전파로 학습하기 위해서는 많은 양 메모리가 소요됨
+ 이를 위해 타임스텝을 일정 크기로 잘라서, 그 크기가 다 찼을때 Backpropagation하여 학습을 진행함
+ 필요한 정보는 히든 스테이트에 저장됨

![캡처](https://user-images.githubusercontent.com/44515744/108011460-75223900-704a-11eb-815a-f9098975eeb5.PNG)

#### 6. Vanishing/Exploding Gradient Problem in RNN
+ 심플한 형태의 Fully Connected로 구성된 Vanila RNN은 많이 사용되지 않음
    + 히든 스테이트가 다음 타임 스텝으로 넘어갈때, 그레디언트가 1보다 클때는 엄청 커지고, 1보다 작을때는 엄청나게 작아지는 패턴이 보임
    + 매 타임스텝마다 히든 스테이트 값이 3배 정도씩 증가

![캡처](https://user-images.githubusercontent.com/44515744/108012130-f4fcd300-704b-11eb-9b9a-c255a2e001fa.PNG)

### (4강) LSTM and GRU
#### 1. Long Short-Term Memory (LSTM)
+ RNN이 배운 Gradient Vanishing/Exploding 문제를 하고 타임스텝이 긴 경우에도 정상적으로 학습할 수 있음
+ 타임스텝마다 히든 스테이트 벡터를 단기 기억을 담당하는 기억소자로 볼 수 있음
+ 단기 기억을 시퀀스가 타임스텝별로 진행하는 원리
+ 이전 타임스텝에서 다른 역할을 하는 두 개의 벡터가 전달(히든 스테이트 벡터, 셀 스테이트 벡터)
    + 셀 스테이트 벡터 : 기억해야할 필요가 있는 모든 정보를 담고있는 벡터
    + 히든 스테이트 벡터 : 현재 타임스텝에서 아웃풋 레이어의 입력으로 사용되는 값
+ 셀 스테이트 벡터를 한번 더 가공한 히든 스테이트 벡터는 아웃풋 레이어 등에 다음 입력 벡터의 입력 값으로 사용
+ 4개의 게이트는 이전 셀 스테이트에서 넘어온 벡터를 적절하게 변환하는 용도로 사용

![캡처](https://user-images.githubusercontent.com/44515744/108013164-748ba180-704e-11eb-9920-701c61225f7c.PNG)

+ Forget gate : 시그모이드 함수를 거쳐서 0과 1사이의 벡터로 변환하고 이전의 스테이트 벡터와 곱해져서 일부 데이터만 보존함(나머지를 잊어버리겠다 = forget)
+ input gate : 시그모이드를 통해서 나온 0~1사이의 값, cell에 업데이트할 정보를 정하는 용도(cut)로 사용
+ Gate gate : 탄 함수를 -1~1 거쳐서 -1~1 사이의 벡터로 변환 
+ 한번의 선형변환 만으로 c_{t-1}를 더해줄 정보를 만들어주기 어려울 경우 

#### 2. Gated Recurrent Unit (GRU)
+ LSTM의 모델 구조를 경량화해서 빠른 계산이 가능해짐
+ 셀 스테이트 벡터와 히든 스테이트 벡터를 일원화하여 히든 스테이트 벡터만 존재
+ 전체적인 동작원리는 LSTM과 비슷함
+ forget gate가 없고 1-input gate를 해당 용도로 사용
+ 현재 스테이트의 값은 input gate 값으로 반영되고, 이전 스테이트의 값은 1-input gate 값으로 곱해져서 서로 더해짐 (가중치 합은 1이됨)
    + 계산양과 메모리를 조금 사용함

![캡처](https://user-images.githubusercontent.com/44515744/108014785-fcbf7600-7051-11eb-88f6-d8afcebae828.PNG)

#### 3. Backpropagation in LSTM?GRU 
+ 필요한 정보를 곱셈이 아닌 덧셈을 통해서 생성하므로 Gradient Vanishing/Exploding가 해결됨
+ 덧셈은 그레디언트를 그대로 보내므로 길어저도 이전 그레디언트를 그대로 보내줄 수 있음

#### 4. 정리
+ RNN은 다양한 시퀀스 데이터에 대한 딥 러닝 모델 구조(Original RNN = Vanila RNN)
+ Vanila RNN은 Gradient Vanishing/Exploding 문제가 있음
+ LSTM, GRU는 타임스탭에서 업데이트하는 것이 기본적으로 덧셈에 기반한 연산이기 때문에 위 문제를 해결할 수 있음