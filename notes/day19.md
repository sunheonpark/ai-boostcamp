## [DAY 18] NLP - 4차
### (7강) Transformer I
#### 1. Transformer : High-level view
+ Attention이 기존에는 추가적인 Add-on 형태로 사용됐다면, Attention만을 사용해서 RNN을 걷어내고 대체할 수 있음
+ Attention만을 사용해서 시퀀스 데이터를 입력으로 받고 예측할 수 있음 (RNN과 CNN 미사용)

#### 2. RNN: Long-term dependency
    + 기존의 RNN은 왼쪽의 단어로부터 정보를 계속 축적하면서 히든 스테이트 벡터를 인코딩
    + Gradient Vanishing,exploding 등 정보의 유실 혹은 유실 이슈가 존재하게됨

![캡처](https://user-images.githubusercontent.com/44515744/108291412-9b281480-71d5-11eb-89ba-e540412f9f7a.PNG)

#### 3. Bi-Drectional RNNs
+ 기존의 RNN의 경우에는 정보의 진행방향 때문에 오른쪽에서 나타난 정보를 담을 수가 없었음
+ 주어진 동일한 시퀀스에 대해 오른쪽에서 왼쪽으로 인코딩하는 방식도 생각할 수 있음
+ Forward RNN 좌측에서 우측으로 인코딩하는 방법
+ Backward RNN 우측에서 좌측으로 인코딩하는 방법
+ 위 두 RNN을 병렬적으로 만들고, 동일한 위치의 두 히든 스테이트 벡터를 콘캣함으로써 (양방향 정보를 동시에 적재)

#### 3. Transformer : Long-Term Dependency
+ 3단어에 대한 인풋 벡터가 주어진다면 
+ 하나의 인풋 벡터가 나머지 인풋 벡터와 내적을 하고 주어진 세개의 벡터 간의 유사도를 구하게 됨
+ 내적에 기반한 유사도를 통한 가중치가 정의됐다면, 특정 벡터에 대한 가중 평균을 인코딩 벡터로 활용할 수 있음
+ 인풋 벡터 : 자기자신과 내적, 다른 인풋과 내적, 유사도 계산, 소프트 맥스를 통한 가중치를 계산, 가중 평균을 계산 후 인코딩 벡터로 계산
+ 디코더 히든스테이트 벡터 인코더 히든스테이드 벡터 간의 구별없이 동일한 세트의 벡터 내에서 적용이 된다는 점에서, Self Attention 모듈이라고 함
+ 주어진 입력에서의 각 word에 대한 벡터가 가중치를 구하는 역할을 함 - Query 벡터
+ Query 벡터와 내적이 되는 다른 입력 벡터들을 Key 벡터라고 함
+ 유사도를 구한 후 가중치를 구한 평균이 재료 벡터로 사용됨 - Value 벡터
+ 한 시퀀스를 인코딩하는 과정에서 각 벡터들이 쿼리, 키, 값 역할을 함
+ 벡터가 각 역할에 따라 다른 형태로 변환됨(확장된 형태)
+ I라는 워드를 인코딩한다고 가정하면
    + Wq라는 매트릭스에 의해서 Q라는 쿼리 벡터를 생성
    + 3개의 키벡터와 Value 벡터를 생성하는데, 첫번째 K 벡터는 첫번째 Value 벡터와 매칭
    + 유사도와 가중치가 매칭되어 같은 개수로 존재
+ 각각의 쿼리 벡터와 키 벡터를 어텐션의 연산과정을 통해서 내적값을 구하고 이를 통해서 소프트 맥스를 구함
+ 소프트 맥스의 결과는 밸류 벡터에 부여되는 가중치가 됨,
+ 이 벨류 벡터의 가중 평균에 대한 결과 벡터가 나오고 이것이 입력 벡터의 결과 h벡터가 됨

![캡처](https://user-images.githubusercontent.com/44515744/108297782-1f32ca00-71df-11eb-9484-476c38561cd8.PNG)

+ 입력벡터들을 콘캣으로 합침 후 Wq 행렬과 내적한 결과가 하나의 쿼리 벡터가 됨
+ 마찬가지로 Wk, Wv 행렬을 곱한게 각각의 결과 벡터가 됨
+ 하나의 입력에 대한 결과 벡터를 만들때 모든 입력 벡터에 대한 정보를 다 갖고 있다는게 장점

![캡처](https://user-images.githubusercontent.com/44515744/108298265-04ad2080-71e0-11eb-9349-092aa21e0830.PNG)

#### 4. Transformer : Scaled Dot-Product Attention
+ q : 쿼리 벡터
+ (k,v) : 키와 벨류 벡터 쌍
+ Output 벡터는 v 벡터에 대한 가중치 평균
+ 가중 평균에 쓰이는 가중치는 쿼리 벡터와 벨류 벡터에 해당하는 키 벡터와의 내적을 통해 구해짐
+ 쿼리 벡터와 키 벡터는 같은 차원의 벡터여야함
+ q와 k를 내적한 것에 대한 소프트 맥스를 통과한 값에 매칭되는 Value 벡터를 곱한 뒤 이를 더해주면 이것이 가중 평균이 됨

![캡처](https://user-images.githubusercontent.com/44515744/108299499-e34d3400-71e1-11eb-92c9-42e15c139847.PNG)

+ 연산을 위해 K벡터를 Transpose해서 사용

![캡처](https://user-images.githubusercontent.com/44515744/108299964-c49b6d00-71e2-11eb-89c5-9cbbe9375a45.PNG)

#### 5. Transformer : Scaled Dot-Product Attention
+ 특정 쿼리 벡터와 키벡터 하나가 내적이 됨, 두 벡터의 차원수는 다양하게 정의될 수 있음
+ 벡터의 각각 원소의 값을 확률 분포로 생각할 경우, 만약 평균이 0, 분산이 1이라면 내적 값의 평균도 0이 됨
+ 최종적인 분산은 2가되게 됨
+ 분산이 클 경우에는 소프트맥스에 입력값의 차이가 클 수 있고 이때, 특정 값으로 확률이 몰리게 됨
+ 내적은 요소의 곱의 합이기 때문에 각 요소들이 더해지면서 분산이 커지게 됨(디멘션이 커질 수록 분산도 커지는 구조)
+ 따라서 쿼리 벡터와 키벡터를 내적한 결과에서 디멘션수의 제곱근을 나눠주어서 분산을 1로 만듬
    + 특정 수를 나눠줄 경우 분산 측면에서는 그 수가 제곱되어 나눠진 것 - 제곱근으로 나눔

![캡처](https://user-images.githubusercontent.com/44515744/108302516-7b99e780-71e7-11eb-8b7a-b644c463453d.PNG)

### (8강) Transformer II
#### 1. Transformer: Multi-Head Attention
+ 여러 버전의 W_k, W_v, W_q가 존재하여 동일한 쿼리 벡터에 대한 다양한 인코딩 벡터가 나옴
+ 어텐션의 개수만큼 인코딩 벡터를 콘캣함으로써 하나의 종합적인 인코딩 벡터를 얻을 수 있음
+ 동일한 시퀀스가 주어졌을 때도 특정한 쿼리 워드에 대해서 서로 다른 기준으로 여러 측면에서 정보를 뽑아야할 수 있음
+ 서로 다른 측면의 정보를 병렬적으로 뽑고 합치는 형태로 어텐션 모듈을 구현할 수 있음(각각의 헤드가 정보를 상호 보완)

![캡처](https://user-images.githubusercontent.com/44515744/108303776-e8ae7c80-71e9-11eb-9ce6-0143a61bd5ad.PNG)


![캡처](https://user-images.githubusercontent.com/44515744/108303756-d6ccd980-71e9-11eb-94ab-562db210b2a6.PNG)

+ 이렇게 Concat한 벡터를 원하는 차원으로 축소함

![캡처](https://user-images.githubusercontent.com/44515744/108303886-26aba080-71ea-11eb-809c-aae1ab4ecc26.PNG)

+ 모델 별 시간복잡도
    + n은 시퀀스 길이
    + d는 표현하는 차원의 수 (하이퍼 파라미터로 설정 가능)
    + k는 콘불루션의 커널 사이즈
    + r은 제한된 셀프 어텐션에 있는 neighborhood의 사이즈
    + Attention은 시퀀스 길이가 길어질 경우, 더 많은 메모리가 요구됨 ( 모든 쿼리와 키 벡터 간의 내적값을 저장하고 있어야함)
    + Self Attention은 시퀀스 길이가 길더라도 GPU 코어수가 뒷받침 된다면 동시에 실행이 가능 ( 더 빠르게 학습이 가능)
        + RNN은 재귀적인 방식으로 진행되기 때문에 타임스탭만큼의 시간이 소요됨

![캡처](https://user-images.githubusercontent.com/44515744/108304375-42637680-71eb-11eb-9923-75e752fbd9af.PNG)

+ Residual connection은 깊은 레이어의 신경망을 만들때 Gradient Vanishing을 막을 수 있는 효과적인 모듈 중 하나
    + Attention의 결과로 나온 아웃풋 벡터에 입력 벡터를 더해서 최종적인 인코딩 벡터를 얻어내는 것
    + 이렇게 만든 데이터는 입력값 대비 차이 값을 어텐션 벡터에서 만들게 됨(입력 백터와 아웃풋 벡터의 크기가 동일해야함)
+ Transformer에서 한 블록은 Multi-Head Attention과 Residual고 나온 Word별로 가지는 인코딩 벡터에,
+ 추가적인 Fully Connected layer를 한번 더 거쳐서 인코딩 벡터를 변환하고 Residual connection과 layer normalization을 한번 더 진행

![캡처](https://user-images.githubusercontent.com/44515744/108304977-755a3a00-71ec-11eb-8f50-7ac3237692a0.PNG)


#### 2. Transformer: Layer Normalization
+ Normalization Layer는 주어진 다수의 샘플에 대해서 평균을 0, 분산을 1로 만들어 준 후 원하는 평균과 분산을 주입할 수 있는 선형변환으로 이뤄짐
+ 특정 노드의 값들에서 평균 값을 구한한 뒤 편차를 구하고, 각 편차를 표준편차로 나눠주면 평균이 0이고 분산이 1이 됨(표준정규분포)

![캡처](https://user-images.githubusercontent.com/44515744/108306011-a0458d80-71ee-11eb-9413-4c5ad5de08b2.PNG)

+ Affine transformation은 노드별로 동일한 연산을 적용하여 평균과 분산을 변경함

![캡처](https://user-images.githubusercontent.com/44515744/108307443-18ad4e00-71f1-11eb-9e20-9eaa2e5d8567.PNG)

#### 3. Transformer: Positional Encoding
+ 문장의 순서가 바뀐채로 인코딩될 경우, 동일한 결과 값이 나오게 됨(서로 다른 단어간의 유사도만을 기준으로 벡터가 생성되기 때문에) - 순서가 없는 집합에 인코딩하는 구조
    + 시퀀스 정보를 고려하는 RNN과 달리 시퀀스 정보를 반영할 수 없음
+ Positional Encoding
    + 'I'라는 단어의 입력 벡터의 위치에 해당하는 값에 특정 값을 더함(순서를 특정지을 수 있는 상수 벡터를 각 순서에 해당하는 벡터에 더해줌) 
    + 실제로는 sin과 cos등으로 이루어진 주기함수를 사용하고 여러 함수 값을 모아서 위치를 나타내는 벡터를 사용

![캡처](https://user-images.githubusercontent.com/44515744/108308449-1d730180-71f3-11eb-9643-8cb3b13441a5.PNG)

+ 위 벡터에서 입력 벡터 포지션에 해당하는 row 번째 디멘션을 입력벡터에 더해줌으로써 위치에 따라 다른 값이 나오게함
![캡처](https://user-images.githubusercontent.com/44515744/108308688-80fd2f00-71f3-11eb-8b16-bcddeaad7891.PNG)

#### 4. Transformer: Warm-up Learning Rate Scheduler
+ 최적화 과정에서 사용하는 learning rate를 학습 과정중에 적절하게 변경하는 방식
+ 초반에는 기울기가 클 수 있으므로 learning 작게해서 보폭을 감소 시키고
+ 어느정도 진행되면 다시 보폭을 키워주고 그 다음부터는 이를 줄여준다.

![캡처](https://user-images.githubusercontent.com/44515744/108309337-b6eee300-71f4-11eb-8dad-993a5fa16768.PNG)

#### 5. Transformer: Decoder

+ 인코딩 블록을 N번으로 여러번 쌓아서 보다 Highlevel의 encoding vector를 구한다.

![캡처](https://user-images.githubusercontent.com/44515744/108309554-13520280-71f5-11eb-84e3-ec69f20f7d26.PNG)

+ 디코더 입력 값이 주어지고, 이를 Attention한 아웃풋 벡터에 인코더의 key, value 벡터의 최종값을 활용하여 Attention을 진행함
    + Residual connection을 통해 디코더와 인코더에서 가져온 정보가 잘 결합되는 형태

![캡처](https://user-images.githubusercontent.com/44515744/108310484-cc650c80-71f6-11eb-976d-a20c50dac585.PNG)

#### 6. Transformer: Masked Self-Attention
+ 디코더에서 학습을 위해 사용되는 기법
+ attention을 모두가 볼 수 있도록하고, 후처리로 보지 말아야할 영역의 가중치를 0으로 만들고 이후 Value 벡터와 가중 평균을 내는 방식으로 Attention을 변형한 방식
+ 학습 당시에는 배치 프로세스에 의해서 값이 동시에 주어지긴 하나, SOS를 쿼리로 해서 Attention 모듈을 사용할떄는 접근 가능한 Key,Value에서 나머지 값들을 제외해줘야함
+ 아래와 같이 대각선 라인을 기준으로 윗부분을 0으로 만들어주고 다시 확률의 총합이 1이 되도록 Normalize를 해줌

![image](https://user-images.githubusercontent.com/44515744/108311105-dc312080-71f7-11eb-808f-f6d22cb9591e.png)