## [DAY 14] Recurrent Neural Networks
### [AI Math 10강] RNN 첫걸음
> 시퀀스 데이터의 개념과 특징, 이를 처리하기 위한 RNN을 소개
#### 1. 시퀀스 데이터 이해하기
+ 소리, 문자열, 주가 등의 데이터를 시퀀스(sequence) 데이터로 분류함
+ 시퀀스 데이터는 독립동등분포(i.i.d.) 가정을 잘 위배하기 때문에 순서를 바꾸거나 과거 정보에 손실이 발생하면 데이터의 확률분포도 바뀌게 됨
+ 과거 정보 또는 앞뒤 맥락 없이 미래를 예측하거나 문장을 완성하는 건 불가능

#### 2. 시퀀스 데이터를 다르는 방법
+ 시퀀스 정보를 가지고 앞으로 발생할 데이터의 확률분포를 다루기 위해 조건부 확률을 이용
+ 베이즈 법칙을 활용하여 시퀀스 데이터를 다룸 ( 조건부 확률을 반복적으로 분해 )
+ 조건부확률을 모델링할때는 곱셈으로 조건부확률를 전개함
+ 시퀀스 데이터를 다루기 위해선 길이가 가변적인 데이터를 다룰 수 있는 모델이 필요함
+ 조건부확률은 과거의 모든 정보를 사용하지만 시퀀스 데이터를 분석할 때 모든 과거 정보들이 필요한 것은 아님
    + τ(타우)라는 고정된 길이만큼의 시퀀스만 사용하는 경우, AR(τ) Autoregressive Model 자기회귀모델이라고 부름
    + 또 다른 방법은 이전 정보를 제외한 나머지 정보들을 H_{t}라는 잠재변수로 인코딩해서 활용하는 잠재 AR 모델
    + autoregressive : 모델이 결과를 예측하고 그 결과를 다시 입력으로 넣고 예측하는 걸 반복하는 것

![캡처](https://user-images.githubusercontent.com/44515744/106832639-63997280-66d5-11eb-8f24-d7f40624685e.JPG)

+ 잠재변수 H_{t}를 신경망을 통해 반복해서 사용하여 시퀀스 데이터의 패턴을 학습하는 모델이 RNN

![캡처](https://user-images.githubusercontent.com/44515744/106833594-1ddda980-66d7-11eb-9320-485010e6bea2.JPG)

#### 3. RNN 이해하기
+ 기본적인 RNN 모형은 MLP와 유사한 모양
+ 이 모델은 t 시점의 데이터를 기준으로만 모델링하기 때문에 과거의 정보를 다룰 수 없음(현재 시점의 데이터를 기준으로만 예측할 수 있음)

![캡처](https://user-images.githubusercontent.com/44515744/106833709-5bdacd80-66d7-11eb-8095-de273b7f529e.JPG)

+ RNN은 이전 순서의 잠재변수와 현재의 입력을 활용하여 모델링
    + 잠재변수인 H_{t}를 복제해서 다음 순서의 잠재변수를 인코딩하는데 사용함
    + 가중치 행렬은 t에 따라서 변하지 않음
    + W_x(1) : 입력 데이터에서부터 선형모델을 통해서 잠재변수로 인코딩하는 가중치 행렬
    + W_h(1) : 이전 시점의 잠재변수로 부터 정보를 받아서 현재 시점의 잠재변수로 인코딩 가중치 행렬
    + W(2) : 잠재변수를 통해서 출력으로 만들어주는 가중치 행렬

![캡처](https://user-images.githubusercontent.com/44515744/106834066-0c48d180-66d8-11eb-8f41-024938b86f4b.JPG)

+ RNN의 역전파는 잠재변수의 연결그래프에 따라 순차적으로 계산
    + 잠재변수의 연결그래프에 따라서 순차적으로 계산, 모든 예측이 이뤄진 다음에 맨 마지막 시점에서의 그레디언트가 과거까지 흐르는 방법
    + BPTT(Backprogagation Through Time)으로 RNN의 역전파 방법
    + 잠재변수에 들어오는 그레디언트는 다음 시점과 출력에서 오는 그레디언트 벡터 두개가 전달됨
+ BPTT를 통해 RNN의 가중치 행렬의 미분을 계산해보면 아래와 같이 미분의 곱으로 이루어진 항이 계산됨

![캡처](https://user-images.githubusercontent.com/44515744/106834611-1ddea900-66d9-11eb-921b-1c6131ee1594.JPG)

+ 시퀀스 길이가 길어지는 경우 BPTT를 통한 역전파 알고리즘의 계산이 불안정해지므로 길이를 끊는 것이 필요함
    + truncated BPTT : 미래의 정보 중에서 몇개는 gradient를 끊고 과거의 정보에 해당하는 몇개의 블록을 나눠서 연산하는 과정
    + 잠재변수에서 오는 gradient를 받는데, 블록 단위로 나눠서 H_t에 들어오는 끊고 O_t에서 오는 그레디언트만 받게하는 원리
+ 이런 문제들 때문에 Vanilla RNN은 길이가 긴 시퀀스를 처리하는데 문제가 존재함
    + 이를 해결하기 위해 등장한 RNN 네트워크가 LSTM과 GPU

![캡처](https://user-images.githubusercontent.com/44515744/106834775-6007ea80-66d9-11eb-872e-9934fbb8d07b.JPG)


### [DLBasic] Sequential Models - RNN
#### 1. Sequential Model
+ 일상생활에서 접하는 데이터는 데이터는 시퀀스 데이터
    + 손동작, 음성, 영상 등등
+ 평소에 얻고 싶은거는 하나로 정의되는 라벨 또는 정보임, 시퀀스 데이터는 정의상 길이가 언제 끝나는지 모름 -> 입력의 차원을 알 수가 없음
+ 입력이 여러번 들어올때 다음 입력에 대해 예측하는 것
    + 과거에 고려해야할 정보량이 더 많아짐
    + 이를 쉽게하기 위해 과거의 τ만큼의 데이터만 조회

![캡처](https://user-images.githubusercontent.com/44515744/106835979-9f373b00-66db-11eb-8e4c-ef333a373056.JPG)

+ Markov model(first-order autoregressive model)
    + 내가 가정하기에 현재는 바로 전 과거에만 종속됨
    + 그 이전의 과거에 대한 정보를 버리게 됨
    + Joint distribution을 표현하기 쉬워짐
    + 과거의 많은 정보를 고려해야하는데 그러지 못하는 문제

+ Latent autoregressive model
    + Hiddent State가 과거의 정보들을 요약하고 있음

![캡처](https://user-images.githubusercontent.com/44515744/106836284-38fee800-66dc-11eb-9a14-046f44a9a3cd.JPG)

#### 2. Recurrent Neural Network
+ MLP와 다르게 자기 자신으로 돌아오는 구조가 존재함
+ h_{t}는 X_{t}뿐만 아니라 이전 t-1 시점에 얻어진 특정 state에도 종속됨
+ RNN이라고 불리는 Recurrent 구조를 사실상 시간순으로 풀면 입력이 많은 Fully connected layer로 볼 수 있음
+ Short-term dependencies : 과거의 정보가 미래에까지 살아남기 힘듦 

![캡처](https://user-images.githubusercontent.com/44515744/106836757-1a4d2100-66dd-11eb-9711-4dda008abce5.JPG)

This image is from. C.Olah's blog

+ RNN은 연산의 수가 아래와 같이 지속적으로 늘어남
+ 활성홤수가 시그모이드면 값이 계속 줄어들게 됨 (varnishing gradient) 그래서 학습이 안됨
+ 활성함수가 ReLU면 0보다 weight들이 계속 곱해지기 때문에 expolding gradient 값이 커져서 네트워크가 폭파됨

![캡처](https://user-images.githubusercontent.com/44515744/106837539-07871c00-66de-11eb-812d-647d018081b0.JPG)

#### 3. Long Short Term Memory
+ long term dependency를 해결하는 방법

![캡처](https://user-images.githubusercontent.com/44515744/106836757-1a4d2100-66dd-11eb-9711-4dda008abce5.JPG)

+ 3개의 입력력을 받고 2개의 출력값을 다음 단계로 전달

![캡처2](https://user-images.githubusercontent.com/44515744/106838177-22a65b80-66df-11eb-8194-8f9000f496d5.JPG)

+ 중간에 흘러가는 cell state가 핵심 아이디어
    + t까지 들어온 정보를 요약하는 것 컨베이어 벨트에 올라온 정보를 판단하고 조작하여 다음으로 넘김
    + 이것을 어떻게 빼고 조작할지가 게이트의 역할

+ Forget Gate 
    + 어떤 정보를 버릴지를 결정함. 
    + 현재의 입력값 x_{t}와 이전의 output인 h_{t-1}가 들어가서 f_{t}를 만들어냄
    + f_{t}는 이전의 Cell State 중 어느것을 버릴지를 결정하는 역할

![캡처4](https://user-images.githubusercontent.com/44515744/106838723-29819e00-66e0-11eb-952b-96ebc831598e.JPG)

+ Input Gate
    + 현재 입력을 Cell State에 올리는 것이 아니라 어느 정보를 올릴지를 정함
    + 예전의 h_{t-1}와 x_{t}를 갖고 i_{t}라는 정보를 생성
    + i_{t}는 어떤 정보를 올릴지를 정함
    + C 틸다가 현재 정보와 이전 출력값으로 만들어진 Cell State 예비군
    + 이전의 cell state와 현재의 cell state를 섞어서 업데이트

![캡처5](https://user-images.githubusercontent.com/44515744/106838725-2ab2cb00-66e0-11eb-9a05-97ccb0321497.JPG)

+ Update Cell
    + forget gate에서 결과만큼 버릴건 버리고
    + C 틸다를 i_{t}만큼 곱해서 이 결과를 기존의 cell state와 합침

![캡처6](https://user-images.githubusercontent.com/44515744/106838761-3b634100-66e0-11eb-878c-ad88e4b0948b.JPG)

+ Output Gate
    + 업데이트된 Cell State를 활용하여 어떤 값을 밖으로 내보낼지를 결정함
    + Output 결과는 다음 Cell로 전달함

![캡처7](https://user-images.githubusercontent.com/44515744/106838762-3c946e00-66e0-11eb-819d-bed26fcce2e5.JPG)

#### 4. Gated Recurrent Unit
+ reset, update 게이트 2개만 존재함 ( output 게이트가 사라짐 )
+ cell state가 없고 hidden state만 존재
+ LSTM 보다 더 자주 활용하나 Transformer가 나오면서 RNN 구조가 다 Transformer로 바뀌는 중
![캡처8](https://user-images.githubusercontent.com/44515744/106839584-da3c6d00-66e1-11eb-8839-9620fcab4d54.JPG)

#### 5. LSTM 실습 강의
+ Sequential 데이터는 전처리가 많이 필요함
+ 원래는 단어로 Dictionary를 만들고, Dictionary를 통해서 one-hot 벡터를 만들고, 인베딩을 만들고 집어넣어야함

### [DLBasic] Sequential Models - Transformer
> 주의! 난이도가 높은 내용
> 자료 출처 : http://jalammar.github.io/illustrated-transformer/
#### 1. Sequential의 문제점
+ Sequential 데이터를 다루는 방법론
+ Sequential 데이터의 길이는 항상 달라지고 내용이빠질 수 있음 -> 다루기가 어려움
    + Original Sequence : 1 2 3 4 5 6 7
    + Trimmed Sequence : 1 2 3 4 5
    + Omitted Sequence : 1 2 4 7
    + Permuted Sequence : 2 3 4 6 5 7
+ Sequential 데이터의 문제를 해결하기 위해 Transformer 구조를 사용함

#### 2. Transformer
+ 재귀적인 구조가 없이 시퀀스를 다루는 모델, Attention이라는 구조를 사용함
+ 시퀀셜한 데이터를 처리하고 인코딩하는 문제, 기계어 뿐만 아니라 이미지 분류, 디텍션, Visual Transformer 등에서도 활용할 수 있음
+ 어떤 문장이 주어지면 이거를 영어 문장으로 바꾸는 것
+ 입력 시퀀스와 출력 시퀀스의 숫자와 도메인이 다를 수 있음
+ 신경망 기계 번역(Neural machine translation, NMT)에 활용

![캡처](https://user-images.githubusercontent.com/44515744/106884056-2a3c2380-6724-11eb-9fdd-c70ed0335824.JPG)

+ RNN은 3개의 단어가 들어가면 3번 돌아가는데, Transformer은 개수와 상관없이 한번에 N개의 단어를 처리할 수 있음
+ 동일한 구조를 갖지만 네트워크 파라미터가 다르게 학습되는 encoder와 decoder가 스택되어 있음
    + 1. N개의 단어가 어떻게 인코더에서 한번에 처리가 되는지?
    + 2. 인코더와 디코더 사이에 어떤 정보를 주고 받는지
    + 3. 디코더가 어떻게 Generation할 수 있는지?

#### 3. Encoder의 구조
+ Self-Attention과 Feed Forward Neural Network(MLP와 동일)를 거치는게 Encoder의 구조
    + Self-Attention이 Transformer가 왜 잘되게 되는지를 나타냄

![캡처](https://user-images.githubusercontent.com/44515744/106884489-b8b0a500-6724-11eb-8cdd-33da61cc40a3.JPG)

+ NMT 문제를 예시로 들자면 각 단어마다 벡터로 표현됨 (3개의 벡터가 생성) 

![캡처](https://user-images.githubusercontent.com/44515744/106884918-455b6300-6725-11eb-858a-8a342794bfe0.JPG)

+ self-Attention은 3개의 단어가 주어지면 3개의 벡터를 찾아줌 ( 단어 -> 벡터 )
    + 각각의 i번째 x 벡터를 Z_{i}로 바꿀때, 나머지 n-1개의 x벡터를 같이 고려해줌
    + 단어를 만들때 나머지 단어들을 모두 활용함

+ Feed Foward는 z_{1}, z_{2}, z_{3}를 같은 Feed-forward에 통과시켜서 변환해주는 것이 전부

![캡처](https://user-images.githubusercontent.com/44515744/106885454-f104b300-6725-11eb-8c71-0bd242b287d0.JPG)

+ Transformer은 단어들간의 관계성을 보고, 이를 학습시킴
![캡처](https://user-images.githubusercontent.com/44515744/106885728-5f497580-6726-11eb-9b4b-bc759384e14d.JPG)

+ 기본적으로 self-Attention 구조는 3가지의 벡터를 만들어 냄(3개의 뉴럴 네트워크가 존재)
    + 각각의 단어마다 Queries, Keys, Values 벡터가 존재(=embedding)

![캡처](https://user-images.githubusercontent.com/44515744/106885944-adf70f80-6726-11eb-9ff8-b8b7860448de.JPG)

+ x1이라는 불리우는 첫번째 단어에 대한 임베딩 벡터를 새로운 벡터로 변경
    + 각각의 단어마다 Score 벡터를 생성
    + Thinking에 대한 스코어 벡터를 계산할때 인코딩을 하고자 하는 쿼리 벡터와 나머지 모든 N-1개 단어에 대한 키벡터를 구해서 내적을 함 ( 두 벡터가 얼마나 유사한지를 정함 )

![캡처](https://user-images.githubusercontent.com/44515744/106889647-a5ed9e80-672b-11eb-925f-56387733e88e.JPG)

+ 특정(i) 벡터와 나머지 단어들 사이에 얼마나 interaction을 해야하는지 파악함

![캡처](https://user-images.githubusercontent.com/44515744/106889729-c289d680-672b-11eb-9fb2-46c9fee02934.JPG)

![캡처](https://user-images.githubusercontent.com/44515744/106889911-ffee6400-672b-11eb-8ef9-287312b09c43.JPG)

+ 스코어 벡터가 나오면 이를 Normalize를 함
    + Score 벡터가 나오면 key vector의 dimension
    + 참고. 모든 데이터 포인트가 동일한 정도의 스케일(중요도)로 반영되도록 해주는 게 정규화(Normalization)의 목표



+ 인코딩을 하고자하는 쿼리벡터와 나머지 모든 N-1개의 벡터의 Key 벡터를 구해서 내적을 함
    + i번째 단어와 나머지 단어들간에 얼마나 Interaction을 해야하는지를 알아서 학습하게함
    + 키 벡터, 쿼리 벡터의 dimension(차원)을 square root를 하여 나눠주고
    + Normalize된 값을 Sum to 1이 될 수 있도록 Softmax를 적용함

![캡처](https://user-images.githubusercontent.com/44515744/106890671-03361f80-672d-11eb-9c7d-b486e8357ff6.JPG)

+ Softmax 결과를 각각의 단어에서 나오는 value 벡터들과 곱하고 이를 합한 것이 인코딩 벡터가 됨
+ 쿼리 벡터와 키 벡터의 차원은 같아야함, Value 벡터의 크기는 달라도 됨

![캡처](https://user-images.githubusercontent.com/44515744/106891130-a5ee9e00-672d-11eb-95be-a5239d5b2e93.JPG)

+ 아래 이미지의 구성 의미
    + X - 2 by 4 (단어가 2개, 4는 단어의 인베딩)
    + Query Vector, Key Vector, Value Vector 매트릭스를 찾아내는 3개의 MLP가 있다고 생각할 수 있음
    + MLP는 인코딩된 단어마다 shared됨
    + 참고. Embedding은 우리가 표현하고자 하는 대상을 벡터공간의 좌표로 매핑하고 표현하는 과정이다.

![캡처](https://user-images.githubusercontent.com/44515744/106892111-f6b2c680-672e-11eb-8e1f-eb4edd3ae984.JPG)

![캡처](https://user-images.githubusercontent.com/44515744/106892408-5f9a3e80-672f-11eb-86c8-90009f1d3eef.JPG)

+ 정리
    + Transformer은 하나의 인풋이 고정되어 있더라도 다른 단어에 따라서 인코딩된 값이 달라지게 됨
    + 따라서 훨씬 더 많은 것을 표현할 수 있음 -> 더 많은 컴퓨테이션이 필요함
    + 한번에 처리하고자 하는 단어가 1000개면 1000^1000의 코스트(n^2)가 필요하기 때문에 length가 길어짐에 따라 메모리를 더 많이써야하는 한계가 존재함

#### 4. Multi-headed attention(MHA)
+ 하나의 인코딩된 벡터에 대해서 쿼리 키 벨류 벡터를 여러개 생성

![캡처](https://user-images.githubusercontent.com/44515744/106893263-a472a500-6730-11eb-9029-489d39d72396.JPG)

+ N개의 Attention을 반복하게되면 N개의 인코딩된 벡터가 나오게됨 
+ 인베딩된 벡터의 디멘션과 인코딩되어 나온 벡터가 항상 같은 차원이어야 함

![캡처](https://user-images.githubusercontent.com/44515744/106893557-121ed100-6731-11eb-8ee2-55cb0eb81fd4.JPG)

+ (learnable) linear map
    + 원래가 10 dimension이었고 8개가 나왔으면 80 dimension 짜리 인코딩된 벡터가 나왔다고 볼 수 있음
    + 이것에 80 x 10 행렬을 곱해서 10차원으로 줄여버림
    + 주어진 임베딩 dimension이 100이라면 실제로 Query, key, value vector를 만드는건 10 dimension 짜리 입력만 갖고 돌아가게 됨 ( 코드랑 이 내용이 살짝 다름 )

![캡처](https://user-images.githubusercontent.com/44515744/106893707-46928d00-6731-11eb-8ed9-d46a93cb7ef1.JPG)

![캡처](https://user-images.githubusercontent.com/44515744/106894151-e7814800-6731-11eb-8fd2-c0b42428bb09.JPG)

#### 5. Positional Encoding
+ 입력에 특정 벡터 값을 더해줌 bias라고 볼 수 있음
+ n개의 단어를 시퀀셜하게 넣었다고 해도 시퀀셜한 정보가 포함되어 있지않음
+ A, B, C, D 각 단어의 순서가 인코딩되는 결과와 다르게 동일한 결과
+ 순서의 개념을 추가하기 위해 주어진 입력에 어떤 값을 더함

![캡처](https://user-images.githubusercontent.com/44515744/106894782-bead8280-6732-11eb-9526-20204c2a664c.JPG)

+ Positional Encoding은 각 위치에 해당하는 값을 벡터에 더해주는 개념(offset을 부여)
    + 최근에는 positional encoding 방식이 변경되었다고 함

![캡처](https://user-images.githubusercontent.com/44515744/106895144-354a8000-6733-11eb-9aee-a6c7a7ecd44c.JPG)

#### 6. Transformer 구조

![캡처](https://user-images.githubusercontent.com/44515744/106895632-d0dbf080-6733-11eb-8ec0-c907b4024b77.JPG)

#### 7. decoder
+ TramsfoRrmer은 Decoder에 Key와 Value를 보낸다.
+ i번째 단어를 만들때 query 벡터와 나머지 단어들의 key 벡터를 곱해서 attention을 만들고
+ 여기에 value vector를 weight sum을 함
+ 그래서 input에 있는 단어들을 decoder에 출력하고자하는 단어들의 attention을 만들려면,
+ input에 해당하는 단어들의 key vector와 value vector가 필요함( 가장 상위 레이어의 단어들을 만든다.)
+ Decoder에 들어가는 단어들로 만들어지는 Query vector와 입력으로 얻어지는 두 개의 벡터를 가지고 최종 출력을 할 수 있게 됨
+ 출력은 Autoregressive하게 한 단어씩 만들게 됨
+ 학습할 때는 입력과 출력 정답을 알고 있음, 학습 단계에서 Masking을 함
    + 이전 단어들에 대해서는 종속하지만 이후 단어에 대해서는 종속하지 않게함
    + "Encoder-Decoder Attention" 레이어는 디코더에 들어간 단어들로 쿼리를 만들고 key, value는 encoder에서 나온 벡터를 사용함

![캡처](https://user-images.githubusercontent.com/44515744/106896808-675ce180-6735-11eb-8cb4-5fb68076448b.JPG)

#### 8. Vision Transformer
+ 이미지 분류, 디텍션 등에도 Transformer를 사용함
+ ViT 논문에서는 이미지 분류를 할때 인코더만 활용함, 맨처음 인코딩한 벡터를 Classfi..(?)에 집어넣음, NMT에서는 문장들이 주어져서 단어들의 시퀀스가 존재함. 이미지는 특정 영역으로 나누고 각 영역에 있는 Patch들을 linear layer에 통과 시켜서 하나의 입력인것처럼 처리

![캡처](https://user-images.githubusercontent.com/44515744/106897675-9aec3b80-6736-11eb-8a99-e60eca4c9ea0.JPG)

#### 9. DALL-E
+ 문장이 주어지면 문장에 대한 이미지를 만들어냄
+ Transformer에 있는 Decoder만 활용, 이미지도 16 by 16으로 나눠서 시퀀스로 transformer에 집어넣음
+ 문장도 역시나 단어들의 시퀀스로 집어넣음
+ GPT-3를 활용했다고 함


### 피어세션 - 논문리뷰
#### 1. Deep Residual Learning for Image Recognition
+ 심층 신경망 훈련을 용이하게 하기 위한 residual learning framework(잔류 학습 프레임워크)에 대한 내용
+ deep model 일수록 파라미터를 줄일 수 있어서 성능이 더 좋다고 알려져 있었고, layer가 깊어지고 있었음 "과연 더 많은 레이어를 쌓는 것만큼 network 성능이 좋아지는가?"에 대한 문제에서 출발
+ 이전에는 깊이가 깊을 수록 네트워크의 파라미터를 줄일 수 있었지만 Varninshing Gradient랑 Exploding Gradient라는 악명 높은 문제가 있었다. ( 정규화된 초기화 및 중간 정규화된 계층에서 해결되었음 )
+ 네트워크 깊이가 증가함에 따라 정확도가 포화 상태가 되고, 그 후 급속히 성능이 저하되는 문제가 발생했다. Degradation Problem(network가 깊어질 수록 accuaracy가 떨어지는 문제)가 존재
+ 이 문제를 해결하기 위해 Deep residual learning framework라는 개념을 도입
    + 쌓여진 레이어가 그 다음 레이어에 바로 적합되는 것이 아니라 잔차의 mapping에 적합하도록 설계
    + 기존의 바로 mapping 하는 것이 H(x)라면 논문에서는 비선형적인 layer 적합인 F(x) = H(x)-x를 제시함 => H(x) = F(x) + x
    + F(x) + x는 Shortcut Connection과 동일한데 이는 하나 또는 이상의 레이어를 skip하게 만들어줌
    + x는 input, F(x)는 Model, H(x)는 결과
    + H(x)를 기본 매핑으로 간주하고 x 가 input 일때 다수의 비선형 레이어가 복잡한 함수를 점근적으로 근사 할 수 있다고 가정하면 잔차함수, 즉 H (x)-x를 무의식적으로 근사 할 수 있다는 가설과 같다고 합니다. (즉 복잡한 함수를 여러개의 비선형 레이어가 근사시킬 수 있으면 잔차함수도 근사할 수 있음.)
    + 즉 input 과 output, x와 F를 모두 같은 차원으로 만들어주는 것이 추가된 과정입니다.
    Shortcut Connection이란 같은 차원으로 나오는 것
    + degradation 문제가 어떻게 해결됐는지는 나와있지 않음
+ residual(잔차)를 이용한 최적화가 결국 skip connection이고 이게 기존보다 더 최적화에 효율적이었다.
+ 입력 x가 들어왔을 때 그 층이 100개 이상이됨( x를 맵핑함수에 넣어만든 결과가 y_너무 많이 쌓이면 정보의 손실이 생김_이미지 정보다 전달이 안됨_따로 빼서 넣어준다 )
+ 여러 연산을 거치면 입력값과 멀어지면서 패딩,스트라이딩 등으로 표현력이 떨어짐 f(x)를 0으로 만들어서 x값을 잘 표현하게 하는 것이 Resnet의 학습방향
+ Varnishing Gradient = degration ( 기울기 소실을 방지함_ 기존값을 더해줌으로써 )

#### 2. Deformable Convolutional Networks
+ 기존의 CNN은 (Receptive Field, 출력 레이어의 뉴런 하나에 영향을 미치는 입력 뉴런들의 공간)의 크기가 고정되어 있는 형태
    + object detection에서 object의 크기는 각각 제각기인데, 고정된 필터사이즈로는 학습해서 제대로된 결과를 얻기가 힘들다.
    + regular MNIST에서는 기존의 CNN이 더 좋은 결과를 보이나, Scaled 또는 rotation MNIST에서는 더 좋은 결과를 보인다.
    + 이미지의 특징에 상관없이 동일한 연산을 수행
+ 이미지의 특성에 따라서 필터의 모양을 유기적으로 변형시키는 모델을 제안
    + 기존의 Receptive Field에는 정해진 개수의 픽셀들이 입력 이미지를 커버하기 위해서 존재하는데, 이 픽셀들의 위치를 유동적으로 변경하기 위해 Offset을 학습하여 적용
    + 필터의 각 픽셀의 위치를 유동적으로 변화시켜서 만들어진 다양한 모양의 필터들을 통해 Convolution을 진행 ( 이 과정에서 입력 특징 맵과 같은 크기를 가지는 Offset Field를 학습)
    + 필터의 해당 픽셀을 offset 방향만큼 이동시키는 것
    + network는 STN (Spaial Transform Network)와 유사한 method를 사용한다. STN을 간단하게 설명을 하자면 아래 이미지와 같이 각각의 치우친 숫자들을 가운데로 정렬시키는 방법
+ Deformable RoI Pooling은 PS Rol Pooling을 기반으로 진행
+ PS Rol Pooling은 pooling 은 크기가 변하는 사각형 입력 region 을 고정된 크기의 feature 로 변환하는 과정이다.

+ 이 모델이 주목받는 이유는, 데이터의 특징을 필터를 통해서 찾는 것이 아니라 입력 데이터 x에서 직접 찾으려는 시도를 했다는 점입니다.
+ 기존 논문들은 weight 를 구하는 방법에 초점을 맞췄다면, 이 논문은 어떤 데이터 x 를 뽑을 것인지에 초점을 맞췄다는 것이 참신하다는 평가

