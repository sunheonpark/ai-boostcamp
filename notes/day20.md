## [DAY 20] NLP - 5차
### (9강) Self-supervised Pre-training Models
#### 1. 최신 동향
+ Transformer 및 self-attention은 범용적인 인코더, 디코더로 자연어 처리 뿐만 아니라 다양한 분야에서도 활발하게 사용되고 있음
+ Transformer은 self-attention 블록을 6개만 쌓았다면 최근에는 더 많은 블록을 쌓고있는 추세
+ self-attention은 추천 시스템, 신약개발, 영상처리 등에서도 사용되는 중
+ Transformer의 self-attention에 기반한 모델들도 자연어 생성이라는 테스크에 있어서 자연어를 좌측부터 하나씩 생성한다는 그리디 디코딩 프레임워크에서 벗어나고 있지 못하고 있음

#### 2. GPT-1
+ 다양한 스페셜 토큰을 제안해서 심플한 테스크 뿐만 아니라 다양한 자연어 처리의 테스크를 처리할 수 있는 통합된 모델을 제안함
+ Text & Position Embed를 더한 후 self-attention 블록을 12개를 쌓아놓음
+ Text-prediction는 첫단어부터 다음단어를 예측하는 language 처리
+ 심플한 렝귀지 모델 테스크 뿐만이 아니라 다수의 문장이 있더라도 모델의 큰 변형이 없이 활용될 수 있도록 학습의 프레임워크를 제시함
+ 문장 레벨의 감정 분류라는 테스크를 수행하기 위해 [Start 토큰 > 문장 > Extract 토큰]을 시퀀스를 넣고 최종적으로 나온 Extract 토큰을 Ouput layer에 넣고 문장을 분류함

![캡처](https://user-images.githubusercontent.com/44515744/108448085-6a62e080-72a4-11eb-9273-46405bb39bfe.PNG)

+ 다음 단어를 예측하는 테스크를 활용하는 모델을 활용함 (pre-Prediction), 메인 테스크는 기존의 학습된 모델을 통해 학습하며 기존에 학습한 모델에서는 상대적으로 learning rate를 적게줘서 큰 변화가 발생하지않게함
+ Pre-Training은 별도의 레이블이 필요하지 않아서 많은 데이터를 이용해서 학습할 수 있음, 메인 Task인 Classifier은 레이블이 필요하고 소량의 데이터 밖에 없기 때문에 pre-training을 한 모델을 사용(전이학습)함
+ Text Prediction : 
+ Task Classfier : 

#### 3. BERT
+ 현재까지 가장 넓게 쓰이는 Pre-Training 모델임
+ 기존에는 LSTM 모델을 활용해서 Pre-Training을 하는 모델도 있었음
+ GPT-1의 경우에는 Standard한 language 모델을 사용했음 (문장의 왼쪽부터 하나씩 예측, 문맥을 이해 X)

#### 4. Pre-training Tasks in BERT
+ 각각의 단어에 대해 일정한 확률로 특정 단어를 MASK라는 단어로 치환하고 단어가 무엇인지를 맞추게하는 형태로 학습을 진행
+ 마스크를 처리하는 비율을 하이퍼 파라미터로 설정이 가능
+ BERT에서는 15%가 가장 적정한 수준으로 파악
    + 마스크가 적을 경우 : 학습 비용이 높아짐
    + 마스크가 많을 경우 : 문장을 제대로 포착하기 위한 정보가 부족

![캡처](https://user-images.githubusercontent.com/44515744/108450441-91bbac80-72a8-11eb-96e6-af3b9420080d.PNG)

+ 마스크로 치환한 다면 15개 중 12(80%)개는 [MASK]로 치환하고 나머지 1.5(10%)을 랜덤한 단어로 바꿔서 단어가 Regular한 단어더라도 원래 있어야할 단어로 잘 복원할 수 있도록 문제의 난이도를 높여줄 수 있음
+ 나머지 1.5(10%)는 맞춰야할 단어로 예측을 했을때 원래 단어와 동일해야하도록 알려주는 용도로 쓰임

#### 5. Pre-training Tasks in BERT: Masked Language Mode
+ [CLS] 토큰은 두개의 연속된 문장이 시퀀스에 있을 경우, 이것이 내용상 이어진 문장인지에 대한 식별하는 용도로 활용
+ 입력 시퀀스를 넣을때 워드를 쪼갠 서브 워드의 단위로 임베딩하여 입력벡터로 넣어줌
+ positional embedding도 학습되도록 설정
+ [CLS], [SEP] 토큰이 추가
+ BERT : MASK를 활용해서 모두를 볼 수 있게하는 Encoder에서 사용하는 self attention 모듈을 사용
+ GPT : Decoder에서 사용하는 Masked Multi Self Attention을 사용함

#### 6. BERT: Input Representation
+ Segment Embedding : 시퀀스에서 여러 문장이 들어올 경우, 단어의 위치뿐만 아니라 문장 위치라는 개념이 추가적으로 필요함, segment Embedding이라는 추가적인 정보를 위한 문장레벨의 position 정보를 Token의 임베딩 더해줌 

![캡처](https://user-images.githubusercontent.com/44515744/108452897-b580f180-72ac-11eb-9239-889f49cd879c.PNG)

#### 7. SQuAD 2.0
+ 스탠포드 올려놓은 질문 응답에 대한 데이터 세트 - https://rajpurkar.github.io/SQuAD-explorer/
+ 데이터 셋을 내려받아 language 모델 학습 및 평가를 할 수 있음

### (10강) Advanced Self-supervised Pre-training Models
#### 1. GPT-2
+ transformer를 더 많이 사용하고 왼쪽부터 생성하는 Language Model을 사용
+ 굉장히 많은 데이터셋과 그 중에서 높은 수준의 Data를 활용함
    + Reddit이라는 질의 응답 플랫폼을 활용함 (좋아요가 많은 응답에 있는 외부링크에 가서 다큐먼트를 활용)
+ 주어진 첫 문장이 있다면, 그럴싸하게 이어서 글을 작성할 수 있음
+ 서브 워드를 활용함(Byte pair encoding)
+ 대화형의 질의응답이 있을때, 테스크를 수행하기 위해서는 
+ 레이어가 올라갈 수록 레이어의 initializing되는 값을 더 작게 만듦(위쪽에서의 선형 변환이 더 작게 일어남)
+ 모든 테스크들은 다 질의형답 형태로 바꿀 수 있다에 입각해서 질의 응답이 아니고 특정 문장에 대해서 다음 문장을 출력하는 형태로 진행
+ (TL;DR:)이라는게 나오면 앞쪽의 내용의 요약해주는 기능


#### 2. GPT-3
+ 기존의 GPT-2의 모델 사이즈에 비해 비교할수 없을 정도로 많은 파라미터를 쓸 수 있도록 attention block을 더 쌓은 것, 큰 배치사이즈로 학습하여 학습의 성능을 높임
+ Zero-shot : 번역에 대한 데이터를 안줬음에도 불구하고 내용을 번역함
+ One-shot : 불어로 번영된 예시를 주고 단어를 줘서 번역에 대한 예측을 하게함
+ Few-shot : 예시를 여러개 줬을때 더 높은 성능을 냄

![캡처](https://user-images.githubusercontent.com/44515744/108458675-722c8000-72b8-11eb-8524-a38379ca2ce4.PNG)

#### 3. ALBERT : Lite BERT
+ 모델의 사이즈는 줄이고 학습사이즈도 빠르게 만드는 형태의 self-supervised Learning의 Task를 제안함
+ 디멘션이 작으면 정보를 담을 수 있는 공간이 줄어들고, 디멘션이 크면 모델 사이즈와 연산량도 많이 증가함  
+ 첫번재 레이어의 워딩 임베딩 레이어는 위쪽에 있는 히든 스테이트 레이어에 비해서 담겨있는 정보량이 작음
+ 워딩 임베딩 벡터의 사이즈를 줄이는 방법으로 모델을 제시함
+ 히든 스테이트 디멘션과 차원 수를 맞추기 위해 선형변환을 위한 레이어를 하나더 추가함
+ 12개 각각에 존재하는 Wq Wk Wv를 공유되는 파라미터로 구성하면 어쩔지를 생각하고 계산함
+ feed-forward network parameter만 share했을때
+ attention parameter를 share 했을때
+ 모두 share했을때 성능을 테스트함
+ 두가지 모두를 share했을 때 성능의 하락폭이 낮지 않았음

#### 4. Sentence Order Prediction
+ Next Sentence Prediction의 실효성이 적음, 이것을 변형하여 
+ 기존에는 동일한 문서에서 추출한 서로 다른 문장을 Next Sentence가 아니라고 학습을 시켰음 
    + 이 경우에는 내용이 많이 겹치지 않을 가능성이 높음
+ Nagative Samples를 활용해서 학습함

#### 5. ELECTRA
+ BERT에서 사용한 Mask Language 모델, GPT에서 사용한 다음 단어를 예측하는 Standard한 형태
+ Language Model을 통해 단어를 복원해주는 Generator 모델을 둔 다음
+ 예측된 단어인지 Discriminator를 통해 단어별로 예측하는 두번째 모델을 갖고 있음 (GAN과 유사)
+ Generator는 BERT 모델로 생각할 수 있음
+ genrative adversial network(GAN) 모델을 사용해서 Pre-train을 제안하는 것

![캡처](https://user-images.githubusercontent.com/44515744/108462607-3b5a6800-72c0-11eb-818c-f6ccef01e39f.PNG)

#### 6. Light-weight Models
+ 경량화 모델은 비대해진 모델을 적은 Layer와 파라미터를 갖고있게하는 모델
+ 클라우드 서버나 고성능의 리소스를 사용하지 않더라도 빠르게 학습을 할 수 있음
+ DistillBERT : Teacher 모델은 Student를 가르침 ( Student는 작고 경량화된 모델 )
    + Teacher Model의 확률 분포를 Teacher Model의 확률분포로 사용함 ( 최대한 잘 묘사할 수 있도록 학습 )
+ TinyBERT : Teacher 모델의 Target Distribution을 ground truth로 학습하는 하는 것뿐만 아니라, 임베딩 레이어 W_k, W_v, W_q 등 결과의 임베딩 벡터 모두 닮도록 Student가 학습하는 것
+ Teacher Model의 Vector가 주어지면 Student Model의 Vector가 닮아지도록 MSE 학습을 진행함
    + 그러나 차원이 다르므로 유사해지게 하기위해 둘 사이에 full-connected Layer를 하나를 둬서 차원을 일치시키고 학습시킴

#### 7. Fusing Knowledge Graph into Language Model
+ Knowledg Graph : 개념, 개체를 잘 정의하고 개체 간의 관계를 정리한 것이 Knowledge Graph라고 함