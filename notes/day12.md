## [DAY 12] 최적화
### [DLBasic] Optimization
#### 1. Gradient Descent
+ 1차 미분한 값을 사용해서 반복적으로 최적화함. 로컬 미니멈을 찾는 것이 목적

#### 2. Important Concepts in Optimization
+ Generalization (일반화)
    + 학습 데이터에 대한 Training Error가 0이 된다고 해도 최적화가 안될 수도 있음 (테스트 데이터에서의 성능도 중요)
    + Generalization Performance : 학습 데이터와 테스트 데이터 사이 간의 성능
    + Test Error : 학습에 사용하지 않은 데이터에서 발생한 에러
    + Training Error : 학습에 사용한 데이터에서 발생한 에러
    + Generalization Gap : 학습 데이터와 테스트 데이터 사이의 차이

+ Under-fitting Vs. Over-fitting
    + Under-fitting : 네트워크가 간단하거나 트레이닝이 부족해서 학습 데이터도 잘 못맞추는 것
    + Balanced : 둘 사이에 적절한 구간
    + Over-fitting : 학습 데이터에서는 잘 동작하지만, 테스트 데이터에서는 잘 독장하지 않는 것
+ Cross validation
    + K Fold validation 이라고도 함
    + 트레이닝 데이터와 테스트 데이터를 나눠서 줄때가 많음
    + 학습시킨 모델이 학습에 사용되지 않은 validation data 기준으로 얼마나 잘되는지 보는 것
    + K-1개로 학습을 시키고 나머지 한개로 validation을 해보는 것
    + 뉴럴 네트워크 학습시 많은 하이퍼 파라미터(정하는 값, learning Rate, 어느 loss Function)가 존재함
    + Cross Validation을 해서 최적의 하이퍼 파라미터 셋을 찾고, 이를 고정하고 학습할 때는 모든 데이터를 다 사용함
    + 테스트 데이터를 활용해서 학습을 하면 안됨
+ Bias and Variance
    + Variance : 입력을 넣었을 때 출력이 얼마나 일관적인지에 대한 것
    + Bias : 같은 입력에 대해서 출력이 분산이 되더라도 평균으로 봤을 때는 True 타겟과 가까운 것
    + Bias and Variance Tradeoff : 학습 데이터에 노이즈가 껴있을 때, 이 노이즈가 껴있는 학습 데이터를 minimize하는 것은 3가지로 구분해서 볼 수 있음
        + t : Target, f : 뉴럴 네트워크의 출력값
        + Cost를 Minimize 한다는 것은 bias, Variance, Noise를 Minimize를 한다는 것을 의미(TradeOff 관계)

    ![image](https://user-images.githubusercontent.com/44515744/106540491-61071380-6543-11eb-9cb9-98bd2bd4ba65.png)

+ Bootstrapping
    + 학습 데이터 중 일부만 활용하는 것, 일부를 여러개 사용하면 여러개의 모델을 만들 수 있음
    + 하나의 입력에 대해 각각의 모델이 다른 값을 예측할 수 있음, 이 모델들이 예측하는 값들의 컨센선스를 보고 전체적인 모델의 불확실성을 파악하기 위해서 사용
    + 학습 데이터를 여러개를 만들고, 여러 모델을 만들어서 비교

+ Bagging vs Boosting
    + Single Iteration : 학습 데이터에 대해 하나의 모델을 만드는 것
    + Bagging (Bootstrapping aggregating)
        + 학습 데이터를 다 사용해서 모델 1개르 만드는 게 아니라, 학습 데이터를 여러개로 나눠서 여러 모델을 사용
        + 학습 데이터를 다 사용해서 결과를 내는 것보다 테스트 데이터에 여러 모델을 모두 돌려보고 평균이나 몰팅해서 나온 출력값을 쓰는 것이 더 좋을 떄가 많음 (앙상블)
        + 병렬적으로 동작
    + Boosting
        + 학습 데이터가 100개가 있으면, 모델을 간단하게 만들고 모델을 학습 데이터에 대해서 돌려봄
        + 예측하지 못한 데이터에 대해서만 잘 동작하는 모델을 만든다.
        + 하나 하나(weak runner)의 모델들을 순차적으로 합쳐서 하나의 모델(strong runner)를 만든다.
        + 여러 모델이 순차적으로 동작

#### 3. Practical Gradient Descent Methods
+ Stochastic gradient descent
    + 하나의 샘플을 통해서만 gradient를 계산해서 업데이트 하는 것
+ Mini-batch gradient descent (일반적)
    + 샘플들을 얻어다가 gradient를 계산해서 계속 업데이트 하는 것
+ Batch gradient descent
    + 모든 데이터의 gradient를 계산해서 업데이트 하는 것

#### 4. Batch-size Matters
+ 큰 사이즈의 배치 사이즈를 사용하면 sharp minimizer에 도달하게 됨.
    + sharp minimum은 Generalization 퍼포먼스가 떨어짐 
+ 작은 사이즈의 배치 사이즈를 사용하면 flat minimizer에 도달하게 됨.
    + flat minimum은 Generalization 퍼포먼스가 높음

#### 5. Gradient Descent Methods
+ Stochastic gradient descent(SGD)
    + gradient를 Learning rate만큼 곱해서 가중치에서 뺌
    + Learning rate를 적절하게 잡아주는 것이 어려움
+ Momentum(관성)
    + gradient 값을 참고하는 momentum을 갖고 있음, 모멘텀과 gradient를 합침 accumulation을 활용해서 업데이트 시킨다.
    + 한번 흘러가기 시작한 gradient를 유지하기 때문에 gradient가 잘 왔다갔다해도 학습을 잘 시켜주는 효과가 있음

![캡처](https://user-images.githubusercontent.com/44515744/106542312-ec35d880-6546-11eb-875f-9eaa4cc6ad95.JPG)

+ Nesterov Accelerate Gradient(NAG)
    + Momentum을 사용하나 추가적인 옵션이 추가된 로직
    + Lookahead gradient : A라는 정보가 있으면 그 방향으로 가보고 이 방향에서 gradient를 계산한 것을 가지고 반영함
    + local minimum으로 빠르게 conversion할 수 있음

![캡처](https://user-images.githubusercontent.com/44515744/106543089-69158200-6548-11eb-91be-aa0da867fed3.JPG)

+ Adagrad
    + 앞서 배운 방법들은 gradient를 활용하여 이동시 반영하는 로직이었음
    + 뉴럴 네트워크의 파라미터가 지금까지 얼마나 많이 변했는지에 대한 정보(G)를 갖고 있음
    + 많이 변한 파라미터는 적게, 조금 변한 파라미터를 많이 변화시키는 방법
    + G가 계속 커지기 때문에 가면갈수록 분모가 커져서 학습이 잘안되는 경향이 있음

![캡처](https://user-images.githubusercontent.com/44515744/106543605-65cec600-6549-11eb-8461-0d981e224975.JPG)

+ Adadelta
    + G가 계속 커지는 것을 방지하기 위한 방법론
    + 현재 timestep에 대한 시간에 따른 변화를 보는 것
    + window size를 100으로 두면 이전 100개 동안의 G라는 정보를 들고 있어야함.
    + 100억개의 파라미터가 있을 경우에는 이를 100개씩 들고있어야해서 메모리가 부족해짐

![캡처](https://user-images.githubusercontent.com/44515744/106543910-00c7a000-654a-11eb-9a03-a966f3d5471b.JPG)

+ RMSprop
    + Adadelta와 유사하나 에타라는 step 사이즈를 분자에 추가함

![캡처](https://user-images.githubusercontent.com/44515744/106544060-48e6c280-654a-11eb-9ce4-4db92ad05333.JPG)

 + Adam 
    + 가장 무난하게 사용하는 방법론
    + EMA of gradient squares와 이전에 gradient 정보에 해당하는 모멘텀 2개를 잘 합친 것

![캡처](https://user-images.githubusercontent.com/44515744/106544524-05d91f00-654b-11eb-8609-69ee6adecbe8.JPG)


#### 6. Regularization
+ 학습에 대한 규제를 둬서 테스트 데이터에도 잘 동작할 수 있도록 만들어주는 것
+ Early Stopping
    + validation data를 활용하는 방식으로 Iteration이 돌아가면서 학습하는 중간에 test error가 validation error와의 gap이 커지는 순간에 학습을 종료함
+ Parameter norm penaly(Weight DK)
    + 뉴럴 네트워크의 파라미터가 너무 커지지 않게하는 것
    + 학습 데이터의 크기를 줄이는 것으로 데이터를 부드럽게 한다고도 함
+ Data augmentation
    + 주어진 데이터를 지지고 볶아서 데이터 수를 늘리는 방법
    + 찌그러지거나 회전시켜서 이미지를 변환
    + 이미지의 라벨이 바뀌지 않은 한도 내에서 변환하는 방법
+ Noise robustness
    + 입력 데이터에 노이즈를 추가하는 것
    + Weight에도 노이즈를 추가하면 학습이 더 잘되는 효과가 있음
+ Label smoothing
    + 데이터 두개를 뽑아서 섞어주는 것
    + 분류 문제를 푸는 것은 decision boundary를 찾는것
    + Label smoothing을 하면 dicision boundary가 부드럽게 됨
    + Mixup : 두 이미지를 합침(블랜딩)
    + CutOut : 이미지를 일부분 자름
    + CutMix : 두 이미지의 일부분을 섞음
+ Dropout
    + 뉴럴 네트워크의 weight를 0으로 바꾸는 것
    + 각각의 뉴런들이 로버스트한 피쳐를 잡을 수 있다고함
+ Batch Normalization
    + 내가 적용하고자 하는 레이어의 statistics를 정규화하는 것
    + 레이어가 깊게 쌓아지게되면 성능이 늘어남

#### 7. 실습
+ ADAM이 잘맞추는 건 Momentum과 Adaptive Learning Rate를 잘 써야한다는 것을 의미함
+ SGD는 너무 많은 Iteration이 있어야함
+ Momentum 이전 gradient를 현재도 반영시키기 때문에 데이터를 한번에 더 많이 보는 효과
+ Momentum에 Learning Rate를 합치기 때문에 파라미터의 Learning Rate가 가변적으로 빠르게 맞출 수 있음
+ 실습 주소 : https://colab.research.google.com/drive/1kZc5_hekETL-DlniCE12O3tf2BNpp2K8?usp=sharing

### [AI Math 9강] CNN 첫걸음
> Convolution 연산과 다양한 차원에서의 연산방법에 대한 내용
#### 1. Convolution 연산
+ 다층신경망(MLP)
    + 지금까지 배운 다층신경망(MLP)는 각 뉴런들이 선형모델과 활성함수로 모두 연결된 fully connected 구조
    + 입력 X에 대한 가중치 행 W를 내적하여 잠재변수  h를 구할 수 있음
    + i가 바뀌게 되면 가중치 행렬의 행도 바뀌기 때문에 가중치 행렬의 사이즈가 커지게 됨
+ Convolution 연산
    + 커널(kernel)을 입력벡터 상에서 움직여 가면서 선형모델과 합성함수가 적용되는 구조
    + 모든 i에 대해 적용되는 커널은 V로 같고 커널의 사이즈만큼 x 상에서 이동하면서 적용
    + 활성화 함수를 제외한 Convolution 연산도 선형변환에 속함

![캡처](https://user-images.githubusercontent.com/44515744/106567962-f4593c80-6575-11eb-8e5c-e6622b778735.JPG)

+ Convolution 연산의 수학적인 의미는 신호(signal)를 커널을 이용해 국소적으로 증폭 또는 감소시켜서 정보를 추출 또는 필터링하는 것
+ 커널은 정의역 내에서 움직여도 변하지 않고(translation invariant) 주어진 신호에 국소적(local)으로 적용합니다.

#### 2. 영상처리에서 Convolution
+ 커널의 종류에 따라 이미지에 다양한 효과를 줄 수 있음

#### 3. 다양한 차원에서의 Convolution
+ Convolution 연산은 1차원뿐만 아니라 다양한 차원에서 계산이 가능
+ 데이터의 성격에 따라 사용하는 커널이 달라짐
    + 1D-conv : 음성, 텍스트
    + 2D-conv : 흑백 이미지
    + 3D-conv : 컬러 이미지

#### 4. 2차원 Convolution 연산
+ 2D-Conv 연산은 커널(kernel)을 입력벡터 상에서 움직여가면서 선형모델과 합성함수가 적용되는 구조
    + 성분곱을 계산함
    + 커널의 위치는 고정되어 있고, 입력

![image](https://user-images.githubusercontent.com/44515744/106569631-22d81700-6578-11eb-9c45-abbef8b02f4c.png)

+ 입력 크기를 (H,W), 커널 크기를(K_H, K_H), 출력 크기를 (O_H, O_W)라 하면 출력의 크기는 아래와 같음
+ 가령 28x28 입력을 3x3 커널로 2D-Conv 연산을 하면 26x26이 된다.

![캡처](https://user-images.githubusercontent.com/44515744/106570158-de994680-6578-11eb-8647-75ceedc0097d.JPG)

+ 채널이 여러개인 2차원 입력의 경우 2차원 Convolution을 채널 개수만큼 적용
+ 채널이 여러개인 경우 커널의 채널 수와 입력의 채널수가 같아야 한다.
+ 텐서를 직육면체 블록으로 이해하면 좀 더 이해하기 쉬움
+ 출력을 여러개 만들고 싶을 경우에는 커널도 여러개를 생성하면 된다.

#### 5. Convolution 연산의 역전파 이해하기
+ Convolution 연산은 커널이 모든 입력데이터에 공통으로 적용되기 때문에 역전파를 계산할 때도 convolution 연산이 나오게 됨
+ 각 커널에 들어오는 모든 그레디언트를 더하면 결국 convolution 연산과 같다.

![캡처](https://user-images.githubusercontent.com/44515744/106572425-beb75200-657b-11eb-971c-c675df92dd5f.JPG)
