## [DAY 15] Generative model
### [DLBasic] Generative Models 1
> 출처 - 스탠포드대 자료 : https://deepgenerativemodel.github.io
#### 1. generative model 생성 모델을 학습하는 것?
+ 강아지를 샘플링 할 수 있는 모델
    + Generation : 강아지와 같은 이미지를 만들어 낼 수 있음 (Sampling)
    + Denstiy estimation : x가 들어왔을 때 확률 값하나가 튀어나와서 이미지가 강아지 같은지 고양이 같은지를 분류 (anomaly dection)
        + 이러한 모델을 explicit models라고 함 - 확률 값을 얻어낼 수 있음
        + 생성을 위한 것은 implicit models 모델이라고 함
    + Unsupervised representation learning
        + 강아지 귀가 2개 꼬리가 있고, 특성들이 있다 -> 이런 것을 Feature Learning이라고 하는데 generative model 이를 할 수 있다.

#### 2. Basic Discrete Distributions
+ Bernoulli distribution (coin flip)
    + 0 또는 1 ( 동전 앞 또는 뒤 )
    + 이것을 표현하는 분포는 숫자가 1개 필요함
        + 앞이 나올 확률이 p면 아닐 확률이 1-p

+ Categorical distribution (m-sided dice)
    + D = {1,...,m}
    + 총 m-1개의 파라미터가 필요함
    + m-1인 이유는 하나의 파라미터는 1에서 나머지 파라미터를 빼줘서 만들 수 있음

+ Example
    + RGB
        + RGB 이미지 하나를 표현하는데 0~255 개의 파라미터를 갖고 있음
        + RGB는 256 x 256 x 256개의 컬러를 표현할 수 있음
        + 이를 나타내기 위해서는 255 x 255 x 255개의 파라미터가 필요함
    + binary image
        + fixel이 n개 이면 2^n개의 경우의 수가 존재함
        + 이를 위해서는 2^n-1개의 파라미터가 필요함

#### 3. Structure Through Independence
+ 픽셀이 독립적이라고 가정함
    + 각각의 픽셀을 표현하는 n개의 파라미터만 필요함
+ Conditional Independence
    + 파라미터가 줄어들어서 좋지만 표현할 수 있는 이미지가 적어짐
    + 중간에 있는 무언가를 만들기 위한 세가지 중요한 규칙이 있음
    + Chain rule
        + n의 joint distribution을 n개의 conditional distribution으로 변경, 항상 만족함
        + 어떤 가정을 하지않고 수식만 변경했으므로 Fully dependent model과 같은 파라미터를 사용
        + P(x_{1}, ..., x_{n}) = 1 + 2 + 2^2 + ... + 2^{n-1} = 2^{n}-1개의 파라미터가 필요

    ![캡처](https://user-images.githubusercontent.com/44515744/106977241-492acc00-679d-11eb-9e59-811c4657b7c1.JPG)

    + Bayes' rule

    ![캡처](https://user-images.githubusercontent.com/44515744/106977372-960ea280-679d-11eb-8ebe-24e31fdfe963.JPG)

    + Conditional independence
        + 가정, z가 주어졌을 때 x,y가 independent하다.
        + chain rule에서 나온 conditional distribution의 뒷단에 있는 condition을 날려주는 효과
        
    ![캡처](https://user-images.githubusercontent.com/44515744/106977423-ab83cc80-679d-11eb-81fe-3af78a47b36c.JPG)

    + Markov assumption : i+1 번째 픽셀은 i 픽셀과만 dependent하고  1~i-1 픽셀과는 indepentdent함
        + chain rule을 간단하게 만들 수 있음
        + 2n-1개의 파라미터만 필요하게 됨
        + joint distribution을 쪼개고 Markov assumption을 가했더니 파라미터를 줄일 수 있음
        + 이러한 방법을 Auto-regressive Models이라고 함
    
    ![캡처](https://user-images.githubusercontent.com/44515744/106978952-a2e0c580-67a0-11eb-9679-85808d19bfa8.JPG)

#### 4. Auto-regressive Model
+ 28 x 28 이미지의 픽셀이 존재한다면, p(x) = p(x_{1},...,x_{784})를 구하는 것이 목표
+ 사용하는 chain rule을 사용해서 joint distribution을 나눔
    ![캡처](https://user-images.githubusercontent.com/44515744/106979808-6c0baf00-67a2-11eb-82cc-0d585be15598.JPG)
+ 하나의 정보가 이전 정보들에 Auto-regressive Model이라고 함
    + i번째 픽셀이 i-1에만 dependent한 것도 Auto-regressive Model이지만, i번째 픽셀이 1까지 dependent한 것도 Auto-regressive Model임
    + 이전 n개를 고려한걸 AR-N모델, 1개만 고려한 것을 AR-1 모델이라고 함
+ 이미지 도메인을 Auto-regressive Model로 활용하기 위해서는 ordering이 필요함
    + 순서를 매기는 것은 명확하지 않음, 순서를 매기는 것에 따라 성능과 방법론이 달라질 수 있음

#### 5. NADE : Neural Autoregressive Density Estimator
+ 입력 w가 점점 커지게됨 
    + 1번째 픽셀을 만드는데는 아무것도 필요없고(weight, bias만 필요)
    + 3번째 픽셀을 만드는데는 2개의 입력을 받는 weight가 필요
    + 100번째 픽셀을 만드는데는 99개의 이전 입력들을 받을 수 있는 뉴럴 네트워크가 필요

![캡처](https://user-images.githubusercontent.com/44515744/106980695-00c2dc80-67a4-11eb-9eae-760b9b0442a0.JPG)

![캡처](https://user-images.githubusercontent.com/44515744/106980730-10dabc00-67a4-11eb-9c96-c247e9d79c5c.JPG)

+ NADE는 단순히 제네레이션 뿐만 아니라 임의의 784개의 벡터가 주어지면 확률로 계산이 가능함
+ 첫번째 픽셀에 대한 확률분포를 알고 있고, 첫번째 픽셀이 주어졌을때 두번째 픽셀에 대한 확률분포를 알고있으므로, 각자를 집어넣어서 p(x_{i}|x_{1:i-1})을 계산할 수 있음
    + implicit model은 generation만 할 수 있음
+ continuous output일 때는 a mixture of Gaussian Model을 사용함

#### 6. Pixel RNN
+ 이미지에 있는 픽셀들을 만들어내고 싶은 것
+ RNNs를 Auto-regressive model을 만들기 위해 사용할 수 있음

![캡처](https://user-images.githubusercontent.com/44515744/106981634-b0e51500-67a5-11eb-8a71-dd18dd4a5f15.JPG)

+ ordering 방법에 따라 두 가지 알고리즘이 존재
    + Row LSTM : i번째 픽셀을 만들때 위쪽에 있는 정보를 활용
    + Diagonal BiLSTM : 이전 정보들을 전부다 활용

![캡처](https://user-images.githubusercontent.com/44515744/106982028-8cd60380-67a6-11eb-8063-b1eb44ea7bfd.JPG)

#### 7. Latent Variable Models
+ autoencoder은 generative model이 아니다.
+ Variational Auto-encoder
    + Variational inference(VI)
        +  최적화시킬 수 있는 어떤 걸로 근사하겠다가 목표, 관심있는 posterior distribution에 잘 근사할 수 있는 Variational distribution을 찾는 것을 Variational inference라고함
        + posterior distribution : 사후 확률
        + Variational distribution : 가변 확률
        + 근사하기 위해서는 loss function이 필요한데 KL divergence라는 매트릭을 활용함

#### 8. ELBO (Evidence lower bound)
+ 뭔지도 모르는 posterior을 근사할 수 있는 Variational distribution을 찾는 것은 어불성설
+ 가능하게 해주는 Variational distribution에 있는 ELBO 트릭임
    + posterior distribution, Variational distribution 사이의 KL divergence를 줄이는게 목적
    + ELBO를 계산해서 키움으로써 반대 급부로 원하는 Objective를 원하고자 하는 것
    + 아래 수식이 exact하기 때문에 임의의 posterior distribution와 Variational distribution의 거리를 ELBO를 Maximizing함으로 써 줄일 수 있음

![캡처](https://user-images.githubusercontent.com/44515744/106983414-34543580-67a9-11eb-88e5-4e2868a9c716.JPG)

#### 9. GAN (Generative Adversarial Network)
+ 도둑이 Generation하고 싶어하는데, 위조 지폐를 잘 분별하는 경찰이 존재
+ 경찰은 더 잘 위조지폐를 구분하게됨, 이를 반복해서 Generator의 성능이 좋아지게 됨

![캡처](https://user-images.githubusercontent.com/44515744/106984783-8f872780-67ab-11eb-8485-13b647b47e6e.JPG)

+ GAN은 generator와 discriminator가 존재하고 이 두가지를 학습시킨 다는 것이 가장 큰 특징
+ Minimax Game : 한쪽은 높이고 싶어하고 한쪽은 낮추고 싶어하는 것

![캡처](https://user-images.githubusercontent.com/44515744/106985364-aa0dd080-67ac-11eb-876f-89a4c810fe54.JPG)

+ Puzzle-GAN : 이미지 안에 써페치들이 들어가면 이것을 통해서 원래이미지를 복원하는 것
+ CycleGAN : 이미지 사이의 도메인을 변경함
    + Cycle-consistency loss를 활용함
+ Star-GAN : 인풋이 있으면 이미지를 변경함
    + 이미지를 컨트롤해서 변형됨
+ Progressive-GAN : 고차원의 이미지를 만들 수 있는 방법론
    + 4 by 4에서 1024 by 1024까지 이미지로 키워냄