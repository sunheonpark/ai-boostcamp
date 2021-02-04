## [DAY 13] Convolutional Neural Networks
### [DLBasic] CNN - Convolution은 무엇인가?
#### 1. Convolution이란?
+ 두 개의 함수를 잘 섞어주는 용도로 사용
+ 적용하고자 하는 이미지에 컨볼루션 필터를 찍는 것
+ 이미지와 필터를 성분 곱한 결과가 Output이 됨
+ 2D Convolution을 한다는 것은 해당 컨볼루션의 모양을 이미지에 찍는다는 것
+ 3 by 3 컨볼루션 필터에 값이 모두 1/9이 들어가 있을 경우, 9 by 9 image에 찍히는 3 by 3 영역의 평균이 output이 됨
+ 32 by 32 by 3 Image를 컨볼루션한다고 하더라도 필터는 5 by 5 by 3으로 크기(3)가 동일함
    + output은 28 by 28 by 1이 됨
+ 컨볼루션 필터가 여러개 일 경우, 그 갯수만큼 여러개의 output이 생성됨

#### 2. Stack of Convolutions
+ 32 by 32 by 3 => 28 by 28 by 4 ( 4개의 5 by 5 by 3 필터가 필요)
+ 28 by 28 by 4 => 24 by 24 by 10 ( 10개의 5 by 5 by 4 필터가 필요 )

#### 3. CNN의 구성
+ Convolution layer, Pooling layer 그리고 fully connected layer로 구성
    + Convolution and Pooling layers : feature extraction
    + Fully connected layer : decision making (e.g. classification)
        + 최소화시키는 추세 ( 파라미터 숫자에 종속됨 )
        + 파라미터가 많으면 generalization perfomance가 떨어짐
        + 파라미터를 줄이는 여러 테크닉들을 CNN에서 사용

#### 4. Stride과 Padding
+ Stride : 보폭을 의미, 컨볼루션 필터를 얼마나 자주, dense하게 찍을건지를 정의하는 것
+ Padding : 컬본루션 필터를 찍을 수 없는 가장자리에 값을 채워주는 것
    + 컨볼루션 필터가 5 by 5이면 padding이 2가 필요
    + 컨볼루션 필터가 7 by 7이면 padding이 3이 필요
    + 원하는 출력 값에 맞춰서 Zero padding과 Stride를 줄 수 있음

![캡처](https://user-images.githubusercontent.com/44515744/106684982-d5f15080-660a-11eb-9f4e-da29c6702cdf.JPG)

#### 5. dense Layer
+ dense layer(fully connected layer)의 파라미터 수는
    + input에 있는 파라미터의 개수와 output에 있는 뉴런의 개수를 곱한 것만큼 있음
    + convolution layer에서 dense layer로 넘어갈때 크기가 엄청 달라짐
        + convolution operator와 각각의 커널이 모든 위치에 대해서 동일하게 적용되기 때문
        + 따라서 neural network의 깊이가 깊어지고 파라미터 숫자가 줄어들게 됨

#### 6. 1x1 Convolution
+ Dimension reduction(Channel을 줄일 수 있게됨)
+ 깊이를 늘리면서 파라미터를 줄일 수 있게됨
+ 관련해서 bottleneck architecture가 있음

### [DLBasic] Modern CNN - 1x1 convolution의 중요성
> ILSVRC라는 Visual Recognition Challenge와 대회에서 수상을 했던 5개 Network 들의 주요 아이디어와 구조에 대한 내용
> 네트워크의 뎁스는 깊어지고 네트워크의 파라미터는 줄어들고 성능을 올라감
#### 1. ILSVRC
+ Image Large-Scale Visual Recognition Challenge
    + Classification / Detection / Localization / Segmentation
    + 1000가지의 다른 카테고리가 존재
    + 100만개가 넘는 이미지
    + 트레이닝 셋은 45만 정도가 존재
    + 2015년부터는 오차율이 3.5% 밑으로 줄어듬(사람은 약 5.1% 정도)

#### 2. AlexNet
+ 네트워크가 2개로 나뉘어져 있음 (GPU 성능 부족으로 이를 극복하기 위한 전략)
+ 11 by 11 by 3 필터를 사용함 - 커널이 보는 영역이 커지지만 많은 파라미터가 필요함
+ 5개의 Convolution 레이어와 3개의 dense 레이어로 이뤄짐
+ 활성홤수로 ReLu를 사용함(*핵심)
    + 깊게 쌓았을 때 네트워크를 망칠 수 있는 성질들이 많이 없음
    + 0 보다 작은 값을 0으로 바꿔짐, 0보다 크면 그대로 유지
    + linear 모델들이 갖고 있는 좋은 성질을 갖고 있기 때문에 학습에 용이
    + sigmoid, tanh는 값이 클 수록 slop가 줄어들게 됨(0에 가까워짐) -> Vanishing gradient가 발생
+ 2개의 GPU를 사용
+ Local Response Normalization(LRN) - response가 많으면 몇개를 지워버림, 지금은 잘 사용 X
+ Data augmentation
+ Dropout

#### 3. VGGNet
+ 3 by 3 Convolution Filter만을 사용함(*핵심)
    + Convolution Filter가 크면 한번 찍었을 때 고려되는 input의 크기가 커짐
    + Receptive field : 하나의 Convolution Map 값을 갖기 위해서 고려할 수 있는 입력의 Spatial dimension
    + 사실상 3x3을 두번 거치면 5x5가 됨, receptive field 차원에서는 동일한 결과
        + 3x3을 두번 쌓는게 파라미터를 더 적게 사용할 수있음
        + 3x3 = 294,912 파라미터를 사용
        + 5x5 = 409,600 파라미터를 사용
+ 1 by 1 Convolution Filter도 사용함 (파라미터를 줄이는 목적은 X)
+ Dropout (p=0.5)
+ VGG16, VGG19 (레이어의 개수에 따라 숫자가 다름)

#### 4. GoogLeNet
+ 중간중간 1x1 필터를 잘 사용하면 전체적인 파라미터 숫자를 줄일 수 있음
+ 총 22개의 레이어로 구성됨
+ 네트워크 안에 비슷한 모양의 네트워크가 반복되는 구조 (NiN : network-in-network)
+ Incepton blocks : 여러개가 퍼졌다가 하나로 합쳐지게 됨
    + 3x3 컨볼루션, 5x5 컨볼루션을 하기전에 1x1 컨볼루션을 진행함
    + 파라미터 수를 줄어줌
    + 1x1은 채널 방향으로 디멘션을 줄일 수 있다. 전체적인 파라미터 숫자는 줄이지만 receptive field와 input과 output 채널이 동일함 ( AlexNet보다 파라미터 수가 현저히 적음)

    ![캡처](https://user-images.githubusercontent.com/44515744/106695170-66d12780-661d-11eb-98d0-e14ef98dce8d.JPG)

#### 5 ResNet
+ 파라미터 수가 많으면 Overfitting(트레이닝 에러는 주는데 테스트 에러가 큰것)이 되게 됨
+ identity map이라는 개념을 추가함
    + skip connection : f(x) -> x + f(x)
    + f(x)가 학습하는 것은 차이만 학습하길 원함
+ layer가 많으면 적은 것보다 학습 잘하질 못했는데, identity map을 사용해서 깊게 쌓아도 좋은 성능을 낼 수 있도록 바꿈
+ Convolution 뒤에 Batch Norm을 넣는 구조
+ Bottlenect architecture
    + 3 by 3 전에 1 by 1 convolution을 사용해서 input channel을 줄임

#### 6. DenseNet
+ Convolution에서 나온 결과 값을 더하지 않고 Concatenate를 하게 됨
+ 2배씩 파라미터가 커지게 됨
+ Dense Block
    + 각 레이어가 feature map들을 concatenates함
    + 채널의 수가 계속 증가하게 됨
+ Transition Block
    + Dense Block을 통해 많아진 파라미터를 줄여줌
    + BatchNorm -> 1x1 Conv -> 2x2 AvgPooling
    + Dimension reduction

#### 7. Summary
+ Key takeaways
    + VGG : repeated 3x3 blocks -> receptive field를 늘리는 입장에서는 3x3 필터를 사용
    + GoogLeNet : 1x1 convolution -> 채널수를 줄여서 파라미터를 줄임
    + ResNet : skip-connection -> 네트워크를 깊게 쌓을 수 있음
    + DenseNet : concatenation -> 피쳐맵을 더하는 대신 레이어들을 쌓아놓으면서 좋은 성능을 갖게 됨

### [DLBasic] Computer Vision Applications
#### 1. Semantic Segmentation(Dense ca)
+ 이미지가 있을 때 이미지의 구성 요소들을 픽셀단위로 분류하는 것
+ 이미지의 모든 픽셀이 어떤 라벨에 속해져 있는지 확인
+ 자율 주행에서 활용(바로 앞에 있는게 사람인지 인도인지 차도인지 확인)

#### 2. Fully Convolutional Network
+ 일반적인 네트워크는 이미지가 들어오면 Convolution Layer 다음에 dense Layer를 통과시켜서 output을 출력
+ dense layer를 없애는 방법 output을 convolution layer로 바꾸는 과정(Convolutionalization)
+ flat하고 reshape를 하는 것과 Convolution으로 나타낸 결과가 동일함(우측)
+ 기존의 flat->dense를 하는 방법과 파라미터 사용수에는 변함이 없음

![캡처](https://user-images.githubusercontent.com/44515744/106703004-af441180-662c-11eb-97a6-0b99d7b6a63d.JPG)

+ Fully Convolution Network(FCN)의 가장 큰 특징
    + input dimension에 독립적임
    + input 이미지에 상관없이 네트워크가 돌아감, output이 커지게되면 그거에 비례해서 뒷 단의 Spatial dimension이 커지게됨
    + 더 큰 이미지에 대해서는 기존 네트워크에서는 reshape가 있었기 때문에 할 수 없었음
    + convolution이 가지는 shared  parameter의 성질 때문에 그러함
    + 동작이 히트맵과 같은 효과가 발생하게 됨 ( Spatial dimension이 계속 줄어들게 됨)
    + 이를 통해 분류만 했던 네트워크에서 히트맵을 만들 수 있는 방법이 생김
    + Output Dimension이 즐어들게 된 것을 unsample을 통해서 늘려줘야함

#### 3. Deconvolution (conv transpose)
+ 컨볼루션의 역연산을 하는 방법(완전한 역연산은 아님)
+ 각 성분곱을 합치는 것을 역으로 복원하는 것은 불가능함
+ 파라미터의 숫자와 네트워크 아키텍쳐의 크기를 계산할 때는 역연산이라고 생각하면 편함
+ pixel에 패딩을 많이줘서 결과론 적으로 원하는 크기의 결과물을 생성

![캡처](https://user-images.githubusercontent.com/44515744/106704649-d05a3180-662f-11eb-8944-6ca2b5ae1091.JPG)

#### 4. Detection
이미지 안에서 어느 물체가 어디에 있는지, bounding box를 찾는 것

#### 5. R-CNN
+ 이미지 안에서 Fetch를 많이 뽑는 것
+ 여러개의 region을 뽑아낸 다음, CNN에서 돌아가기 위해 똑같은 크기로 맞춘다 
+ Feature은 AlexNet으로 뽑음
+ 서포트 벡터 머신(Support-vector machine)을 활용해서 이미지를 분류함
+ 이미지마다 conversion Feature map을 뽑기 위해 Alexnet을 2000번 돌려야함
+ bounding box regression : 바운딩 박스를 어떻게 옮기면 좋을지에 대한 내용
+ 많은 이미지 또는 패치를 모두 CNN에 통과시켜야함
+ CPU에서 하나의 이미지를 처리하는데 1분가량이 소모됨

#### 6. SPPNET
+ SPP(spatial pyramid pooling)
+ 이미지 안에서 CNN을 한번만 돌리는 것
+ 이미지에서 바운딩 박스를 뽑고, 뽑힌 바운딩 박스에 해당하는 컨볼루션 피쳐맵의 텐서를 뜯어오는 방식

#### 7. Fast R-CNN
+ input 이미지에서 바운딩 박스를 미리 추출함
+ ROI pooling을 통해 각 리전의 크기를 수정
+ 바운딩 박스 regressor과 label을 찾게됨
+ 뒷단의 RoI feature vector를 통해서, 바운딩 박스 리그레션과 분류했다는 것이 SPPNET과의 차이점

#### 8. Faster R-CNN
+ 바운딩 박스를 봅아내는 region Proposal도 학습을 하는 방법, 이 네트워크를 RPN(Region Proposal Network)라고 부름
+ Faster R-CNN = RPN(Region Proposal Network) + Fast R-CNN

#### 9. Region Proposal Network
+ 이미지에서 특정 영역이 바운딩 박스로써 의미가 있을지 없을지를 찾아내줌
    + anchor boxex : 미리 정해놓은 바운딩 박스의 크기
    + 어떤 크기의 물체들이 있을것 같다라는 것을 미리 알아내고 템플릿을 만들어 놓는 방식
+ RPN의 방식
    + 9 : 9개의 리전 사이즈를 구하는 것
    + 4 : 각각의 리전 사이즈마다 키우고 줄일지에 대한 정보
    + 2 : 바운딩 박스가 쓸모있는지 없는지를 체크하는 값

![캡처](https://user-images.githubusercontent.com/44515744/106706551-3300fc80-6633-11eb-9b96-e9b3f4114bca.JPG)

#### 10. YOLO
+ YOLO(v1)은 faster R-CNN보다 훨씬 빠름
+ 바운딩 박스를 따로 뽑는 과정이 없음
+ 이미지가 들어오면 SxS Grid로 나누게 됨
+ 이 이미지의 중앙이 해당 그리드 안에 들어가면, 그 그리드 셀이 해당 물체에 대한 바운딩 박스와 그 해당물체가 무엇인지 예측
+ 바운딩 박스의 x,y,w,h를 찾아주고 각 박스가 쓸모있는지를 찾아줌
+ 셀의 중점에 obejct가 어느 클래스인지를 파악
+ 위 두정보를 취합하여 결과를 얻게 됨
+ SxSx(B*5+C) size의 tensor를 반환
    + SxS : Number of cells of the grid
    + B*5 : B bounding boxes with offsets (x,y,w,h) and confidence
    + C : Number of classes
