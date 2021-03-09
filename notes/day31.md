## [DAY 32]
### (3강) Image classification 2
#### 1. Going deeper with convolutions
+ 네트워크의 깊이가 깊어질 수록 receptive fields가 더 커짐
+ 더 많은 수용하고 비선형이 더 많아짐
+ 깊어질 수록 Gradient vanishing / exploding이 발생하고, 계산 복잡도가 올라가서 더 큰 메모리가 필요함, Degradation 문제가 발생하게 됨

#### 2. GoogLeNet - [Szegedy et al., CVPR 2015]
+ GoogLeNet은 Inception Module을 제안함
    + 하나의 레이어에서 다양한 크기의 컨볼루션 필터를 사용해서 여러 측면으로 Activation을 관찰 ( Depth가 아닌 Width 수평확장)
    + 결과 엑티베이션 맵들을 채널축으로 Concatenation해서 다음 블록으로 넘겨줌
    + 계산 복잡도와 필요한 용량이 많이 커짐 ( 채널 디멘션을 1x1 Convolitions을 사용해서 줄여줌 - 압축 )
    + Max Polling에서도 1x1 convolutuions을 통해 채널 디멘션을 바꿈

    ![캡처](https://user-images.githubusercontent.com/44515744/110424655-78a75e00-80e6-11eb-97ad-353718e9f4f0.PNG)

    ![캡처](https://user-images.githubusercontent.com/44515744/110424776-adb3b080-80e6-11eb-8064-461fc927a736.PNG)

+ 기울기 소실 문제를 해결하기 위해 Auxiliary classifiers를 사용함 ( 중간에 loss를 저장 )
    + 역전파시 주사기처럼 네트워크에 기울기를 전달해주는 역할을 함
    + 학습시에만 활용함 

    ![캡처](https://user-images.githubusercontent.com/44515744/110424946-f0758880-80e6-11eb-92b7-2f35722ec2e6.PNG)

#### 3. ResNet - [He et al., CVPR 2016]
+ 처음으로 인간 수준의 인지능력을 뛰어넘음
+ 기존 네트워크들의 Depth가 성능에 중요하다는 것은 증명되었으나 기울기 관련 문제로 해결 못하던 상황
+ 학습이 진행될 수록 Training Error가 줄다가 멈춤, Test Error도 동일하게 줄어들게 됨
    + 네트워크를 깊게 쌓을 수 있는 문제가 Overfitting이 문제가 아니라 degrade gradient 문제
+ 자기 자신을 보존한테 잔차를 학습하는 형태로 진행
    + Shortcut Connection

    ![캡처](https://user-images.githubusercontent.com/44515744/110426013-b0afa080-80e8-11eb-89e0-5800154f3c52.PNG)

+ Shortcut Connection은 2^n으로 gradient가 지나갈 수 있는 input, output을 생성함 

    ![캡처](https://user-images.githubusercontent.com/44515744/110426248-1a2faf00-80e9-11eb-9bf8-4b8845d79969.PNG)

#### 4. 기타 모델
+ DenseNet - [Huang et al., CVPR 2017] : Skip Connection이 아니라 채널축으로 Concatenation을 진행함 이전에 정보들도 넘겨주는 형태
    + 상위 레이어에서도 하위레이어의 특징을 재참조할 수 있게함
    + resnet과 달리 더하기가 아닌 Concatentaion을 함(Feature의 정보를 그대로 보존함)

    ![캡처](https://user-images.githubusercontent.com/44515744/110427973-e43ffa00-80eb-11eb-96e7-694325f2f9cf.PNG)

+ SENet - [He et al., CVPR 2018] : 현재 주어진 Activation간의 관계를 모델링하고 중요도를 파악하고 이를 Attention하는 방법
    + Squeeze : 각 채널의 공간정보를 없애고 분포를 봄(공간을 1로 만들고 채널의 평균 정보를 포함)
    + Excitation : 채널 간의 연관성을 고려함 W를 고쳐서 채널간의 연관성을 파악하여 스코어링 함
    
    ![캡처](https://user-images.githubusercontent.com/44515744/110428062-12253e80-80ec-11eb-9e1b-fc7bd0642efc.PNG)

+ EfficientNet - [Tan and Le, ICML 2019] : 기존 모델은 포화상태가 빨리옴, 3개의 팩터를 적절한 비율로 잘 조절을해서 한스텝 한스텝 스케일링을 하여 성능을 향상함 ( Compound Scalining )

    ![캡처](https://user-images.githubusercontent.com/44515744/110428510-b5765380-80ec-11eb-92b6-cf510f769da1.PNG)

+ Deformable convolution - [Dai et al., ICCV 2017] : 사물의 크기와 모양은 제각각이므로 사물의 모양 을 기준으로 컨볼루션 필드의 위치를 바꿔가며 사용함

    ![캡처](https://user-images.githubusercontent.com/44515744/110428675-038b5700-80ed-11eb-89d7-cd6cfba5d174.PNG)

+ 종합 - [Canziani et al., CVPR 2016]
    + AlexNet : 간단한 컴퓨팅으로 구현 가능하나 많은 메모리와 낮은 정확도를 가짐
    + VGGNet : 많은 메모리와, 많은 연산이 필요하나 간단하게 구현이 가능
    + ResNet : 깊은 레이어와 residual blocks
    + (AlexNet, VGG, ResNet)과 비교해서 GoogLeNet은 가장 효율적인 모델이지만 사용하기 복잡함
        + 따라서 VGGNet과 ResNet이 여러 모델의 backbone으로 많이 사용되고 있음

    ![캡처](https://user-images.githubusercontent.com/44515744/110430230-65e55700-80ef-11eb-87ed-6b4a01c1166f.PNG)

### (4강) Semantic segmentation
#### 1. Semantic segmentation
+ 이미지 분류를 픽셀단위로 진행하는 것, 영상 속에 있는 물체의 마스크를 생성하게 됨
+ 같은 클래스지만 서로 다른 물체를 구분하진 않음 ( 사람이 여러명 있으면 같은 색으로 구분 )
+ 메디컬 이미지나 자율 주행에 사용됨, 컴퓨터를 이용하여 사진을 편집하는 기술 등에도 활용이 가능

#### 2. Fully Convolutional Networks (FCN)
+ 첫번째 semantic segmentation 
+ Fully connected layer : 공간 정보를 고려하지 않고 이미지가 주어지면 결과를 벡터로 출력하는 디멘션
+ Fully Convolutional layer : 입력이 이미지이고 출력도 이미지로 나옴
    + 하나의 feature vector를 분류하기 위해서는 채널을 기준으로 벡터들을 뽑은다음에 이를 선형변환함
    + 필터의 개수만큼 feature map을 생성할 수 있음

    ![캡처](https://user-images.githubusercontent.com/44515744/110435187-09396a80-80f6-11eb-8fec-d1b1f27a37fd.PNG)

+ 작은 스코어맵을 얻게됨, 인풋이 크면 아웃풋이 작게됨 ( stride와 pooling layer를 통해서 데이터가 압축됨 )

#### 3. Up Sampleing
+ FCN은 원본의 해상도를 유지하기 위해 Upsampleing을 사용하게 됨, 일단을 작게만든다음 Up Sampling을 통해서 크기를 맞춰줌
    + Transposed Convolution
        + input에 필터만큼 크기가 커지고 중첩된 부분은 더하기가 되어 결과가 나옴
        + 중첩된 부분이 계속 더해지다보니 같은 내용이 오버랩되어 나오는 문제점이 존재
        + kernel 사이즈와 stride 파라미터를 잘 사용해서 중첩이 발생하지 않도록 튜닝을 해야함

        ![캡처](https://user-images.githubusercontent.com/44515744/110435789-c926b780-80f6-11eb-92aa-a3aeaa80eb37.PNG)

        + 쉽게 구현하고 성능도 좋은 대안이 Up Sampling과 Convolution을 같이 사용하는 것
            + 학습 가능한 업샘플링을 하나의 레이어로 한방에 처리한 것
            + Up Sampling convolution은 영상처리에 많이쓰이는 interpolation을 먼저 적용하고 Nearest-neighbor(NN), Bilinear 등을 사용하고 컨볼루션을 적용함

        ![캡처](https://user-images.githubusercontent.com/44515744/110438585-ddb87f00-80f9-11eb-8a27-b40e2909076f.PNG)
    
    + 낮은 레이어에서는 Receptive size가 작기 때문에 국지적이고 작은 디테일을 보고 작은 차이에도 민감함
    + 높은 레이어에서는 큰 Receptive Field를 갖고 전반적이고 의미론적인 결과를 갖음
    + Semantic Segmentation에서 필요한 건 위 두 개의 내용
    + 이 높은 층과 중간층에 있는 내용들을 업샘플링해서 가져온 다음 콘케트네이션을 하여 합침

    ![캡처](https://user-images.githubusercontent.com/44515744/110439023-4c95d800-80fa-11eb-9fb9-496644a40c81.PNG)

#### 4. U-Net - [Ronneberger et al., MICCAI 2015]
+ Neural NEt 모델 중 영상과 비슷한 크기의 출력, Segment Detection 등 영상의 일부를 자세히 봐야되는 기술들에서는 U-Net에 기원을 많이 두고 있음
+ Fully Convolutional Networks임
+ Skip connection을 통해 높은 층과 낮은 층의 feature의 특징을 잘 결합하는 방법을 제시함
+ U의 앞쪽 부분에서는 해상도를 낮추고 pooling을 해서 채널을 작게함 기존과 동일함
+ 단계별로 Activation 맵의 해상도와 채널 사이즈를 올려줌, 대칭으로 대응되는 레이어와 매칭하여 합칠 수 있게함
+ 공간적으로 높은 해상도와 입력이 경계선이나 공간적으로 중요한 정보들을 뒷쪽 레이어에 바로 전달함

![캡처](https://user-images.githubusercontent.com/44515744/110439652-0e4ce880-80fb-11eb-86d6-2dfb5d7ff77b.PNG)

+ feature map의 사이즈가 홀수개 인경우에는??
    + 피쳐 사이즈가 7x7가 다운 샘플링이 되면 버림이 되어 3x3이 됨
    + 3x3을 업샘플링될 경우 6x6이 됨
    + 입력 영상을 넣어줄때 중간에 어떤 레이어에서도 홀수가 나오지 않게 유의해야함

#### 5. DeepLab
+ Conditional Random Fields (CRFs)
    + 후처리로 사용되는 툴 그래프 모델링이나 최적화 등 다양한 배경지식이 필요함
    + 픽셀과 픽셀 사이의 관계를 이어주고 레귤러와 그리드 픽셀 맵을 그래프로 본것
    + 시멘틱에서 결과를 뽑고나면 blur같은 결과가 나오게 됨, 
+ Dilated Convolution
    + Weight 사이를 한칸씩 띄어서 보다 컨볼루션 커널보다 넓은 영역을 고려할 수 있게하고 파라미터 수가 늘어나지 않음
+ Depthwise separable convolution
    + 커널 전체가 채널 전체에 대해서 내적을 해서 하나의 값을 얻는 절차를 둘로 나눔
    + 채널별로 컨볼루션을 해서 값을 뽑고, 채널별로 뽑은 값을 Pointwise Convolution을 사용해서 하나의 값으로 뽑음
        + 두 개의 프로세스로 나누면 표현력을 유지하면서 계산량을 획기적으로 줄일 수 있음

        ![캡처](https://user-images.githubusercontent.com/44515744/110446998-1f99f300-8103-11eb-80ca-95ed454d20db.PNG)

