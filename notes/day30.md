## [DAY 31]
### (1강) Image classification 1
#### 1. Overview
+ Artificial Intelligence (AI)
    + 일반적으로 인간의 지능을 필요로하는 역할을 수행하는 컴퓨터 시스템의 이론과 개발 시각인식 ,음성인식, 의사결정, 번역 등이 있음
    + 소셜 등 다양한 복합적인 감각을 시스템에 제공함, 기기인식은 현재도 개발되는 연구능력
    + 오감 중 시각에 많이 의존함 ( 75% 정도의 정보가 눈으로 들어옴 )
    + 컴퓨터 그래픽스(Rendering) 정보를 통해서 2D 이미지를 Drawing 하는 것
        + Computer Vision (inverse rendering)이라고도 함
    + 인간의 눈과 시각을 담당하는 뇌의 학습도 시각 기능이 bias 되어 학습이 되어 있음
+ What is computer vision?
    + Machine Learning : Input -> Feature extraction -> Classification -> output : 기존에는 Feature를 토대로 분류하여 결과물을 출력하는 패러다임이었음
    + Deep Learning : Input -> Feature extraction + Classification -> output : Gradient Descent가 특징을 찾는게 사람보다 낫다고 평가됨

#### 2. Image Classification
+ 영상 분류는 영상이 입력으로 주어질 떄 영상이 해당하는 카테고리를 출력하는 클래스다.
+ 모든 분류 문제는 k Nearest Neighbors (k-NN) 문제로 변환해서 풀 수 있음
    + k Nearest Neighbors (k-NN)은 질의 데이터가 들어오면 근방에 포진하고 있는 이웃 데이터를 DB에서 찾고 이들이 갖고 있는 라벨정보를 기반으로 분류한는 것
    + 영상 분류 문제가 검색문제로 바뀜 (데이터가 많으면 검색 시간이 데이터 수에따라 증가하게 됨)
    + Neural networks는 이런 방대한 데이터를 제한된 복잡도의 시스템에 압축 시켜서 녹여넣는 것으로 이해할 수 있음
+ 모든 픽셀들을 서로 다른 가중치로 Weighted Sum(Fully-connected)을 하고, 활성 함수를 통해 분류 스코어로 출력
+ 템플릿(가중치)을 시각적으로 표현하면 각 클래스마다의 정규적인 표현, 평균 영상들을 찾을 수 있음
+ 레이어가 한층이면 단순하여 평균이미지가 아니면 표현이 어려움
    + 실제 테스트 시에는 템플릿의 위치나 스케일이 달라지면 다른 결과나 해석을 보여주게됨
+ 위 문제를 해결하기 위해서는 locally connected를 사용하게됨
    + 극부적인 영역들을 학습함
    + 전영역을 순회하면서 히든 노드의 activation을 뽑는 것을 CNN임(더 적은 파라미터로도 효과적인 특징을 추출할 수 있음)
    + 영상에서 위치가 바뀌더라도 결과를 추출할 수 있음
+ CNN은 CV의 backbone으로 사용됨

#### 3. CNN architectures for image classification1
+ AlexNet(2012) : 뉴럴 네트워크가 연구자들에게 이목을 받게됨
    + Yann LeCun(1998)에 의해 소개됨 Conv - Pool - Conv - Pool - FC - FC 구조
    + AlexNet은 위 구조에서 많은 Motivation을 따옴
        + 레이어가 7개로 늘어나서 딥해짐
        + Activation Function으로 ReLu를 사용하고, dropout이라는 정규화 기술을 사용함
    
        ![캡처](https://user-images.githubusercontent.com/44515744/110264467-b0d27200-7ffc-11eb-966c-2c5d8e20682b.PNG)

    + Local Response Normalization(LRN)은 현재 사용되지 않음 - 명암을 Nomalization 하는 역할
    + Batch normalization이 대중적으로 쓰이는 중

+ VGGNet
    + 3x3 conv filters block, 2x2 pooling만 사용
        + 작은 컨볼루션 레이어들도 스택을 많이쌓으면 큰 Receptive field size를 얻을 수 있음(더 적은 파라미터로 더 깊은 레이어들로 학습할 수 있음)
    + 일반화가 잘되는 특징을 추출하게 됨

### (2강) Annotation data efficient learning
#### 1. label 데이터의 효율적인 학습 기법
+ 뉴럴 네트워크는 데이터를 컴퓨터로만 이해할 수 있는 지식의 형태로 압축해놓은 것
+ 실제 세상에 존재한 데이터는 bias 보통 사람이 보기좋게 찍은 사진들임
+ 이 세상의 데이터 중 일부만 취득할 수 있음 + 이 데이터에는 bias가 껴있음

#### 2. Data augmentation
+ 학습 데이터에 있는 데이터를 기본적인 Operation을 통해서 여러장으로 만듬
    + 기하학, 칼라, 명암 변환 등에 해당  (Crop, Brightness, Rotate)
+ OpenCV, Numpy에 이런 data augmentation에 대한 기능들을 지원함
+ 변환 기법들
    + Rotate
    + flip
    + Crop
    + Affine transformation(선간의 평행이 유지됨)
    + Cutmix (두 영상을 잘라서 합성 - 라벨도 동일한 비율로 합성해줌 )
    + 여러가지 다양한 기법들을 조합해서 학습할 수 있음 
+ Rand Augmentation을 활용하여 수행한 다음 성능이 잘 나오는 것을 가져다 쓸 수 있음
    + 어떤 Augmentation 방식을 사용, 얼마나 세게 적용할지를 파라미터로 활용함

#### 3. Leveraging pre-trained information
+ Transfer learning : 기존에 미리 학습시킨 지식을 사용해서 연관된 새로운 task에 적은 노력으로도 높은 성능으로 도달이 가능함
+ 하나의 데이터셋에서 배운 학습내용이 다른 데이터셋에도 적용될 것
+ Approach 1 : 프리트레인 테스크에서 새 테스트로 학습 전이하는 방법
    + 프리 트레인 모델의 마지막 레이어를 제거하고 새로운 FC layer를 추가 Convolution Layer의 지식을 유지할 수 있음
+ Approach 2 : 마지막 레이어(FC)를 자르고 대체함 그리고 Larning rate를 높게줘서 학습함, Convolution Layer도 Learning rate를 낮게 잡고 학습을 진행
+ Approach 3 : Teacher-student learning ( Student Network에 학습 내용을 주입하는데 사용 )
    + 모델 압축에 활용됨
    + pseudo-lableling에 활용

+ Teacher-student network structure

    ![캡처](https://user-images.githubusercontent.com/44515744/110269957-1f68fd00-8008-11eb-94a3-0f6cc4c1a44e.PNG)

+ Hard label vs Soft label

![캡처](https://user-images.githubusercontent.com/44515744/110270097-6eaf2d80-8008-11eb-8acc-dead3caa8bbd.PNG)

+ Softmax with temperature(T)
    + 기존의 Softmax는 극단적인 값을 반환함, T 를 주어서 이것은 완화할 수 있음

![캡처](https://user-images.githubusercontent.com/44515744/110270175-9c947200-8008-11eb-8b53-d43eda325668.PNG)

+ Distillation Loss, Student Loss
    + Distillation Loss : KLdiv loss(Soft label, Soft prediction)
    + Student Loss : Cross Entropy loss(Hard label, Soft prediction)

![캡처](https://user-images.githubusercontent.com/44515744/110270362-fd23af00-8008-11eb-8554-7b15de7c2912.PNG)

#### 4. Leveraging unlabeled datset for training
+ 한정된 Supervised 데이터가 주어졌을때 만족할만할 성능을 도달하기 위해 unlabeled dataset을 활용하는 방법
    + semi-Supervised learning : unlabeled data를 목적성있게 잘 사용하는 방법
    + Supervised data는 labeling이 필요함 라지 스케일로 대규모 데이터를 구축하는데 한계가 있음
    + 라벨이 필요없다면 온라인상의 많은 데이터를 사용할 수 있음
    + Labeled 데이터로 프리트레인(Labeled dataset)을 한 다음, Unlabeled 데이터의 Pseudo-label을 이 모델을 통해서 잔뜩생성함
    + 마지막으로 Labeled dataset와 Pseudo-labeled dataset 두 데이터를 활용하여 재학습된 모델을 생성

#### 5. Self-Training
+ Data Augmentation, Knowledge distillation,Semi-Supervised Learning 방식을 조합하여 사용이 가능( 
    Pseudo label-based method)

+ Self-training 구조 (출처 : [Xie et al., CVRP 2020])
    ![캡처](https://user-images.githubusercontent.com/44515744/110271319-1fb6c780-800b-11eb-820d-94735e9adc50.PNG)

    ![캡처](https://user-images.githubusercontent.com/44515744/110271482-83d98b80-800b-11eb-8386-c2db64032c74.PNG)

    