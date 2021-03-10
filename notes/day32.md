## [DAY 33]
### (5강) Object detection
#### 1. Object detection
+ Instance segmentation, Panoptic segmentation이 강조되고 있음
    + Semantic segmentation에서 이 단계로 넘어가기 위해서는 인스턴스를 구분하는 지에 대한 고려도 필요함
    + Object detection은 Classification과 Box localization을 모두 사용한 것
    + 영상인식보다는 좀더 고차원으로 생각해야함
    + 자율 주행뿐만아니라 OCR 등에서도 활용 됨

#### 2. Two-stage detector
+ 16년 전에 사람 영상의 경계선 평균(Average Gradient)을 냈더니 사람의 형상이 나타나게 됨
+ Gradient-based detector : 영상의 그레디언트를 기반으로한 디텍터들이 많이 사용됐었음
+ Selective search : 영상의 색을 기준으로 잘게 분한한 다음, 그레디언트 분포 또는 색상을 기준으로 비슷한 유형끼리 합쳐줌, 반복해서 합쳐주면 많게됨
    + 큰 세그멘테이션이 있는데 이를 포함하는 타이트한 바운딩 박스를 추출해서 물체의 후보군으로 사용할 수 있음

#### 3. R-CNN
+ 기존의 image classfication을 활용하기 위해서 간단하게 설계됨
    + region proposals을 image classification 네트워크에 적절한 크기로 wraping을 해줌
    + 미리 학습된 CNN에다가 이를 넣어주고 분류함
    + 단점으로는 region proposals 하나하나마다 모델에 넣어서 프로세싱을해야했기 때문에 속도가 느림, Hand Design된 알고리즘을 사용해서 성능 향상에 한계가 존재함

#### 4. Fast R-CNN - Girshick et aI., ICCV 2015
+ 영상 전체에 대한 피쳐를 한번에 추출을 하고 이를 재활용하여 여러 Object들을 Detection하는 것
+ Convolution Layer만 거친 Feature map은 Tensor 형태를 갖고 있음
+ Fully Convolutional한 구조의 네트워크는 입력사이즈에 상관없이 Feature map을 추출할 수 있음
+ 한번 뽑아 놓은 Feature를 여러번 재활용하기 위해 Roi(Region of interest) pooling을 사용함
+ Feature가 Pooling이 되면 Class와 더 정밀한 bounding 박스를 추출하기 위해 FC레이어를 활용함

![캡처](https://user-images.githubusercontent.com/44515744/110598075-b24e9680-81c4-11eb-9cee-d7e51ae4ae51.PNG)

#### 5. Faster R-CNN - Ren et al., NeurlPS, 2015
+ region proposal을 뉴럴 네트워크 기반으로 대체하여 Object Detection에서는 Faster R-CNN이 최초의 End-to-End 모델이 됨
+ IoU(Intersection over Union) : 두 영역의 교집합 /두 영역의 합집합을 계산하여 이 수치가 높을 수록 두 영역이 잘 정합됐다고 볼 수 있음
+ Anchor boxes : Region Proposal을 위한 기능으로 각 위치에서 발생할 것 같은 후보군 박스들을 미리 정의해놓음(비율과 스케일이 다름)
    + Groud Truth와의 IoU가 0.7보다 크면 Positive Sample, 작으면 Negative Sample이라고 할 수 있음
+ Selective search의 서드파티 알고리즘 대신에 Region Proposal Network(RPN) 모델을 제안함
    + anchor box를 촘촘하게 많이쓰면 문제가 되지않으나 계산속도가 어마어마하게 느려짐, 적당한 양의 Anchor box들을 만들고 정교한 위치는 Regression 문제로 푸는 것이 원리

    ![image](https://user-images.githubusercontent.com/44515744/110599503-4705c400-81c6-11eb-8216-7e91a687f5ba.png)

+ Non-Maximum Suppression (NMS)
    + 그럴듯한 바운딩 박스만 남겨놓음 
    + Step 1 : 높은 점수의 박스를 선택
    + Step 2 : 다른 박스들과의 IoU를 계산
    + Step 3 : IoU가 50% 이상힌 바운딩 박스를 제거
    + Step 4 : 다음으로 점수가 높은 객체를 선택
    + Step 5 : 2-4를 반복

#### 6. R-CNN 정리
+ R-CNN : Region Proposal을 별도의 알고리즘을 사용, Deep CNN도 타겟 테스크에 대해 학습되지 않고 Pretrain됨
+ Fast R-CNN : RoI pooling을 사용해서 Feature로 부터 여러개의 물체를 탐지 가능하도록 만들어서 CNN 부분을 학습 가능하게 만듬
+ Faster R-CNN : RPN이라는 구조를 만들어서 Region Proposal도 학습할 수 있도록 만듬

#### 7. Single-Stage Detector [illustrations from [Ndonhong et aI., Offshore Technology Conference 2019]]
+ 정확도가 떨어지더라도 속도를 확보해서 Realtime Detection이 가능하게 됨, RoI Pooling을 사용하지 않고 곧바로 Box regression과 Classification을 진행

![캡처](https://user-images.githubusercontent.com/44515744/110601411-3eae8880-81c8-11eb-827a-e7cfbaeeeb30.PNG)

#### 8. You only look once(YOLO)
+ Input 이미지를 s X s grid로 나눔, Bounding boxes + confidence + class probability map을 활용하여 찾음

![캡처](https://user-images.githubusercontent.com/44515744/110601662-8a613200-81c8-11eb-8c6d-00e6d8faa6dd.PNG)

#### 9. Single Shot MultiBox Detector (SSD)
+ 맨 마지막 레이어에서 한번만 Prediction을 하기 때문에 Localization 성능이 떨어질 수 있음, 이를 보완하기 위해서 SSD가 나옴
    + SSD는 멀티 스케일 오브젝트를 더 잘 처리하기 위해 각 해상도에 적절한 스케일을 지정할 수 있게 해줌
    + Yolo 보다 빠르고 Faster R-CNN보다 정확함

#### 10. Two-stage detector vs one-stage detector
+ single stage 방법은 RoI Pooling이 없다보니 모든 영역에서의 loss가 계산되고 일정 Gradient가 계산됨
    + 일반적인 영상에서는 Background가 넓고 물체는 일부분임 (Possitive Sample이 적음)
    + Class Invalance 문제
+ Focal loss
    + 낮은 loss 값을 반환하고 맞추지 못하면 큰 loss를 붙임
    + 어렵고 잘못판별된 애들에 대해서는 강한 Weight 쉬운 것에는 약한 Weight를 줌
+ Feature Pyramid Networks(FPN)    
    + Unet과 유사한 구조, 클래스 헤드와 박스 헤드 따로 구성해서 classfication과 box regression을 dense하게 수행하게 됨
    + SSD와 유사하면서 더 높은 성능을 가짐

#### 11. Detection with Transformer
+ NLP에서 큰 성공을 거둔 Transformer을 사용하는 연구가 트렌드
    + DETR은 Transformer을 Object Detection에 활용한 사례
    + Object query : 디코더 학습한 위치에 대한 인코딩을 질의함
    + 디코더는 각 위치에 대한 결과를 반환함

#### 12. Further reading
+ Bounding box가 아닌 다른 형태로 Detection 할 수 있는 방법에 대한 연구가 진행됨    

### (6강) CNN visualization
#### 1. Visualizing CNN
+ Neural Network
    + Input image -> Black box -> Output prediction
    + 이 Black box에 있는 내용을 알아볼 수 있음
+ CNN은 여러 단계를 거쳐서 학습을 통해서 만들어진 Weight 들의 조합
+ Deconvolution을 통해서 각 레이어가 어떤 것을 보고 있고 어떤 역할을 할 수 있는지 확인할 수 있음
+ 뒷쪽 레이어는 차원수가 높아서 사람이 직관적으로 알아볼 수 있는 형태를 알아볼 수가 없음
+ neural network visualization을 통해 모델의 행동과 결정에 대한 설명을 파악할 수 있음
    + 오른쪽으로 갈수록 데이터에 좌측으로 갈 수록 모델에 중점을 둠

    ![캡처](https://user-images.githubusercontent.com/44515744/110611070-352a1e00-81d2-11eb-9421-b4568704ae63.PNG)

#### 2. Analysis of model behaviors
+ Nearest neighbors (NN)은 데이터베이스가 있고, 데이터베이스 내에 분석을 위한 예제 데이터셋이 존재함
+ 이미지에 대한 특징 벡터들을 뽑아놓고 이게 고차원 공간상에 존재하게 됨, 입력한 이미지에 대한 특징 벡터를 뽑고 주변 벡터들의 특징을 뽑아서 해당 이미지가 무엇인지 파악할 수 있음
+ 고차원 스페이스에 있는 것을 차원축소를 통해서 2차원으로 표현하는 방법들이 존재함
+ t-SNE(t-distributed stochastic neighbor embedding) : 고차원의 데이터를 클래스로 구분하여 저차원으로 표현할 수 있는 방법
+ Layer activation : 레이어의 activation을 분석하여 모델을 해석하는 방법, 각 레이어의 히든 노드들의 역할을 파악할 수 있음
+ Maximally activating patches : 히든 노드에서 가장 높은 값을 갖고있는 위치의 근방의 패치를 뜯을 수 있음
    + STEP1) 특정 레이어의 채널을 하나 고름
    + STEP2) 예제 데이터를 백본 데이터에 넣어서 activation을 뽑고 저장을 함
    + STEP3) 저장된 채널 중 가장 큰 값을 갖는 위치를 파악, 입력도메인의 맥시멈 값을 도출하게한 receptive 필드를 계산한 다음에 히든 패치를 가져옴
+ Class visualization : 네트워크가 내재하고 있는 이미지가 어떤 것인지 분석하는 방법 
    + 간단한 연산으로는 구할 수가 없고, 합성 영상을 위해 loss를 만들고 이를 최소화하는 방식으로 얻음
        + STEP1) 더미 이미지에 대한 예측 값을 얻음
        + STEP2) 역전파로 입력 값을 찾음
        + STEP3) 현재 이미지를 업데이트함

#### 3. Model decision explanation
+ 모델이 특정 입력을 가졌을때 어떤 각도로 해석하는지에 대한 내용
+ Occlusion map : 영상이 주어졌을때 영상의 중요도를 파악하는 방법, 특정 패치(Occlusion A)를 가리고 값을 넘겨줌
    + 스코어가 높을 수록 중요하지 않은 영역이 됨
    + 각 Occlusion Patch의 위치를 하나하나 테스트함 ( 위치에 따라 변하는 스코어 지도를 기록 )
+ via Backpropagation : 특정 이미지를 classification 해보고 결정적으로 영향을 끼친 부분이 어느 것인지 파악하는 방법
    + 입력 영상을 넣어주고 Class score를 얻음
    + Backpropagation을 입력 도메인까지 쭉함
    + 얻어진 gradient를 절대 값을 취해준다음 이미지 형태로 출력
+ Rectified unit(backward pass) : backward pass 시 음수를 0으로 마스킹함(Relu를 적용), 현재 activation과 다음 activation을 나타냄
    + Guided backpropagation : foward, backward pattern을 모두 0으로 만들어줌, 이미지를 클리어하게 확인할 수 있음
        + Foward시 결정적인 역할을 한 양수를 참조하고 backward를 할때도 그레디언트를 통해서 강화해주는 activation을 고르는 것
+ Class activation mapping (CAM) : Neural net의 일부를 개조해야함 Convolution Feature map을 바로 통과하지 않고 GAP layer를 넣는다.
    + 이미지에 영향을 준 영역이 히트맵으로 강조되어 표현됨
    + CAM은 모델 구조를 바꾸기 때문에 모델 성능에 영향을 줄 수 있음, 따라서 재학습을 해야하는 경우가 생김
    + ResNet과 GoogLeNet은 마지막에 FC 레이어가 하나만 구성되어 있어서 CAM을 활용하기 좋은 구조임
+ Grad-CAM : 구조를 변경하지 않고 재학습없이 Grad-CAM이라는 것이 제안됨
    + 영상이 아니더라도 backbone이 CNN이기만 하면 구현이 가능, 마지막 Feature map을 weighted sum으로 합성함
    + Guided Backporp과 Grad-CAM을 합쳐서 Guided Grad-CAM을 보는 것이 일반화됨