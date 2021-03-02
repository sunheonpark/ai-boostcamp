## [Day 27] 서비스 향 AI 모델 개발 & AI 시대의 커리어 빌딩
### (특강 1강) 서비스 향 AI 모델 개발하기
#### 1. 서비스향 AI 모델 개발 VS 수업/학교/연구 AI 모델 개발
+ 연구 관점에서 AI 개발이란?
    + 보통 수업/학교/연구에서는 정해진 데이터셋/평가 방식에서 더 좋은 모델을 찾을 일을 함
    + 학습 데이터셋과 테스트 데이터셋이 주어져 있음
+ 서비스 관점에서 AI 개발이란?
    + 학습 데이터셋과 테스트 데이터셋도 주어지지 않은 경우가 많음
    + 서비스 요구사항만 있는 경우가 많음
    + 정확히는 서비스 요구사항으로 부터 학습 데이터셋의 종류/수량/정답을 정해야 함
    + 질의응답을 통해 데이터셋의 종류/수량/정답 관련 요구사항을 구체화해야 한다.
+ 예시) 수식을 사진으로 찍어서 인식하는 기술
    + 종류 : 초중고 수식, 손글씨/인쇄
    + 정답 : Latex String
    + 수량 : 종류로 정의해서 각각 몇 장을 수집 할 것인지 정해야함
+ 학습 데이터셋 준비

    ![캡처](https://user-images.githubusercontent.com/44515744/109586132-095ec680-7b48-11eb-91b7-7bf0601710b0.PNG)

+ 기술 모듈 설계

![캡처](https://user-images.githubusercontent.com/44515744/109586194-20051d80-7b48-11eb-9f7f-6aa093375072.PNG)

+ 하나의 이미지에 수식이 여러개가 있을 경우, 기술 모듈이 아래와 같이 수정됨

![캡처](https://user-images.githubusercontent.com/44515744/109586344-665a7c80-7b48-11eb-9778-c2f7550ef2d0.PNG)

+ 학습데이터의 '정답'은 AI 모델 별로 입력에 대한 출력쌍이다.
+ 수식 영역 검출에 대한 모델

![캡처](https://user-images.githubusercontent.com/44515744/109586659-fc8ea280-7b48-11eb-8614-fea061b313f2.PNG)

+ Image To Latex에 대한 모델, 각 모델(총 4개 모델) 별로 입출력(정답) 정의가 필요하다.

![캡처](https://user-images.githubusercontent.com/44515744/109586758-32338b80-7b49-11eb-8012-6625e0ce292c.PNG)

+ 기획자, 학습 데이터셋, 모델링 인원이 지속적으로 논의하면서 안에 대해서 결과가 점점 수렴함

![캡처](https://user-images.githubusercontent.com/44515744/109587071-c6055780-7b49-11eb-95bc-360af9c3a46f.PNG)

+ 학습 데이터셋의 일부를 테스트 데이터셋으로 사용하는 경우가 많음, 서비스 요구사항으로부터 테스트 방법을 도출해야함
+ 결국, 서비스에서의 품질이 중요하기 떄문에 OFFLINE 테스트 결과와 유사하게 OFFLINE 테스트를 잘 설계해야 함

![캡처](https://user-images.githubusercontent.com/44515744/109591611-2cda3f00-7b51-11eb-905b-30ee748d48d0.PNG)

+ 모델 요구사항 도출
    + 처리 시간 : 처리 시간은 하나의 입력이 처리되어 출력이 나올 때까지의 시간
        + OFFLINE TEST : 이미지 입력 후 수식 영역 정보가 출력될 때까지의 시간
        + ONLINE TEST : 이미지 촬영 후 이미지에서 수식 영역 정보가 화면 상에 표현되기까지의 시간
    + 목표 정확도 : 해당 기술 모듈의 정량적인 정확도
        + OFFLINE TEST : 입력된 이미지 내 카드 번호/유효기간에 대한 EDIT DISTANCE
        + ONLINE TEST : 사용자가 AI 모델의 결과값을 수정할 확률
    + 목표 qps : QPS(Queries Per Second)는 초당 처리 가능한 요청 수를 의미함
        + 향상 방법 
            + 장비를 늘린다(N대를 늘리면 QPS가 N배 올라간다)
            + 쿼리 시간을 줄인다(AI 모델의 처리 속도가 N배 올라가면 QPS도 N배 올라간다.)
            + 모델 크기를 줄인다(한 GPU에 올라가는 모델 수가 N배가 되면 QPS도 N배 올라간다, 배수만큼 줄여줘야 QPS가 올라감)
    + Serving 방식
        + 기술 모듈이 Mobile에서 동작 하기 원하는지
        + Local CPU/GPU Server에서 동작 하기 원하는지
        + Cloud CPU/GPU Server에서 동작 하기 원하는지
    + 장비 사양
        + 가끔은 Serving 장비조차 없어서 장비 구축까지 같이 요규하는 경우도 있음
        + 이때 예산/QPS에 맞춰서 장비 사양도 정해야함
    
#### 2. 서비스 향 AI 모델 개발 기술팀의 조직 구성
+ AI 모델팀
    + 모델러는 Metric에 맞춰서 AI 모델 구조를 제안, 모델 성능 분석/디버깅
    + Data Curator : 학습 데이터 준비(외주 업체 대응, 작업 가이드 문서, QnA 대응), 정량 평가 수립, 정성 평가 분석
    + IDE Developer : 라벨링 툴 개발, 모델 분석 툴 개발, 모델 개발 자동화 파이프 라인 개발
    + Model Quality Manager : 이 전체를 총괄하여 모델의 품질을 관리하는 사람이 필요함
+ AI 모델 서빙팀
    + Model Engineer
        + Mobile에서 구동하기 위해서 Pytorch -> Tensorflow -> TFLite로 변환
        + GPU Server에서 구동하기 위해서 Pytorch -> Tensorflow -> TensorRT로 변환
        + 모든 연산을 C++/C로 변환
        + GPU 고속 처리를 위해 CUDA Programming
        + 메모리를 줄이기 위한 Lighweight(경량화) 작업 (Distillation, Quantization)

### (특강 2강) AI 시대의 커리어 빌딩
#### 1. Carrers in AI
+ 학교 VS 회사
    + Academia : 논문을 쓰는 곳
        + 논문을 써서 연구 성과를 만드는 것이 목표
        + 논문을 쓰는 방식을 지도 받을 수 있음
    + Industry : 서비스/상품을 만드는 곳
        + 서비스/상품을 만들어서 돈을 많이 버는 것이 목표
        + 논물을 쓸 시간적 여력이 부족할 수 있음
        + 상대적으로 데이터, 계산자원이 풍부함
        + 다 정제되어 있는 작은 규모의 토이 데이터가 아니라 어마어마한 양의 트래픽을 통해 발생되는 대규모의 리얼 데이터를 만질 수 있음

+ AI를 다루는 회사의 종류
    + AI for X
        + AI로 기존 비즈니스를 더 잘하려는 회사
            + 비용을 줄이거나, 품질을 높이는데 AI를 활용
            + AI는 보조 수단, 대부분의 회사가 여기에 해당
    + AI centric
        + AI로 새로운 비즈니스를 창출하려는 회사
            + 새로운 가치창출을 하는데 AI를 활용
            + 신생 회사들이 많음

+ AI를 다루는 팀의 구성
    + Business : 사업 기획자, 서비스 기획자, 마케팅/세일즈/PR, 법무/윤리학자
    + Engineering : Data engineer, AI engineer, AIOps engineer

+ AI 엔지니어의 역할
    + AI/ML 모델링은 팀 전체 업무의 일부
    + 다양한 업무가 있는 만큼 팀 내에는 다양한 역할이 있음
    + ML DevOps, Data Engineer, Data Scientist, ML Engineer, ML Researcher 등 다양한 포지션이 존재함