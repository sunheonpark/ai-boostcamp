## [Day 28] 캐글 경진대회 노하우 & Full Stack ML Engineer
### (특강 3강) 김상훈 - 캐글 그랜드마스터의 경진대회 노하우 대방출
#### 1. Kaggle이란?
+ 2010년 설립된 세계에서 가장 유명한 인공지능 대회 플랫폼
+ 2017년에 구글에 인수됨

#### 2. 국내 유명 경진대회 플랫폼
+ 카카오 아레나(카카오 아레나 전용)
+ 데이톤

#### 3. 캐글을 하는 이유
+ 실력을 인정 받기 위해 (취업 목적)
+ AI개발자로 배우고 성장하기 위해(몇 번 조작으로 빅데이터를 읽어서 훈련 및 학습이 가능)

#### 4. 캐글에서 실력을 인정 받으려면
+ 랭킹 시스템 활용 : 경진대회에서 높은 순위에 들어 포인트를 획득
+ 티어 시스템 활용 : 경진대회에서 메달을 따게 되면 개수에 따라서 티어가 결정됨
+ 상위 랭커가 되려면 혼자 경진대회에서 높은등수를 획득
+ 그랜드 마스터가 되려면 5개의 금메달을 획득해야함, 이 중 1개는 솔로 금메달

#### 5. 캐글 시작해보기
+ 1) 회원가입
+ 2) 참여할 대회 선택
+ 3) 대회의 Data 메뉴로 이동해서 대회 데이터를 다운받을 수 있음
+ 4) 대회를 위한 파이프라인 구축
    + 데이터 전처리
    + 학습
    + 제출 준비
    + 리더보드 제출(Code Competition Type)
+ 5) 캐글로 파이프라인을 빠르게 경험해 보기
    + Notebookes 메뉴를 클릭해서 필요한 노트북을 찾을 수 있음
    + 다른 사용자의 노트북을 복사하여 사용함, 복사한 것을 계정에 저장하기 위해 우측의 [Save Version]을 클릭함
    + 계정에 복사가 되면 하단에 Submit 버튼이 생기는데 이를 통해 리더보드 제출이 가능


#### 6. 캐글 관련 개념
+ 대회 개최 목적
    + Fatured : 사업적 목적의 예측 대회, 우승한 모델을 현업에 적용
    + Research : 연구 목적의 대회
    + Getting Started & Playground : 초심자를 위한 학습 목적의 대회 (랭킹용 포인트나 메달 획득이 불가능)
    + Analytics : 데이터 분석 목적의 대회
    + Recruitment : 리크루팅 목적의 대회
+ 대회 제출 방식
    + General Competition (submission.csv 파일만 메뉴에서 제출)
    + Code Competition (캐글 노트북에서 코드를 실행시켜 파일을 생성해야함, 캐글러들이 쓸모 있는 모델을 만들도록 강제함)

#### 7. 우승하기 위해 필요한 것
+ 파이프라인의 (빠른/효율적) 반복
    + GPU 장비
        + 추천 장비 (200만원)
        + CPU : AMD 라이젠 3세대 3700 이상(8코어)
        + RAM : 64GB 이상
        + SSD : 1TB 이상
        + GPU : RTX 2080Ti x 2대 (블로워 타입 중고로 구입 - 비블로워 타입은 GPU 사이에 빈공간이 없음)

    + 추천 장비2 (300~400만원)
        + CPU : AMD 라이젠 3세대 3700 이상(8코어)
        + RAM : 64GB 이상
        + SSD : 1TB 이상
        + GPU : RTX 3090 1대 or RTX 3080 1대

    + 추천 장비3
        + CPU : AMD 라이젠 스레드리퍼
        + RAM : 128GB 이상
        + SSD : 1TB 이상
        + GPU : RTX 3090 2대 ( or RTX 3080 2대 or RTX 2080Ti 4대)
 
    + 시간 투자
        + 수천팀이 참여하여 2~3달 동안 많은 시간을 투자 (평일 4시간, 주말 10시간 정도의 시간이 소요)

    + 본인만의 기본 코드
        + 이 데이터를 기준으로 대회에서 일부만 수정하여 빠르게 파이프라인을 구축할 수 있고 실수가 적어지게 됨
        + 참고 : https://github.com/lime-robot/categories-prediction

    + 점수 개선 아이디어
        + 캐글 Notebooks
            + 다양한 아이디어를 얻을 수 있음 (Best Score, Most Votes 정렬 기능 사용을 추천)
        + Discussion 탭 참고
            + 다양한 아이디어를 얻을 수 있음, 비슷한 이전 대회, 참고할 논문, 현대 대회 Overview 등등
    
    + (올바른 방향인지) 탄탄한 검증 전략
        + 최종 순위 하락을 피하기 위해 (Public LB 1등, Private LB 70등) - 오버피팅을 피하기 위한 검증 전략이 필요함(순위가 뒤 바뀌는 현상을 Shake-up이라고 함)
        + 리더보드 제출 횟수 제한이 있음 : 보통 하루에 5회 제공
        + 좋은 모델은 일반화 성능이 높은 모델, Training set에서 얻은 점수가 Test set에서도 비슷하게 나오는 모델
        + Test set에서 얻은 점수와 Training set에서 얻어진 점수 갭 0.03을 줄이는 평가 방법
        + 캐글 데이터는 Training set과 Test set이 있음
            + Test set은 (Public set과 Private set 두가지로 나뉨)
        + 트레이닝 셋을 80:20으로 나눠서 Training set과 Validation set으로 나눔
        + 점수 갭을 줄이기 위해 local CV(Cross Validation)을 사용함
        + K-Fold 방식은 Validation set에 하나의 라벨 값만 들어갈 수 있으므로 Stratified k-fold를 사용

    + 기타 꿀팁
        + 앙상블은 여러 모데의 예측 결과를 섞어서 예측 성능을 향상시킴, 싱글 모델보다 좋은 결과를 얻음
        + Stratified k-fold 앙상블
        + 다양한 모델 앙상블
            + 정형 데이터 : LightGBM, XGBoost, Neural Networks(NNs) 등
            + 이미지 데이터 : resnet, efficientnet, resnext 등
            + 텍스트 데이터 : LSTM, BERT, GPT2, RoBert
        + 상위 랭커들이 discussion에 언급한 자신의 싱글모델 점수 참고
        + 주피터에서 터미널을 열 수 있음, 이 터미널은 크롬 창을 닫아도 살아있으므로 원격 학습 가능

### (특강 4강) 이준엽 - Full Stack ML Engineer
#### 1. Full stack ML Engineer?
+ Machine learning 기술을 이해하고, 연구하고, Product를 만드는 Engineer
+ Deep learning의 급부상으로 Product에 Deep learning을 적용하고자 하는 수요 발생

![캡처](https://user-images.githubusercontent.com/44515744/109752530-ea356700-7c23-11eb-8296-8dbc678187bb.PNG)

+ Full Stack engineer란?
    + 내가 만들고 싶은 Product를 시간만 있다면 모두 혼자 만들 수 있는 개발자

+ Full stack ML Engineer란?
    + Deep larning research를 이해하고, ML Product로 만들 수 있는 engineer

    ![캡처](https://user-images.githubusercontent.com/44515744/109753107-fec62f00-7c24-11eb-9942-4f32e7119863.PNG)



#### 2. Pros, cons of Full stack ML engineer
+ 장점
    + 소프트웨어 개발은 컴퓨터에서 돌아가는  작은 세상을 만드는 작업
    + 처음부터 끝까지 만드는 것은 재밌음
    + ML 모델의 빠른 프로토타이핑
    + GO/STOP의 결정을 뒤 바꿀 수 있는 트리거가 될 수 있음 (협업하기는 곤란한 경우가 많음)
    + 연결되는 Stack에 대한 이해가 각 Stack의 깊은 이해에도 도움을 줌
    + 결국 하나의 서비스로 합쳐지는 기술들, 연결되는 부분에 대한 이해가 중요
    + 다른 포지션 엔지니어링에 대한 이해가 발생할 수 있는데, 서로 갈등이 생길법한 부분에서 기술적인 이해가 매우 도움이 됨
    + 성장의 다각화, 여러 분야의 사람이 모인 회의에서 모든 내용이 성장의 밑거름이 됨
+ 단점
    + 하나의 스택에 집중한 개발자 보다는 해당 스택에 대한 깊이가 없어질 수 있다.
    + 하루가 다르게 새로운 기술+새로운 연구가 나오는게 CS + ML 분야
    + 공부할 분야가 많다보니 시간이 많이 들어간다. ( 여러 스택에 개발을 투자하더라도 연봉을 비례하진 않다.)

#### 3. ML Product, ML Team, ML Engineer
+ 요구사항 전달 -> 데이터 수집 -> ML 모델 개발 -> 실서버 배포
+ 요구사항 : 고객의 요구사항을 수집하는 단계
    + 고객사 미팅, 서비스 기획
    + 요구사항, 제약사항 정리
    + ML Problem으로 회귀 (ML이 풀 수 있는 형태의 문제로 회귀하는 작업)
+ 데이터 수집 : Model을 훈련/평가할 데이터를 취득하는 단계
    + Raw 데이터 수집 : 저작권 주의, Bias 없도록 분포 주의, 요구사항에 맞는 데이터 수집
    + Annotation Tool 기획 및 개발 : 데이터에 대한 정답 입력 툴, 모델 Input / Output 고려, 작업속도 / 정확도 극대화
    + Annotation Guide 작성 및 운용 : 간단하고 명확한 Guide 문서를 작성하도록 노력
+ ML 모델 개발 : Machine learning 모델을 개발하는 단계
    + 기존 연구 Research 및 내재화
    + 실 데이터 적용 실험 + 평가 및 피드백
    + 모델 차원 경량화 작업
+ 실 서버 배포 : 서비스 서버에 적용하는 단계
    + 엔지니어링 경량화 작업 : TensoRT 등의 프레임워크 적용, Quantization
    + 연구용 코드 수정 작업 : 연구용 코드 정리, 배포용 코드와 Interface 맞추는 작업
    + 모델 버전 관리 및 배포 자동화 : 모델 버전 관리 및 배포 자동화(모델 버전 관리 시스템, 모델 배포 주기 결정& 업데이트 배포 자동화 작업)

#### 4. ML Team
+ ML Team의 일반적 구성은 아래와 같음
    + 프로젝트 매니저(1), 개발자(2), 연구자(2), 기획자(1), 데이터 관리자(1)
+ 팀에서의 Full stack ML Engineer의 역할
    + 1) 실 생활 문제를 ML 문제로 Formulation, 기존 연구에 대한 폭 넓은 이해와 최신 연구의 수준을 파악하고 있어야함
    + 2) Raw Data 수집 - 웹에서 학습데이터를 모아야 하는 경우도 있음 ( Web Crawler, Scraper를 개발_ 저작권 주의 ) - annotation을 거쳐서 input으로 사용됨
    + 3) Annotation tool 개발 (Vue.js, django, mysql, docker를 사용)
        + 새로운 Task에 대한 Annotation Tool 기획시 모델에 대한 이해가 필요할 수 있음
    + 4) Data Version 관리 및 loader 개발
        + 쌓인 데이터의 Version 관리
        + Database에 있는 데이터를 Model로 Load하 하기위한 Loader package 개발
    + 5) Model 개발 및 논문 작성
        + 기존 연구 조사 및 재현, 수집된 서비스 데이터 적용, 모델 개선 작업 + 아이디어 적용(논문 작성)
    + 6) Evaluation Tool 혹은 Demo 개발
        + 모델의 Prediction 결과를 채점하는 Web application 개발
        + OCR 프로젝트 중 혼자 사용하려고 개발 (정/오답 케이스 분석) -> 이후 팀에서 모두 사용.
        + 모두 사용하다보니 모델 특성 파악을 위한 요구사항 발생 -> 반영하다보니 모델 발전의 경쟁력이 됨
    + 7) 모델 실 서버 배포
        
        ![캡처](https://user-images.githubusercontent.com/44515744/109762090-a9454e80-7c33-11eb-97b5-e99b0f3a3701.PNG)

        + Frontend : Vue.js, Angular.js
        + Backend : django, flask, rails
        + Machine Learning : PyTorch, TensorFlow
        + Database : MySQL, MariaDB, Redis, Amazon DynamoDB
        + ops : docker, github, aws