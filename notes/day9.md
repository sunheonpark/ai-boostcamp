## [DAY 9] Pandas II / 확률론
### [AI Math 6강] pandas II
> Pandas 라이브러리의 groupby,pivot_table,joint method (merge / concat),Database connection,Xls persistence 다섯가지 기능에 대한 내용
#### 1. Groupby 1
+ SQL groupby 명령어와 같음
+ split(index를 기준으로 분리) -> apply(함수) -> combine(병합) 과정을 거쳐 연삼
+ 결과물의 타입을 Series
+ Hierarchical index : groupby를 하게되면 기준 컬럼의 개수에 따라 index가 생성됨
``` python
# 기본 활용법
df.groupby("Team")["Points"].sum() # Team 컬럼을 기준으로 Points를 합산
df.groupby(["Team", "Year"])("Points").sum() # 한 개이상의 Column을 묶을 수 있음
h_index = df.groupby(["Team", "Year"])("Points").mean() # Groupby 명령의 결과물도 결국은 dataFrame
h_index.index # 여러개의 index 출력
h_index["Devils":"Kings"] # index 값으로 슬라이싱 가능
h_index.unstack() # 데이터를 매트릭스 형태로 풀어줌
h_index.reset_index() # 데이터를 풀어서 펼쳐줌(Unnest)
h_index.swaplevel() # index 레벨을 변경할 수 있음
h_index.sortlevel(0) # index 레벨을 기준으로 정렬
h_index.std(level=1) # 특정 레벨로 연산이 가능
h_index.sum(level=1) # 특정 레벨로 연산이 가능
h_index.value_counts() # 그룹별 행 수를 구하는 함수

gouped = df.groupby("Team")

for name, group in grouped: # Groupby에 의해 Split된 상태를 추출 가능함
    print(name)
    print(group)
```

#### 2. Groupby - Apply 기능
+ 추출된 group 정보에는 세 가지 유형의 apply가 가능함
    + Aggregation : 요약된 통계정보를 추출
    + Transformation : 들어온 메서드에 대해 각 원소를 살리고 그 안에 연산결과를 채춤
    + Filteration : 특정 정보를 제거하여 보여주는 필터링 기능

``` python
# 기본 활용법

# Aggregation
from numpy import np
grouped = df.groupby(["Team", "Year"])
grouped.agg(np.sum)
grouped.agg(np.mean)
grouped.agg(np.max) # 컬럼 별로 최대값
grouped.agg(np.sum, np.mean, np.std) # 특정 컬럼에 여러개의 function을 Apply 가능

# Transformation
grouped = df.groupby("Team")
score = lambada x : (x)
grouped.transform(score) # Grouped 된 상태에서 모든 값들을 지정해주는 함수
score = lambda x: (x.max())
grouped.transform(score) # 모든 요소를 보여주나 연산은 Key 값을 기준으로 Grouped처리하여 연산한다.

# Filter
df.groupby('Team').filter(lambda x: len(x) >= 3) # len(x)는 grouped된 dataframe 개수
df.groupby('Team').filter(labmda x: x["Rank"].sum() > 2)
df.groupby('Team').filter(labmda x: x["Points"].sum() > 1000)
df.groupby('Team').filter(labmda x: x["Rank"].mean() > 1)
```

#### 3. groupby 활용
``` python
# 기본 활용법
import dateutil

df_phone["date"] = df_phone["date"].apply(dateutil.parser.parse, dayfirst=True) # String 날짜 파싱구문
df_phone.dtypes # 타입확인

!conda install --y matplotlib # matplotlib 설치
df_phone.groupby("month")["duration"].sum().plot() # 데이터를 그래프로 보여줌
df_phone[df_phone["item"]=="call"].groupby("month")["duration"].sum().plot() # item이 call 것만 뽑아서 보여줌
df_phone[df_phone["item"]=="data"].groupby("month")["duration"].sum().plot()
df_phone[df_phone["item"]=="sms"].groupby("month")["duration"].sum().plot()
df_phone.groupby(["month","item"])["duration"].count().unstack().plot() # 컬럼이 여러개일 경우에는 unstack을 활용해서 unnest를 해주고 plot을 그려야함

df_phone.groupby("month", as_index=False).agg({"duration":"sum"}) # agg에 딕셔너리 형태로 연산 추가가 가능함
df_phone.groupby(["month", "item"]).agg(
    {
        "duration": "sum",
        "network_type": "count",
        "date": "first", # 처음 등장 값
    }
)

df_phone.groupby(["month", "item"]).agg(
    {
        "duration": ["min", "max","sum"],
        "network_type": "count",
        "date": [min, "first", "nunique"] # 처음 등장 값
    }
)

grouped.rename(
    columns={"min":"min_duration", "max":"max_duration", "mean": "mean_duration" }
) # 컬럼명 변경

grouped.add_prefix("duration_") # 컬럼에 prefix를 추가함
```

#### 4. Pivot Table
+ 엑셀의 피봇테이블과 동일함
+ 컬럼에 추가로 라벨링 값을 추가하여, Value에 numeric type 값을 aggregation 하는 형태

``` python
# 기본 활용법 
df_phone.pivot_table(
    values=["duration"],
    index=[df_phone.month, df_phone.item],
    columns=df_phone.network,
    aggfunc="sum",
    fill_value=0
)
 
df_phone.groupby(["month", "item", "network"])["duration"].sum().unstack() #groupby를 사용한 것과 결과같이 유사


```

#### 5. Crosstab
+ 두 칼럼의 교차 빈도, 비율, 덧셈 등을 구할 때 사용
+ pivot table의 특수한 형태
+ User-item Rating Matrix 등을 만들 때 사용 가능
+ 위 세가지를 쓰면 동일한 결과를 구현 가능
``` python
# 기본 활용법
pd.crosstab(
    index=df_movie.critic, 
    columns=df_movie.title, 
    values=df_movie.rating, 
    aggfunc="first"
).fillna(0)
```

#### 6. Merge
+ SQL에서 많이 사용하는 Merge와 같은 기능
+ 조인과 같은 개념
+ 조인 메소드 ( 기본 값은 inner 조인 )
    + INNER JOIN : 양쪽 둘다 있는 것만 보여줌
    + LEFT JOIN : 왼쪽것을 보여주고 우측에 없으면 NaN
    + RIGHT JOIN : 오른쪽것을 보여주고 왼쪽에 없으면 NaN
    + FULL JOIN : 양쪽다 보여주고 없는 부분들은 NaN
``` python
# 기본 활용법
pd.merge(df_a, df_b, on='subject_id')
pd.merge(df_a, df_b, left_on='subject_id', right_on='subject_id') # 두 테이블의 컬럼명이 다를 경우

# Join 적용
pd.merge(df_a, df_b, on='subject_id' how='left') # left 조인
pd.merge(df_a, df_b, on='subject_id' how='right') # right 조인
pd.merge(df_a, df_b, on='subject_id' how='outer') # Full 조인

# Join 기준
pd.merge(df_a, df_b, right_index=True, left_index=True) # index 값을 기준으로 붙임
```

#### 7. Concat
+ 같은 형태의 데이터를 붙이는 연산작업
+ 밑으로 붙이거나 옆으로 붙일수도 있음
+ Concat은 리스트 형태로 입력해서 붙인다.

``` python
# 기본 활용법
df_new = pd.concat([df_a, df_b]) # 동일한 컬럼을 갖고 있을때만 가능
df_new.reset_index(drop=True) # 인덱스를 새로 생성

df_a.append(df_b) 
df_a.reset_index(drop=True) # 인덱스를 새로 생성

df_new = pd.concat([df_a, df_b], axis=1) # 데이터가 옆으로 붙게된다.
df_new.reset_index(drop=False)
```

#### 8. Merge, Concat 활용 케이스
``` python
import os
#파일 읽기
files = [file_name for file_name in os.listdir("./data") if file_name.endswith("xlsx")] 
df_list = [pd.read_excel("data/"+df_filename) for df_filename in files]

# status 파일 저장
status = df_list[0]

# sales 파일 병합
sales = pd.concat(df_list[1:])

# 데이터 Merge
merge_df = pd.merge(status, sales, how="inner", on="account number")
merge_df.head()
```

#### 9. Persistence
+ 데이터 프레임 IO 방법에 대한 내용
+ 데이터 베이스를 사용하는 방법
+ sqlite3 : 파일형태로 데이터베이스를 연결하는 것

``` python
# 기본 활용법
# sqlite3 활용
conn = sqlite3.connect("./data/flights.db")
cur = conn.cursor()
cur.execute("select * from airlines limit 5;") 
results = cur.fetchall() # 튜플 형태로 반환

df_airplines = pd.read_sql_query("select * from airlines;", conn) # 쿼리 결과를 데이터 프레임에 저장

# ExcelWriter 활용
!conda install openpyxl
!conda install XlsxWriter

writer = pd.ExcelWriter('./data/df_routes.xlsx', engine='xlsxwriter')
df_routes.to_excel(writer, sheet_name='Sheet1')

# Pickle 활용
df_routes.to_pickle("./data/df_routes.pickle")
df_routes_pickle = pd.read_pickle("./data/df_routes.pickle")
```

### [AI Math 6강] 확률론 맛보기
> 확률분포, 조건부확률, 기대값의 개념과 몬테카를로 샘플링 방법을 설명
#### 1. 딥러닝에서의 확률론의 필요성
+ 딥러닝은 확률론 기반의 기계학습 이론에 바탕을 두고 있음
+ 기계학습에서 사용되는 손실함수(loss function)들의 작동 원리는 데이터 공간을 통계적으로 해석해서 유도
+ 회귀 분석에서 손실함수로 사용되는 L2-노름은 예측오차의 분산을 가장 최소화 하는 방향으로 학습
+ 분류 문제에서 사용되는 교차엔트로피(cross-entropy)는 모델 예측의 불확실성을 최소화하는 방향으로 학습
+ 분산 및 불확실성을 최소화하기 위해서는 측정하는 방법에 대해 알아야함.

#### 2. 확률분포는 데이터의 초상화
+ 데이터공간을 X x Y라 표기하고 D는 데이터공간에서 데이터를 추출하는 분포
+ 이 수업에선 데이터가 정답 레이블을 항상 가진 지도학습을 상정함
+ 데이터는 확률변수로 (x,y) ~ D라 표기
+ 결합분포 P(x,y)는 D를 모델링함
+ P(x)는 입력 x에 대한 주변확률분포로 y에 대한 정보를 주진 않음
    + 주변확률분포는 x에 대한 정보를 측정할 수 있기 때문에 이걸로 많은 통계적 분석을 돌려볼 수 있음
    + x에 대해서 덧셈, 적분을 해주면 y에 대한 주변확률 분포를 구할 수 있음
+ 조건부확률분포 P(x|y)는 데이터공간에서 입력 x와 출력 y사이의 관계를 모델링함
    + P(x|y)는 특정 클래스가 주어진 조건에서 데이터의 확률분포를 보여줌


#### 3. 이산확률변수 vs 연속확률변수
+ 확률변수는 확률분포 D에 따라 이산형(discrete)과 연속형(continuous) 확률변수로 구분하게 됨
+ 데이터공간 X x Y에 의해 결정되는 것이 아님.
    + 이산형 : 확률변수가 가질 수 있는 경우의 수를 모두 고려하여 확률을 더해서 모델링
        + 질량함수
        + 이산확률분포의 경우엔 급수를 사용

    ![CodeCogsEqn (2)](https://user-images.githubusercontent.com/44515744/106102497-d741f980-6182-11eb-9365-8e4e3d992ec0.gif)

    + 연속형 : 데이터 공간에 정의된 확률변수의 밀도(density) 위에서의 적분을 통해 모델링한다
        + P(x)는 누적확률분포의 변화율을 모델링하며 확률로 해석하면 안된다.
        + 연속확률분포의 경우엔 적분을 사용

    ![CodeCogsEqn (3)](https://user-images.githubusercontent.com/44515744/106102665-11130000-6183-11eb-9190-5b9ab9ff1cd9.gif)

#### 4. 조건부확률과 기계학습
+ 조건부확률 P(y|x)는 입력변수 x에 대해 정답이 y일 확률을 의미함
    + 연속확률분포의 경우 P(y|x)는 확률이 아니고 밀도로 해석함
    + 로지스틱 회귀에서 사용했던 선형모델과 소프트맥스 함수의 결합은 데이터에서 추출된 패턴을 기반으로 확률을 해석하는데 사용
    + 분류 문제에서 softmax은 데이터 x로부터 추출된 특징패턴과 가중치행렬 W을 통해 조건부확률 P(y|x)을 계산함
    + 회귀 문제의 경우 조건부기대값 E[y|x]을 추정함
    + 딥러닝은 다층신경망을 사용하여 데이터로부터 특징패턴을 추출함

#### 5. 기대값
+ 기대값은 확률변수가 취할 수 있는 모든 값들의 평균
+ 확률분포가 주어지면 데이터를 분석하는 데 사용 가능한 여러 종류의 통계적 함수(statical functional)
+ 기대값(expectation)은 데이터를 대표하는 통계량이면서 동시에 확률분포를 통해 다른 통계적 범함수를 계산하는데 사용
+ 기대값을 이용해 분산, 첨도, 공분산 등 여러 통계랑을 계산할 수 있음

#### 6. 몬테카를로 샘플링
+ 기계학습의 많은 문제들은 확률분포를 명시적으로 모를 때가 대부분
+ 확률분포를 모를 때 데이터를 이용하여 기대값을 계산하려면 몬테카를로 샘플링 방법을 사용해야함
    + 몬테카를로는 이산형이들 연속형이든 상관없이 성립함
    + 몬테카를로 샘플링은 독립추출만 보장되면 대수의 법칙에 의해 수렴성을 보장함
    + 적분값을 구하기 힘들 경우에는 몬테카를로 적분을 활용함 (샘플사이즈가 작으면 오차범위가 크고 참값에 멀어질 수 있음)



### 추가학습
#### 1. 역전파 알고리즘
+ 오차 함수의 음의 기울기를 구하면 모든 가중치와 편향, 이 모든 연결을 어떻게 변경해야 하는지를 알 수 있음, 이 방식으로 가장 효율적으로 오차함수를 최소화할 수 있음
+ 역전파 알고리즘은 그 많은 복잡한 기울기를 계산하기위한 알고리즘
+ 그래디언트의 부호는 상응하는 입력 벡터의 요소가 커져야할지 작아져야할지를 알려줌, 더 중요한건 이 요소들의 상대적인 크기
+ 이 크기는 어떤 요소를 조정하는 것이 더 큰 영향을 미칠지를 알려줌
+ cost 함수의 그래디언트 벡터(편미분 모음)를 각각의 가중치와 bias의 중요도를 표현하는 것으로 볼 수 있음
+ 오차함수의 출력은 기울기가 0.1인 가중치보다 기울기가 3.2인 가중치가 32배 더 민감하다고 해석할 수 있음, 가중치가 큰 기울기를 조정하는 것이 오차를 좀더 효과적으로 줄일 수 있음
+ 오차함수는 수만 가지의 훈련예제에 대해 예제당 특정 오차를 평균화함
+ 앞의 레이어에서 연결된 가장 밝은 뉴런과의 연결이 가장 큰 효과를 냄
    + 가중치에는 더 큰 활성 값이 곱해지기 때문
    + 가장 큰 가중치 증가, 가장 큰 연결 강화, 가장 활동적인 뉴런 사이에서 발생
+ 뉴런의 활성화를 증가시킬 수 있는 3가지 방법
    + 이전 계층의 모든 활성화를 변경하는 것
    + 양의 가중치 신경을 밝히고, 음의 가중치 신경을 밝혀서 뉴런의 활성치를 전체적으로 높인다.
+ 마지막 층에서 출력 뉴런 별로 변화 욕구를 계산하고 합산하면, 즉 각각의 뉴런이 변화할 필요가 있는 양을 모두 합하면, 근본적으로 두 번째에서 마지막 층으로 일어나기를 원하는 목록을 얻을 수 있다. 
    + 재귀적으로 동일한 모든 레이어에 프로세스를 적용할 수 있다. 
+ 학습 데이터를 무작위로 섞은 다음 이를 전체 배치로 나눈다. 미니 배치에 따라 단계를 계산함.
    + 미니배치는 꽤 좋은 근사값을 제공하고 계산 속도가 현저히 빠르다 ( 확률적 구배강하 )
+ 하나의 훈련 예가 가중치와 편향을 조금씩 움직이기를 원하는 지를 결정하기 위한 알고리즘
+ 위 아래로 가는 것뿐만이 아니라 변화에 대한 상대적인 비율이 비용을 가장 빠르게 감소
+ 하나의 배치에서 각 뉴런의 원하는 변화를 평균화함, 시간이 오래걸리므로 학습데이터를 나눠서 미니배치로 학습