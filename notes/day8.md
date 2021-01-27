## [DAY 8] Pandas I / 딥러닝 학습방법 이해하기
### 1. [AI Math 5강] pandas I
> panel data 의 줄임말인 pandas는 파이썬의 데이터 처리의 사실상의 표준인 라이브러리입니다.
#### 1. Pandas란>
+ 구조화된 데이터의 처리를 지원하는 Python 라이브러리, Python의 엑셀
+ 고성능 Array 계산 라이브러리인 numpy와 통합하여 강력한 "스프레드시트" 처리 기능을 제공
+ 인덱싱, 연산용 함수, 전처리 함수 등을 제공
+ 데이터 처리 및 통계 분석을 위해 사용
+ Tabular 데이터 (엑셀 형태의 테이블 데이터)

#### 2. Tabular 구조
+ Data Table : 데이터 전체
+ Attribute, Field : 열
+ Instance, Row : 행
+ Feature Vector : 하나의 Feature에 있는 Vector 값들을 의미
+ Data : 하나의 값(셀)

#### 3. Pandas 설치&활용
+ 설치 : conda install pandas
``` python
import pandas as pd
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df_data = pd.read_csv(data_url, sep="\s+", header=None) # sep에는 정규식이 들어감 \s는 띄어쓰기

# 데이터 일부만 가져오기
df_data.head() # 처음부터 다섯줄 출력
df_data.head(n) # 처음부터 N줄 출력

df_data.columns = ["CRIM","ZN","INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"] # 컬럼값 지정
df_data.values # dataframe에 있는 데이터를 Numpy로 반환
```

#### 4. Pandas의 구성
+ Series : DataFrame 중 하나의 Column에 해당하는 데이터의 모음
    + Series는 사실상 Numpy의 Wrapper, Numpy를 Pandas에서 사용할 수 있게해주는 기능, 추가적으로 index 값이 존재함
    + index는 미지정시 0부터 지정됨
``` python
from pandas import Series, DataFrame
import pandas as pd

list_data = [1,2,3,4,5]
list_name = ["a","b","c","d","e"]
dict_data = {"a":1, "b":2, "c":3, "d":4, "e":5} # dict Type도 지정 가능(key -> index)
example_obj = Series(data = list_data, index=list_name) #index는 생략 가능, 0부터 자동으로 채워줌

example_obj["a"] = index를 넣어주면 해당하는 값을 조회할 수 있음
example_obj.astype(int) # value 자료형 정수로 변경
example_obj.astype(float) # value 자료형 실수로 변경
example_obj.index # index 리스트만 출력

example_obj.name = "number" # 시리즈에 이름을 지정가능, 잘 안씀
example_obj.index.name = "alphabet" # 인덱스에도 이름 지정 가능, 잘 안씀

indexes = ["a","b","c","d","e","g","h"]
series_obj_1 = Series(indexes) # index 값을 기준으로 series 생성 (index만 있다면 값은 NaN이 입력)
```

+ DataFrame : Data Table 전체를 포함하는 Object, Column Vector를 표현하는 Object
    + DataFrame에 값을 접근하려면 index와 Column을 알아야함
    + 컬럼마다 data Type이 다를 수 있음
    + Series를 사용해서 만들거나, CSV 파일을 불러와서 DataFrame을 생성
    + 데이터 보다 많은 columns 컬럼을 추가하면, 해당 컬럼들은 NaN(Not a Number) 값으로 채워짐
``` python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

raw_data = {'first_name':['Jason','Molly','Tina','Jake','Amy'], 'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 'age': [42, 52, 36, 24, 73], 'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']} # 일반적으로 이렇게 직접 구성 X
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city'])

df.first_name # 시리즈를 조회하는 방법1 (컬럼명)
df["first_name"] # 시리즈를 조회하는 방법2 (컬럼명)

df.loc[1] # index loacation을 넣어주면 index에 해당하는 값들을 가져옴
df["age"].iloc[1:] # 슬라이싱도 가능하면 iloc을 사용하면 index가 문자여도 숫자로 변형해서 접근해줌

s = pd.Series(np.nan, index=[49,48,47,46,451,2,3,4,5]) #loc은 index의 이름을 기준으로
s.iloc[:3] # iloc은 index number를 기준으로 -> 49, 48, 47 반환
df.loc[:, ["first_name", "last_name"]] # 컬럼을 함께 조회가능 ( list로 입력 )
df.debt = df.age > 40 # boolean index가 만들어짐, 이를 다른 Series에 할당할 수 있음, 값은 index에 맞춰서 들어가며 index가 없는 경우에는 NaN으로 채워짐

values = Series(data=["M", "F", "F"], index=[0, 1, 3])
df["sex"] = values # df에 동일한 index가 있다면 Series를 추가할 수 있다(Series에만 index가 있는 경우엔 해당 행만 추가 X)

df.T # Transpose, 행열 변환
df.values # 값 출력
df.to_csv() # CSV 변환
df.index # start=0, stop=5, step=1
del df["debt"] # 컬럼의 메모리 주소 자체를 삭제함
df.drop("debt", axis=1) # 컬럼 기준으로 axis=1을 제거함 (컬럼이 제거됨)

pop = {"Nevada": {2001: 2.4, 2002: 2.9}, "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6}}
DataFrame(pop) # 두 딕셔너리에 있는 키 값이 index가 됨
```

#### 5. Selection with column names
+ 엑셀 활용을 위해 xlrd 설치 : conda install --y xlrd
+ 데이터 조회는 컬럼은 문자열, index는 숫자로 지정
+ DataFrame은 기본적으로 내용을 수정하지않고 반환하는데 inplace 설정을 추가하면 DataFrame이 수정됨
``` python
from pandas import DataFrame, Series
import pandas as pd
import numpy as np

df["account"].head(3) # 한개의 column 선택시
df[["accout", "street", "state"]].head(3) # 한개 이상의 column 선택시
df[:3] # Row index 기준으로 데이터를 조회, 컬럼명을 넣는 방식과 일관성이 없음
df["account"][:3] # column 이름과 함께 row index 사용시, 해당 column만 조회
df["account"][[0,1,3]] # 1개 이상의 index
account_series[account_series < 250000] # boolean index를 사용해서 데이터 조회

df[["name","street"]][:2] # Column과 index number로 데이터 조회
df[:3][["first_name","last_name"]] # 순서바뀌어도 동일한 결과
df.loc[[211829,320563],["name","street"]] # index name과 Column으로 데이터 조회
df.iloc[:2,:2] # index number와 column명으로 데이터 조회

df.reset_index(drop=True) # 기존의 index를 지우고 새로 할당하여 df를 반환
df.reset_index(inplace=True,drop=True) # 기존 df의 내용이 변경됨

df.drop(1) # index 1을 제거, 변화를 적용하려면 inplace=True를 적용해야함
df.drop("city", axis=1) # 해당 컬럼 제거
```

#### 6. Series Operation
+ Numpy와 동일하게 Series 간의 연산을 지원
+ df는 column과 index를 모두 고려해야함
``` python
s1 = Series(range(1,6), index=list("abced"))
s2 = Series(range(5,11), index=list("bcedef"))
s1.add(s2)  # index를 기준으로 연산을 수행 ( 겹치는 index가 없다면 NaN값으로 반환 )
s1 + s2

df1 = DataFrame(np.arange(9).reshape(3,3), columns=list("abc"))
df2 = DataFrame(np.arange(16).reshape(4,4), columns=list("abcd"))
df1 + df2 # index와 column이 일치하는 데이터는 NaN으로 표시
df1.add(df2, fill_value=0)  # 데이터가 없는 영역은 0으로 채워서 NaN 값이 안나오게 함

s2 = Series(np.arange(10,14))
df.add(s2, axis=0) # axis를 기준으로 row broadcasting 실행, dataFrame에 s2 값이 broadcasting 되어 더해짐
```

#### 7. Map for Series
+ pandas의 series type의 데이터에도 map 함수 사용 가능
+ function 대신 dict, sequence형 자료등으로 대체 가능
``` python
s1 = Series(np.arange(10))
s1.head()

s1.map(lambda x: x**2).head(5) # 테이블에 연산 결과값 저장

z = {1: 'A', 2:'B', 3: 'C'}
s1.map(z).head(5) # dict의 키를 index로 하고, 이를 기준으로 데이터 교체

s2 = Series(np.arange(10,20))
s1.map(s2).head(5) # Series의 index를 기준으로 같은 위치의 데이터를 s2로 교체

df.sex.unique() # sex 컬럼의 유니크한 값들을 확인 가능
df["sex_code"] = df.sex.map({"male":0, "female":1})
df.head(5)

df.sex.replace({"male":0, "female":1}) # 딕셔너리로 값을 변경 가능
df.sex.replace(["male","female"], [0,1], inplace=True) # 값 리스트, 변경할 리스트를 추가하여 수정 가능
df.head(5)
```

#### 8. apply for dataFrame
+ map과 달리, Series 전체(column)에 함수를 적용
+ 입력 값을 series 단위로 handling 가능

``` python
df_info = df[["earn", "height", "age"]] # 특정 series만을 추출
f = lambda x : x.max() - x.min()
df_info.apply(f) # 각 컬럼별로 결과값 반환
'''
earn 결과값
height 결과값
age  결과값
'''

# 내장 연산 함수
df_info.sum() # 합계
df_info.mean() # 평균
df_info.std() # 표준편차
df_info.apply(np.sum)
df_info.apply(np.mean)

# series값의 반환
def f(x):
    return Series([x.min(), x.max()], index=["min","max"])
df_info.apply(f)

```

#### 8. applymap for dataFrame
+ series 단위가 아닌 element 단위로 함수를 적용
+ series 단위에 apply를 적용시킬 때와 같은 효과
``` python
f = lambda x : -x
df_info.applymap(f).head(5)
```

#### 9. Pandas Built-in Function
+ describe() : Numeric type 데이터의 요약 정보를 보여줌
    + 개수, 평균, 표준편차, 최소, 최대 등등
+ unique() : Series data의 유일한 값을 list로 반환
+ sum() : 기본적인 Column 또는 row 값의 연산을 지원
    + sub, mean, min, max, count, median, mad, var 등도 지원
+ isnull() : column 또는 row 값의 NaN 값을 boolean index로 반환
+ sort_values() : column 값을 기준으로 데이터를 Sorting
+ corr() : 상관계수를 구함
+ cov() : 공분산
+ corrwith() : 특정 컬럼을 기준으로 다른 컬럼의 상관계수를 구함
+ value_counts() : 오브젝트 타입일 경우 글자의 개수를 알려줌
``` python
df.describe()
df.sex.unique()
df.sum(axis=0) # Column 별 합계
df.sum(axis=1) # Row 별 합계
df.isnull()
df.isnull().sum() # 컬럼별 Null 값의 함
sort_values(["age","earn"], ascending=True) # 오른차순 정렬
df.sex.value_counts(sort=True) # 성별별 개수를 반환 
df.age.corr(df.earn) # age 컬럼과 earn 컬럼의 상관 계수
df.age.cov(df.earn) # age 컬럼과 earn 컬럼의 공분산
df.corrwith(df.earn) # earn 컬럼과 전체 컬럼의 상관 계수
df.corr() # 모든 컬럼들 간의 상관관계를 한번에 볼 수 있음

# 라벨링 코딩, 분류 값으로 치환하여 저장
key = df.race.unique()
value = range(len(key))
df["race"].replace(to_replace=key, value=value)

# 특정 컬럼만 연산
numueric_cols = ["earn", "height", "ed", "age"] # 연산할 컬럼을 조회
df[numueric_cols].sum() # 해당 컬럼들에 대한 합계를 조회

# Null 값의 비율 구하기
df.isnull().sum() / len(df) # NaN 값의 비율 구함

# 참고
pd.options.display.max_rows = 200 # 결과값을 출력에 대한 설정

(df.age < 45) & (df.age > 15) # 참이 되는 index를 반환
df.age[(df.age < 45) & (df.age > 15)].corr(df.earn) # boolean index를 활용한 셀렉션

# 상관 계수를 구할때 숫자를 사용하면 값이 난해하기 때문에 성별을 0,1로 치환
df["sex_code"] = df["sex"].replace({"male": 1, "female": 0})
df.corr()

# 값의 비중확인
df.sex_value_counts(sort=True) / len(df) # 전체에서 각 값들의 비율을 볼 수 있음
```

### [AI Math 5강] 딥러닝 학습방법 이해하기
> 선형모델은 단순한 데이터를 해석할 때는 유용하지만 분류문제나 좀 더 복잡한 패턴의 문제를 풀 때는 예측성공률이 높지 않습니다. 이를 개선하기 위한 비선형 모델인 신경망
#### 1. 신경망을 수식으로 분해
+ 데이터를 선형모델로 해석하는 방법을 배웠다면, 비선형모델인 신경망을 학습
+ 행 벡터 O는 데이터 X와 가중치 행렬 W 사이의 행렬곱과 절편 b 벡터의 합으로 표현
    + O = XW+b ( 데이터 * 가중치 + b = 선형모델)
    + 이때 출력 벡터의 차원은 d에서 p로 바뀜
        + O(n x p)
        + X(n x d)
        + W(d x p)
        + b(n x p)
        
![캡처](https://user-images.githubusercontent.com/44515744/106008541-bf269780-60fa-11eb-9aa2-9a076c07202f.JPG)

#### 2. softmax 함수
+ 출력 벡터 o에 softmax 함수를 합성하면 확률벡터가 됨
+ 소프트맥스(softmax) 함수는 모델의 출력을 확률로 해석할 수 있게 변환해주는 연산
+ 분류 문제를 풀 때 선형모델과 소프트맥스 함수를 결합하여 예측
    + 벡터를 확률벡터로 변환해주는 역할
+ 주어진 데이터가 어떤 특정 클래스에 속할 확률을 계산하는 것이 가능
    + [1, 2, 0] -> [0.24, 0.67, 0.09] # 총합은 1로 각 확률 계산이 가능
+ 추론을 하는 경우에는 원-핫(one-hot) 벡터로 최대값을 가진 주소만 1로 출력하는 연산을 
    + 소프트맥스는 학습을 할 때만 사용

``` python
def softmax(vec):
    denumerator = np.exp(vec - np.max(vec, axis=-1, keepdims=True)) # 지수 함수를 활용해서 계산, 오버플로를 방지하기 위해서 최대 값을 빼준다.
    numerator = np.sum(denumerator, axis=-1, keepdims=True)
    val = denumerator / numerator
    return val

vec = np.array([[1, 2, 0], [-1, 0, 1], [-10, 0, 10]])
softmax(vec)

```

#### 3. 신경망 함수
+ 신경망은 선형모델과 활성함수(activatin function)를 합성한 함수
+ 활성함수 σ는 비선형함수로 잠재벡터 z의 각 노드에 개별적으로 적용하여 새로운 잠재벡터 H를 만든다.
    + 활성함수는 다른 주소에 있는 출력값을 고려하지 않고 해당 주소에 있는 값만을 갖고 계싼(실수 값을 input으로 받음)
    + 선형모델로 나온 출력물을 비선형으로 변환할 수 있고, 이를 잠재벡터, 히든벡터 또는 뉴런이라고 한다.
    + 뉴런으로 이뤄진 모델을 신경망 뉴런 네트워크라고 부른다.
+ 1부터 L까지 순차적인 신경망 계산을 순전파(forward propagation라고 부름    

#### 4. 활성함수(Activation Function)
+ 실수 값을 입력으로 받아서 출력을 실수 값으로 출력하는 비선형(nonlinar) 함수
+ 활성함수를 쓰지 않으면 딥러닝은 선형모형과 차이가 없음
+ 시그모이드(sigmoid) 함수나 tanh 함수는 전통적으로 많이 쓰이던 활성함수지만 딥러닝에선 ReLU 함수를 많이 사용
+ 행벡터 x에 가중치 행렬(w1)을 통해 선형변환한 행벡터 O에 활성함수를 적용한 다음, 이를 또 가중치 행렬(w2)을 선형변환해서 출력하게 되면 가중치를 2번 적용하는 것이기 떄문에 2층(2-layers) 신경망이라고 한다.
    + 시그모이드 함수 : σ(x)
    + tanh 함수 : tanh(x)
    + ReLU 함수 : ReLU = max{0,x} # 활성함수로써 좋은 성질들을 갖고 있음, 0보다 작으면 0반환, 양수면 그대로 반환
+ 신경망 구현은 선형모델을 반복적으로 사용하는데 중간에 활성화 함수를 반드시 써야한다는 것이 신경망 구현의 핵심
+ 활성화 함수는 실수로 입력받고 실수로 출력, 출력된 실수는 벡터에 반영된다.
+ 다층(multi-layer) 퍼셉트론(MLP)는 신경망이 여러층 합성된 함수
    + 층이 깊을수록 목적함수를 근사하는데 필요한 뉴런(노드)의 숫자가 훨씬 빨리 줄어들어 효율적으로 학습이 가능

#### 5. 딥러닝 학습원리 : 역전파 알고리즘
+ 딥러닝은 역전파(backpropagation) 알고리즘을 이용하여 각 층에 사용된 패러미터에한 미분을 구해서 파라미터를 업데이트함, 행렬들의 원수의 모든 개수만큼 경사하강법이 적용되게 됨.
+ 선형 모델에서는 그레디언트 벡터는 한층에서만 계산하는 것이기 때문에 한꺼번에 구할 수 있지만 딥러닝은 층이 여러개라서 역전파 알고리즘을 써야함
+ 각각의 가중치 행렬의 손실에 대해서 미분을 계산할때 사용
+ 윗층에 있는 그레디언트를 계산한 다음에 아래층에 있는 그레디언트를 계산하는 방식
+ 출력에서부터 그레디언트를 밑에층으로 전달하면서 가중치를 조정함

#### 6. 역전파 알고리즘 원리 이해하기
+ 합성함수 미분법인 연쇄법칙(chain-rule) 기반 자동미분(auto-differentation)을 사용
+ 순전파는 미분값을 저장할 필요가 없지만 역전파를 사용하기 위해서 미분값을 저장해야해서 미분 값을 저장해야함

![캡처](https://user-images.githubusercontent.com/44515744/105967618-668ad680-60c9-11eb-8925-f5546c443007.JPG)


+ 딥러닝의 뉴런에 해당하는 값을 텐서라고 표현하는데 이를 컴퓨터가 메모리에 저장하고 있어야지 역전파 알고리즘을 사용할 수 있음
+ x에 대한 미분을 알고 싶으면 같이 사용된 x,y라는 값을 알고있어야지 미분을 사용할 수 있음
+ SGD을 이용해서 각각의 파라미터들을 미니배치로 번갈아가면서 학습, 주어진 목적식을 최소화할 수 있는 파라미터들을 찾을 수 있고 이것이 딥러닝의 학습원리임

### 추가학습
#### 1. 2차원 -> 3차원 행렬에 대해
+ 2차원 입력 벡터와 3차원 출력 벡터는 완전히 다른, 연결되지 않는 공간에 존재하는 벡터
+ 아래의 3 X 2 행렬은 i-hat이 (2,-1,-2)이고 j-hat은 (0,1,1)이다.
    + 기하학적 해석으로는 2차원 공간을 3차원 공간에 매핑하는 것으로 볼 수 있음( 2차원에서는 평면이지만 3차원에서는 기울어져 있음)

![gif](https://user-images.githubusercontent.com/44515744/105968442-70610980-60ca-11eb-8240-8499661b2126.gif)

+ 아래의 2 X 3 행렬은 3차원에서 시작했지만 세 기저벡터의 변환후에 좌표값 2개만 남으므로 2차원으로 이동해야함 ( 3차원 -> 2차원_ 일종의 투사, 투영)

![CodeCogsEqn](https://user-images.githubusercontent.com/44515744/105968921-05fc9900-60cb-11eb-89aa-a49c6d3912a2.gif)

#### 2. 선형대수의 내적(dot prodcut)
+ 내적(Dot Product)은 같은 좌표값으로 짝을 지어 곱하고 모두 더한다. 

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/44515744/105970498-c931a180-60cc-11eb-953d-dfaca873f96d.gif)

+ 아래 연산이 의미하는 것은 벡터 v에 벡터 w를 투영(project:벡터 w에서 벡터 v와 원점을 잇는 선에 수직으로 만남)하고, 원점에서부터 투영된 지점까지의 길이를 벡터 v의 길이와 곱하는 것
    + 벡터 w의 투사체가 벡터 v와 반대 방향이면 내적은 음수
    + 벡터 w의 투사체가 벡터 v와 같은 방향이면 내적은 양수
    + 벡터 w의 투사체가 0 벡터에 위치하면 내적은 0


![gif (1)](https://user-images.githubusercontent.com/44515744/105971058-6987c600-60cd-11eb-9313-fd909edc16bd.gif)

![캡처](https://user-images.githubusercontent.com/44515744/105971266-ace23480-60cd-11eb-8227-6312f22e2abc.JPG)

+ 내적하는 두 행렬의 순서가 바껴도 결과값은 동일하다.

#### 3. 내적 수식 이해하기
+ 1x2 matrics <-> 2d vectors
+ 1x2 행렬과 2차원 벡터사이에는 관련성이 있음
    + 벡터 숫자 표현을 옆으로 기울여서 연관 행렬을 얻거나, 행렬을 세워서 연관 벡터를 얻는 관련성

![gif (2)](https://user-images.githubusercontent.com/44515744/105973511-29761280-60d0-11eb-8b3d-b9618a7deb03.gif)

![캡처](https://user-images.githubusercontent.com/44515744/105975390-40b5ff80-60d2-11eb-89f0-32e88be7e29c.JPG)

+ 2차원에서 원점을 지나는 수선과 이것에 대한 기저벡터 u가 있다. 다른 2차원 벡터들을 이 수선에 투영을 하게 되면, 이것은 사실상 2차원 벡터를 입력받아 숫자를 내놓은 함수를 정의한 것과 동일
    + 이 함수의 출력은 수선의 숫자, 2차원 벡터가아님
+ 결과 공간이 수선인 선형변환을 가지고 있다면, 어떻게 정의하든지 간에 어떤 벡터가 그 변환에 대응하고 있을 것
+ 변환의 적용은 벡터의 내적을 구하는 것과 같음을 알 수 있음
+ 내적은 벡터가 같은 방향을 가리키는지 알아내는 유용한 도구
+ 내적은 투영을 이해하는데 매우 유용한 기하학적 도구
+ 두 벡터를 함께 내적하는 것은 두 벡터 중 하나를 변환 인자로 보는 것
+ 변환의 적용은 내적을 구하는 것과 같음을 알 수 있음

#### 4. 신경망이란 무엇인가
+ 뉴런은 0.0 ~ 1.0 사이의 하나의 숫자를 담는다.
+ 다층 퍼셉트론(Multi-Layer Perceptron)은 입력층(Input Layer), 은닉층(Hidden Layer), 출력층(Output Layer)로 구성 
+ 신경망 안에서 뉴런에 입력되는 값을 입력값(Activation)이라고 한다.
    + 큰 입력값이 주어질수록 각각의 신경망이 더 큰 정도로 활성화됨
+ 출력층(Output Layer)의 Activation은 각 뉴런이 대표하는 값과 얼마나 일치하는 지를 나타냄
+ 신경망은 기본적으로 한 층에서의 활성화가 다음 층의 활성화를 유도하는 방식으로 작동
+ 신경망의 가장 중요한 점은 한 층에서의 활성화가 다른 층의 활성화를 불러일으키는 지에 관한 점
    + 생물의 뉴런이 작동하는 방식과 유사
+ 출력층에서 Activation이 가장 큰 값이 신경망에서 선택된 출력값

#### 5. Hidden Layer의 역할
+ Layer에서 모든 Activation을 가져와서, 각 뉴런의 가중치를 주고 모두 더함
+ 중요한 영역을 제외하고 가중치를 0으로 만들면, 해당 영역에만 가중치를 준 것과 같은 상황이 됨
+ 가중치를 준 영역 주변에 음수 가중치를 줘서 경계를 구분짓는다.
+ 가중치를 준 값을 합하면 어떤 값이든지 나올 수 있으나, 신경망에서 원하는 건 0과 1사이의 값
+ 이때, 이 가중치를 준 값의 합을 0과 1 사이의 숫자로 만들어 주는 함수에 추가함
    + 시그모이드함수 : 매우 큰 양수는 1, 매우 작은 음수는 -1, 0 주위에서는 계속 증가함
+ 뉴런의 활성화는 기본적으로 가중치의 합이 얼마나 더 양에 가까운지에 따라 정해져있음
+ 가중치의 합이 0을 넘을때 뉴런이 활성화 되는 것을 원하는 게 아닐 수 있음, 예를 들어 합이 10을 넘을 때 활성화되기를 원할 수 있음 ( 활성화되지 않기 위한 조건을 달아야함 Bias for inactivity)
    + 이럴 때는 가중치에 -10 처럼 다른 음의 숫자를 더해줌, 활성화함수에 값을 집어넣기 전에
+ 가중치는 두번째 레이어가 선택하려는 뉴런의 픽셀 패턴을 알려줌
+ bias는 뉴런이 활성화되려면 가중치의 합이 얼마나 더 높아야 하는지를 알려줌
+ 각층의 뉴런들은 서로 연결되어 있고, 이 연결들은 각자의 가중치를 갖음
+ 또 각각의 뉴런은 활성화함수로 압축하기 전에 가중치에 더한 값인 bias를 갖음
+ N개의 뉴런으로 구성된 히든레이어는 N개의 bias를 가진 입력량 X N개의 가중치를 의미함
+ 뉴런은 이전 층의 뉴런의 출력을 모두 받아서 0,1 사이의 수를 만들어내는 함수라고 생각하는게 더 정확
+ 현재는 시그모이드 함수보다 ReLU함수(Rectified Linear Unit)를 사용, 0과 a에 max함수를 취한 것으로 a는 뉴런의 활성치를 나타내는 함수 ( 임계값을 넘기면 그 값을 출력하고, 넘기지 못하면 0을 출력_ 단순화된 버전)


#### 6. Learning에 대한 정의
+ 컴퓨터가 실제로 해당 문제를 스스로 해결하기 위해서 수많은 수치들을 찾기 위한 알맞은 환경을 얻는다는 것

#### 7. 경사 하강, 신경 네트워크가 학습하는 방법
+ 제대로 된 출력은 대부분의 뉴런이 정답을 제외하고 0의 활성치를 가져야 함
+ cost는 잘못된 출력과 원하는 출력의 활성값의 차의 제곱을 모두 더하는 것(신경망이 이미지를 올바르게 분류 할 때 이 합계가 작다)
+ AI 엔지니어가 해야할일은 수만 가지의 학습 예시 전체에 대해 평균 cost를 검토하는 것
    + 이 평균 cost는 신경망이 얼마나 엉망인지를 측정하는 수단
+ 신경망 자체가 기본적으로 함수이다. (N개의 값을 입력받아서 M개의 결과를 출력)   
+ 이 모든 것은 가중치와 bias에 의해 매개변화 되어 있음 ( 가중치와 bias를 어떻게 바꿔야 나아지는지 알려줌)
+ C(w)라는 함수에서 출력값을 낮추는 위해서는 어떤 방향으로 이동해야하는 지를 알아야함. 즉, 현재 시점의 기울기를 알면 어디로 이동해야할지를 알 수 있음 ( local Minimum에 도달 가능)
+ 출발점을 어디로 했느냐에 따라서 각기 다른 골짜기에 도착하게 되지만, 도착한 골짜기가 함수에서 가장 최소 값이라는 보장은 없음
+ 이동 거리를 기울기에 비례해서 결정한다면, 기울기가 줄어들수록 이동하는 거리가 작아지고 이는 오버슈팅을 방지함
+ 입력 값이 2개고 하나의 출력 값을 갖는 함수를 상상하면, 함수의 기울기 대신에 이 입력 공간에서 어느 방향으로 움직일지를 결정해야함 ( 다변수 미적분은 편미분을 사용)
+ 그래디언트 벡터의 길이는 가장 가파른 경사가 얼마나 가파른지에 대한 지표
    + 그래디언트 벡터는 각 변수 별로 편미분을 계산한 것
    + cost 함수의 그래디언트의 음의 방향은 그냥 벡터일 뿐, 이 벡터는 어떤 방향이 cost 함수를 가장 빠르게 감소시키는지 알려줌
    + cost 함수는 가중치와 bias들을 cost 함수 값이 줄어드는 방향으로 조정할 것
