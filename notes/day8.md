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

$$
\begin{bmatrix}
- & O_1 & - \\
- & O_2 & - \\
- & \vdots & - \\
- & O_n & - \\
\end{bmatrix}
=
\begin{bmatrix}
- & x_1 & - \\
- & x_2 & - \\
- & \vdots & - \\
- & x_n & - \\
\end{bmatrix}
\begin{bmatrix}
w_11 & w_12 & \cdotx & w_1p \\
w_21 & w_22 & \cdotx & w_2p \\
\vdots & \vdots & \ddots & \vdots \\
w_d1 & w_d2 & \cdotx & w_dp
\end{bmatrix}
+
\begin{bmatrix}
 &  & \cdots & \\
b_1 & b_2 & \cdots & b_p \\
&  & \cdots & \\
\end{bmatrix}
$$


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
