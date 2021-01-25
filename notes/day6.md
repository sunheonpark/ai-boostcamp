## [DAY 6] Numpy / 벡터 / 행렬
### [AI Math 0강] numpy
> numpy(넘파이)는 Numerical Python의 약자로 일반적으로 과학계산에서 많이 사용하는 선형대수의 계산식을 파이썬으로 구현할 수 있도록 도와주는 라이브러리입니다.

#### 1. 코드로 방정식 표현하기
+ 다양한 Matrix 계산을 어떻게 만들 것인가?
+ 굉장히 큰 Matrix에 대한 표현
+ 처리 속도 문제 = Python은 Interpreter ㅇ너어

#### 2. 파이썬 과학 처리 패키지 - numpy
+ 파이썬의 고성능 과학 계산용 패키지
+ Matrix와 Vector와 같은 Array 연산의 사실상의 표준
+ 일반 List에 비해 빠르고, 메모리 효율적
+ 반복문 없이 데이터 배열에 대한 처리를 지원함
+ 선형대수와 관련된 다양한 기능을 제공함
+ C, C++, 포르탈ㄴ 등의 언어오 ㅏ통합 가능

#### 3. ndarray란?
+ 일반적으로 numpy는 np라는 alias(별칭) 이용해서 호출함
+ numpy는 하나의 데이터 type만 배열에 넣을 수 있음
+ list와는 다르게 dynamic typing not supported
+ ndarray는 N-dimensional Array의 약자임
``` python
# 기본 활용법
import numpy as np

test_array = np.array([1, 4, 5, 8], dtype=float)  # ndarray 생성
```

#### 4. Numpy Array vs Python List
+ Numpy Array는 메모리 공간에 값이 차례대로 저장됨 (연산이 빠르다)
+ Python List는 주소 값이 리스트에 저장되고, 주소를 통해서 값을 가져온다. (데이터 조작이 편함)
``` python
# 기본 활용법
a = [1,2,3,4,5]
b = [5,4,3,2,1]
a[0] is b[-1] # True - 파이썬에서 상수 -5 ~ 256은 정적 메모리를 사용하여 메모리 주소가 같다.

a = np.array(a)
b = np.array(b)
a[1] is b[-2] # False - ndarray는 특정 메모리를 할당하고 여기에 값을 순서대로 저장하기 때문에 값마다 메모리 주소가 다르다
```

#### 5. Array Shape
+ shape : numpy array의 dimension의 구성을 반환함, array의 크기, 형태 등에 대한 정보
+ RANK가 늘어날 수록 Shape 값은 한칸씩 밀린다. 추가되는 차원이 앞에 추가됨
    + (4,)
    + (3, 4) x, y
    + (4, 3, 4) z, x, y
+ dtype : numpy array의 데이터 type을 반환함 
+ Tensor는 선형대수에서 값을 표현하는 기법

Rank | Name | Example
---- | ---- | ----
0 | scalar | 7
1 | vector | [10, 10]
2 | matrix | [[10, 10], [15,15]]
3 | 3-tensor(3rd order tensor) | [[[1,5,9],[2,6,10]],[[3,7,11],[4,8,12]]]
n | n-tensor(n order tensor) | ..

#### 6. Array dtype
+ ndarray의 single element가 가지는 data type
+ 각 element가 차지하는 memory의 크기가 결정됨
``` python
# 기본 활용법
np.array([[1, 2, 3], [4.5, 5, 6]], dtype=int)
np.array([[1, 2, 3], [4.5, "5", "6"]], dtype=np.float32)
```

#### 7. Array 기타 Attrubute
+ Array의 메모리 크기를 반환
``` python
# 기본 활용법
np.array([[1, 2, 3], [4.5, 5, 6]], dtype=float32).nbytes # 6 * 4bytes = 24 bytes
np.array([[1, 2, 3], [4.5, "5", "6"]], dtype=np.int8).nbytes #  6 * 1bytes = 6 bytes

np.array([[1, 2, 3], [4.5, 5, 6]], int).ndim # 2 - 디멘션 개수
np.array([[1, 2, 3], [4.5, 5, 6]], int).size # 6 - 데이터의 개수
```

#### 8. reshape
+ reshape : Array의 shape의 크기를 변경함, element의 개수는 동일
+ 결과가 반환되므로 기존 데이터에는 변화가 없다.
    + (2, 4) -> (8,)
+ 

``` python
# 기본 활용법
test_matrix = [[1,2,3,4], [1,2,5,8]]
np.array(test_matrix).shape # (2, 4)
np.array(test_matrix).reshape(8,).shape # (8,)

np.array(test_matrix).reshape(-1,2).shape # -1은 size를 기반으로 개수를 선정, 컬럼이 2이므로 로우는 4가된다.
```

#### 9. flatten
+ 다차원 array를 1차원 array로 변환한다.
``` python
# 기본 활용법
test_matrix = [[[1,2,3,4],[1,2,5,8]], [[1,2,3,4], [1,2,5,8]]]
test_matrix = np.array(test_matrix)
test_matrix.shape # (2, 2, 4)
test_matrix.flatten().shape # (16.)
```

#### 10. indexing & slicing
+ indexing
    + list와 달리 이차원 배열에서 [0,0] 표기법을 제공함 # [x][y] <- X
``` python
# 기본 활용법
print(a[0,0]) # 값 조회 1
print(a[0][0]) # 값 조회 2
test_example[0,0] = 10 # 값 할당
```

+ slicing
    + list와 달리 행과 열 부분을 나눠서 slicing이 가능
    + matrix의 부분 집합을 추출할 때 유용함

``` python
# 기본 활용법
a = np.array([[1,2,3,4,5], [6,7,8,9,10]], int)
a[:, 2:] # 전체 Row의 2열 이상
a[1,1:3] # 1행의 1열 ~ 2열
a[1:3] # 1 Row ~ 2Row의 전체

a[1] # array([6,7,8,9,10]) # 1-dimension만 나온다
a[1:3] # array([[6,7,8,9,10]]) # 2-dimension으로 나온다

a = np.arange(100).reshape(10, 10) # 0~99가 10x10으로 만들어진다.
a[:, -1] # 모든 행의 -1, 즉 마지막 컬럼을 가져온다.

a[:,::2] # 전체 행에서 컬럼은 2step으로 조회
a[::2,::3] # 2 step 행, 3 step 컬럼 조회
```

#### 11. Creation Functions
+ arange : array의 범위를 지정하여, 값의 list를 생성하는 명령어
``` python
# 기본 활용법
np.arrange(30) # integer로 0부터 29까지 배열 추출
np.arrange(0, 5, 0.5) # 0 <= n < 5까진 0.5 스텝으로 배열 추출
```
+ zeros : 0으로 가득찬  ndarray 생성
``` python
# 기본 활용법
np.ones(shape(10,), dtype=np.int8)
np.ones((2,5))
```
+ empty - shape만 주어지고 비어있는 ndarray 생성 ( memory initialization이 X )
    + 공간은 지정되지만 초기화가 안되므로 이전에 사용하던 다른 값이 존재할 수도 있음
``` python
np.empty(shape
# 기본 활용법=(10,), dtype=np.int8)
np.empty((3,5))
```
+ something_like : 기존 ndarray의 shape 크기 만큼 1, 0 또는 empty array를 반환
``` python
# 기본 활용법
test_matrix = np.arrange(30).reshape(5,6)
np.ones_like(test_matrix) # test_matrix의 shpae 형태로 1로 채워진 array를 생성
np.zeros_like(test_matrix) # test_matrix의 shpae 형태로 0로 채워진 array를 생성
np.empty_like(test_matrix) # test_matrix의 shpae 형태로 empty array를 생성
```

+ identity : 단위 행렬(i 행렬)을 생성함
    + matrix 형태를 기준으로 대각행렬이 1로 되어 이있는 array를 반환
``` python
# 기본 활용법
np.identity(n=3, dtype=np.int8) # (3,3)
np.identity(5) # (5,5)
```

+ eye : 대각선이 1인 행렬, k값의 시작 index의 변경이 가능
``` python
# 기본 활용법
np.eye(3) # (3,3)에서 대각선이 1인 행력
np.eye(3,5, k=2) # k=2는 시작하는 컬럼이 2라는 의미
```

+ diag : digonal의 약자 생성이 되어있는 Matrix에서 대각 행렬의 값을 추출함
``` python
# 기본 활용법
matrix = np.arrange(9).rshape(3,3)
np.diag(matrix) # array([0,4,8])
np.diag(matrixm k=1) # array([1,5]) eye와 동일하게 시작열 지정 가능
```

+ random sampling : 데이터 분포에 따른 sampling으로 array를 생성
``` python
# 기본 활용법
np.random.uniform(0, 1, 10).reshape(2,5) # 균등분포
np.ramdom.normal(0,1,10).resahpe(2,5) # 정규분포
np.ramdom.exponential(scale=2, size=100) # 공부하기
```

#### 12. Operation Functions
+ axis : 모든 operation function을 실행할 떄 기준이 되는 dimension 축
    + 3 x 4를 만들면 3은 axis=0 4는 axix=1이 된다 새로 dimension이 추가될 때마다 1씩 밀린다
![캡처](https://user-images.githubusercontent.com/44515744/105655503-8375ad00-5f03-11eb-9efd-2fbf52beceaf.JPG)
``` python
# 기본 활용법
test_array = np.arange(1,13).reshape(3,4)
test_array.sum(dtype=np.float) # 모든 요소를 더한 값을 반환
text_array.sum(axis=1) #4 즉, 열을 기준으로 더한 값 array([10,26,42])
text_array.sum(axis=0) #3 즉, 행을 기준으로 더한 값 array([15,18,21,24])
(3,3,4) # 3 = axix 0, 3 = axis 1, 4 = axis 0

test_array.mean() # 평균
test_array.mean(axis=0) # 평균
test_array.std() # 표준편차
test_array.std(axis=0) #표준편차
```

#### 13. concatenate
+ numpy array를 합치는 함수
``` python
# 기본 활용법
a = np.array([1,2,3])
a = np.array([2,3,4])
np.vstack((a,b)) # 수직으로 쌓는다 Vertical Stack, array([[1,2,3],[2,3,4]])

a = np.array([[1],[2],[3])
a = np.array([[2],[3],[4])
np.hstack((a,b)) # 수평으로 쌓는다 Horizontal STack. array[[1,2],[2,3],[3,4]]

a = np.array([1,2,3])
a = np.array([2,3,4])
np.concatenate((a,b), axis=0) # axis는 붙였을때 생성되는 결과값의 axix라고 생각

a = np.array([[1],[2],[3])
a = np.array([[2],[3],[4])
np.concatenate((a,b), axis=1) # axis는 붙였을때 생성되는 결과값의 axix라고 생각

a = np.array([[1,2], [3,4]])
b = np.array([5, 6])
b.reshape(-1,2) # 축 추가하는 방법
b = b[np.newaxis, :] # 축 추가하는 방법2
np.concatenate((a, b.T), axis=1) # b.T는 행과 열을 변경
```

#### 14. Array Operation
+ numpy는 array간의 기본적인 사칙 연산을 지원함
+ Element-wise Operations Array간 Shape가 같을 때 일어나는 연산
``` python
# 기본 활용법
test_a = np.array([[1,2,3], [4,5,6]], float)
test_a + test_a # 요소간 합
test_a - test_a # 요소간 빼기
test_a * test_a # 요소간 곱

test_a = np.arange(1,7).reshape(2,3)
test_b = np.arange(7,13).reshape(3,2)
test_a.dot(text_b) # Matrix의 기본 연산, Dot Product라 함, 앞의 열과 뒤의 행이 같은 경우 연산 가능
```
#### 15. Transpose
+ 전치행렬은 T attribute를 사용한다. 로우와 컴럼을 바꾼다
``` python
# 기본 활용법
test_a = np.array([[1,2,3], [4,5,6]], float)
test_a.T # 전치행렬1
test_a.transpose() # 전치행렬2
```

#### 16. broadcasting
+ Shape이 다른 배열 간 연산을 지원하는 기능
+ Scalar - Vector 외에도 Vector - matrix 간의 연산도 지원함
+ 동일한 shape의 형태로 값이 채워져서 연산되는 형태

``` python
# 기본 활용법
test_matrix = np.array([[1,2,3],[4,5,6]], float)
scalar = 3
test_matrix + scalar # Matrix의 각 요소에 scalar가 더해진다.
test_matrix * scalar
test_matrix // scalar
test_matrix ** scalar

test_matrix = np.arange(1, 13).reshape(4, 3)
test_vector = np.array(10, 40, 10)

test_matrix + test_vector
```

#### 17. Numpy performance
+ %timeit 함수를 사용해서 속도측정이 가능
+ 일반적으로 속도는 for loop < list comprehension < numpy
+ 100,000,000번의 loop이 돌 때, 약 4배 이상의 성능 차이
+ Numpy는 C로 구현되어 있어, 성능을 확보하는 대신 dynamic typing을 포기함
+ 대용량 계산에서는 가장 흔히 사용됨
+ 계산이 아닌 할당에서는 연산 속도의 이점은 없음

#### 18. Numpy Comparisons
+ broadcasting 방식으로 각 요소에 대한 조건 검사가 가능
+ 배열의 크기가 동일 할 때 element간 비교의 결과를 boolean type으로 반환
``` python
a = np.arange(10)
a < 4 # broadcasting이 되어 각 요소마다 비교가 됨 [True, True, True .....] 반환 Boolean Array

np.any(a>5), np.any(a<0) # or 조건, 요소 중 하나라도 True면 전체가 True
np.all(a>5) # and 조건, 요소 모두가 참이면 True 아니면 Fals

a = np.array([1, 3, 0], flaot)
np.logical_and(a > 0, a < 3) # and 조건의 condition

b = np.array([True, False, True], bool)
np.logical_not(b) # NOT 조건의 condition

c = np.array(False, True, False], bool)
np.logical_or(b, c) # OR 조건의 condition

np.where(a > 0, 3, 2)  # 조건이 True이면 3, False이면 2를 입력
np.where(a>5)[0]  # 조건에 만족하는 index를 반환 ( 조건에 맞는 요소만 남긴다), Tuple이므로 [0]으로 () 제거

a = np.array([1, np.NaN, np.Inf], float)
np.isnan(a) # np.NaN일 경우에만 True
np.isfinite(a) # 한정된 숫자만 True, 

np.argmax(a) # 가장 큰 값의 index를 반환
np.argmin(a) # 가장 작은 값의 index를 반환

np.argmax(a, axis=1) # axis 1을 기준으로 가장 큰 값
np.argmax(a, axis=0) # axix 0을 기준으로 가장 작은 값

a.argsort() # 오름차수는으로 array의 인덱스 값을 꺼내준다.
a.argsort()[::-1] # 역순으로 뽑는다.

```
#### 19. boolean Index
+ 특정 조건에 따른 값을 배열 형태로 추출
+ Comparison Operation 함수들도 모두 사용 가능
+ boolean array의 shape이 동일해야함
``` pyhton
test_array = np.arange(10)
test_array[test_array > 3] # 조건이 True인 index의 element만 추출
```

#### 20. fancy Index
+ numpy는 array를 index value로 사용해서 값 추출
+ shape은 달라도 되나 크기를 벗어나면 안됨
+ matrix 형태의 데이터도 가능
``` python
a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int) # 반드시 integer로 선언(index 값)
a[b] #bracket index, b 배열의 ㄱ밧을 index로 하여 a의 값들을 추출
a.take(b) # a[b]와 동일

a = np.array([[1, 4], [9, 16]], float)
b = np.array([0, 0, 1, 1, 0], int) # x 값
c = np.array([0, 1, 1, 1, 1], int) # y 값
a[b, c] # b를 row index, c를 column index로 변환
a[b] # row만 넣어주면 row 값만 가져온다. 0 - [1,4] , 1 - [9, 16]
```

#### 21. Numpy IO
+ loadtxt & savetxt : text type의 데이터를 읽고, 저장하는 기능
``` python
a = np.loadtxt("./populations.txt", delimiter="\t")
a_int = a.astype(int)
a_int_3 = a_int[:3]
np.savetxt("int_data_2.csv", a_int_3, fmt="%.2e", delimiter=",")

np.save("npy_test_object.npy", arr=a_int_3) # pickle 형태로 저장
np.load(file="npy_test_object.npy")
```
------------------------
### [AI Math 1강] 벡터가 뭐에요?
#### 벡터의 정의
+ 벡터는 숫자를 원소로 가지는 리스트(list) 또는 1차원 배열(Array)
+ 세로로 된 벡터를 열 벡터(x), 가로로 된 벡터는 행 벡터(x^T)라고 함
+ 벡터에 있는 숫자의 개수를 벡터의 차원이라고 함
+ 벡터는 공간에서 한 점을 나타냄, 원점으로부터 상대적 위치를 표현
+ 벡터에 숫자를 곱해주면 길이만 변함(스칼라곱)
+ 곱해지는 숫자가 음수일 경우에는 방향이 반대가 됨
+ 벡터끼리 같은 모양을 가지면 덧셈, 뺄셈, 곱셈(성분곱, Hadamard product)을 계산할 수 있음
+ 벡터 x는 0벡터에서의 상대적 위치이동을 의미 x = 0 + x
+ 벡터 y+x는 0벡터에서부터 상대적으로 위치 이동한 최종 결과 y+x = 0 + y + x
+ 벡터의 뺄셈은 방향을 뒤집은 덧셈

#### 벡터의 노름
+ 벡터의 노름은 원점에서부터의 거리
+ 노름은 임의의 차원 d에 대해 성립함
+ 노름은 ||x||로 표시
+ ||x||1 , L1 노름은 각 성분의 변화량을 모두 더하는 개념
+ ||x||2 , L2 노름은 피타고라스 정리를 이용해 유클리드 거리를 계산
+ 원점에서부터 거리가 1인 점들의 집합을 원으로 표시함, 노름의 종류에 따라 기하학적 성질이 달라짐
    + L1 : 마름모로 표현 ( Robust 학습, Lasso 회귀)
    + L2 : 원으로 표현 ( Laplace 근사, Ridge 회귀)

``` python
def l1_norm(x):
    x_norm = np.abs(x)
    x_norm = np.sum(x_norm)
    return x_norm

def l2_norm(x):
    x_norm = x*x
    x_norm = np.sum(x_norm)
    x_norm = np.sqrt(x_mnorm)
    return x_norm
```

#### 벡터 사이의 거리를 계산
+ L1, L2-노름을 이용해 두 벡터 사이의 거리를 계산할 수 있음
+ 두 벡터 사이의 거리를 계산할 때는 벡터의 뺄셈을 이용
+ 뺄셈을 거꾸로 해도 거리는 동일함
    + ||y - x|| = ||x-y||

#### 벡터 사이의 각도를 계산
+ L2 노름인 || ||2만 각도 계산이 가능하다.
+ 제2 코사인 법칙에 의해 두 벡터 사이의 각도를 계산할 수 있다.
+ 분자를 쉽게 계산하는 방법이 내적이다.
+ 내적(inner product) = 두 벡터의 성분곱을 취한다음 더한다. 
+ 분자= x,y의 내적 / 분모 = ||x||2||y||2 L2노름을 곱한 것
+ 내적은 np.inner을 이용해서 계산
``` python
def angle(x, y):
    v = np.inner(x,y) / (l2_norm(x) * l2_norm(y))
    theta = np.arcccos(v) # 아크코싸인을 구하면 각도를 계산할 수 있음
    return theta
```

#### 내적이란?
+ 내적은 정사영(orthogonal projection)된 벡터의 길이와 관련있다.
+ Proj(x)는 벡터 y로 정사영된 벡터 x의 그림자를 의미한다. (x에서 시작하여 y에 수직으로 도달하는 좌표)
+ Proj(x)의 길이는 코사인법칙에 의해 ||x|| 노름에 코싸인을 곱한 것과 같다.
+ 내적은 정사영의 길이를 벡터 y의 길이 ||y||만큼 조정한 값이다.
+ 내적은 두 벡터의 유사도를 측정하는데 사용 가능하다.   

### [AI Math 2강] 행렬이 뭐에요?
+ 행렬(matrix0은 벡터를 원소로 가지는 2차원 배열
+ numpy에서는 행(row)이 기본 단위
+ 벡터는 소문자, 행렬은 대문자 볼드체로 표시
+ 행렬은 벡터를 원소로 가지는 2차원 배열
+ 행렬은 행과 열이라는 인덱스를 가짐
+ 행렬의 특정 행(열)을 고정하면 행(열)벡터라 부름
+ 전치행렬(transpose matrix) : 행과 열의 index가 바뀐 행렬
+ 백터가 공간에서 한 점을 의미한다면 행렬은 여러 점들을 나타냄
+ 행렬의 행벡터 xi는 i번째 데이터를 의미
+ 행렬의 xij는 i번째 데이터의 j번째 변수의 값을 의미

#### 행렬의 덧셈, 뺄셈, 성분곱, 스칼라곱
+ 행렬끼리 같은 모양을 가지면 덧셈, 뺄셈을 계산할 수 있음
+ 성분곱은 벡터와 동일, 각 인덱스 위치끼리 곱함
+ 스칼라곱도 동일

#### 행렬 곱셈
+ 행렬 곱셈(matrix multiplication)은 i번째 행벡터와 j번째 열벡터 사이의 내적을 성분으로 가지는 행렬을 계산
    + 행렬곱은 X의 열의 개수와 Y의 행의 개수가 같아야 한다.
+ 행렬 곱셈의 순서가 바뀔 경우 데이터가 달라진다.
``` python
x = np.array([[1,-2,3],[7,8,0],[-2,-1,2]])
y = np.array([[0,1], [1,-1], [-2, 1]])
X @ Y # numpy에서는 @연산으로 행렬곱을 한다.

```

#### 행렬의 내적
+ 넘파이의 np.inner은 i 번재 행벡터와 j번째 행벡터 사이의 내적을 성분으로 가지는 행력을 계선
+ 행의 크기가 같아야 함
+ 수학에서 말하는 내적과는 다름

#### 행렬을 이해하는 방법2
+ 행렬은 벡터공간에서 사용되는 연산자로 이해한다.
+ 행렬곱을 통해 벡터를 다른 차원의 공간으로 보낼 수 있음
+ 행렬곱을 통해 패턴을 추출할 수 있고 데이터를 압축할 수도 있음
+ 모든 선형변환(linear transform)은 행렬곱으로 계산할 수 있음

#### 역행렬 이해하기
+ 어떤 행렬 A의 연산을 거꾸로 되돌리는 행렬을 역행렬(inverse matrix)라 부르고 A^-1라 표기
+ 역행렬은 행과 열 숫자가 같고 행렬식이 0이 아닌 경우에만 계산할 수 있다.
+ 항등행렬 : 임의의 벡터 또는 행렬을 곱했을 때 자기자신이 나오는 행렬
+ A^-1 * A = I

``` python
X = np.array([[1, -2, 3],[7,5,0],[-2,-1,2]])
np.linalg.inv(X) # 역행렬
X @ np.lianlg.inv(x) # 항등행렬이 나오게됨
```

#### 유사역행렬
+ 행과 열 숫자가 다르고너 행렬식이 0일 경우 역행렬을 계산할 수 없다.
+ 이때는 유사역행렬(perudo-inverse) 또는 무어-펜로즈(Moore-Penrose) 역행렬 A+을 이용한다.
    + 행과 열의 숫자가 달라도 계산이 가능
+ 역행렬과 완전히 똑같지 않지만 유사한 기능을 하는 행렬
+ 행의 갯수가 더 많은 경우, 더 적은 경우에따라 계산 방식이 달라짐
+ 행이 열보다 더 많을 경우, 원래행보다 먼저 곱해줘야함
+ 응용1 : 연릭방적식 풀기
+ 응용2 : 선형회귀분석
+ 무어-펜로즈 공식

![캡처](https://user-images.githubusercontent.com/44515744/105692470-cc4c5680-5f41-11eb-8361-9a9b6b7835fa.JPG)

``` python
Y = np.array([[0, 1],[1, -1],[-2, 1]])
np.linalg.pinv(Y) # 유사 역행렬을 구할 수 있음
np.linalg.pinv(Y) @ Y # 항등행렬을 구함
```
