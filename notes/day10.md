## [DAY 10] 시각화 / 통계학
### [AI Math 7강] 시각화 도구
#### 1. matplotlib란?
+ pyplot 객체를 사용하여 데이터를 표시
+ pyplot 객체에 그래프들을 쌓은 다음 flush
+ 최대 단점 argument를 kwargs로 받음
    + 파라미터 파악이 어려우므로 공식 홈페이지에서 보면서 사용
    + https://matplotlib.org/gallery/index.html
+ Graph는 원래 figure 객체에 생성
+ pyplot 객체 사용시, 기본 figure에 그래프가 그려짐
+ Matplotlib는 Firgure 안에 Axes로 구성
+ figure 위에 여러개의 Axes를 생성

``` python
# 기본 활용법
import matplotlib.pyplot as plt

X = range(100)
X = [value**2 for value in X]
plt.plot(X, Y)
plt.show()

# Firgure에 X,Y 쌍을 계속 추가하는 개념으로 이해
import numpy as np

X_1 = range(100)
Y_1 = [np.cos(value) for value in X]

X_2 = range(100)
Y_2 = [np.sin(value) for value in X]
plt.plot(X_1, Y_1)
plt.plot(X_2, Y_2)

# pyplot을 잘라서 설정할 수 있음, 한번에 여러 그래프 반환
import matplotlib.pyplot as plt

X_1 = range(100)
Y_1 = [np.cos(value) for value in X]

X_2 = range(100)
Y_2 = [np.sin(value) for value in X]

fig = plt.figure()
fig.set_size_inches(10, 2)
ax_1 = fig.add_subplot(1,2,1)
ax_2 = fig.add_subplot(1,2,2)

ax_1.plot(X_1, Y_1, c="g")
ax_2.plot(X_2, Y_2, c="b")
plt.show()

# 반복문 활용
fig = plt.figure()
fig.set_size_inches(5,5)
# plt.style.use("ggplot") # 스타일도 변경 가능, 다양한 스타일이 있음

ax = []
colors = ["b", "g", "r", "c", "m","y","k"]
for i in range(1,7):
    ax.append(fig.add_subplot(2,3,i))
    X_1 = np.arange(50)
    Y_1 = np.random.rand(50)
    c = colors[np.random.randint(1, len(colors))]
    
    ax[i - 1].plot(X_1, Y_1, c=c)
plt.savefig("test.png", c="a") # 
plt.show() # show를 하는 순간 파일을 flush를 해서 메모리에서 제거됨

# RGP 코드로 지정가능
plt.plot(X_1, Y_1, color="#eeefff") 

# 제목 설정기능 (제목에 latex 타입도 표현이 가능)
plt.title('$y = \\frac{ax + b}test}$') 
plt.xlabel("$x_line$") # X 라벨 값 지정
plt.ylabel("y_label") # Y 라벨 값 지정
plt.legend(shadow=True, fancybox=True, loc="lower right") # 범례 추가
plt.text(50, 70, "Line_1") # 텍스트 추가
plt.annotate( # 화살표도 그릴 수 있음
    "line_2",
    xy=(50, 150),
    xytest=(20, 175),
    arrowprops=dict(facecolor="black", shrink=0.05),
)
```
#### 2. matplotlib graph
+ 산점도
``` python
# 기본 활용법
data_1 = np.random.rand(512, 2) # 512행 2열 랜덤값
data_2 = np.random.rand(512, 2)

plt.scatter(data_1[:,0], data_1[:,1], c="b", marker="x") # data_1[:,0] 모든 행의 1번째열(0) 데이터 가져오기
plt.scatter(data_2[:,0], data_1[:,1], c="r", marker="^")

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
```
+ Bar 차트
``` python
# 기본 활용법
# 간격을 정해주면서 생성하는 것이 관건
data = [[5., 25.,50., 20.],
        [4., 23., 51., 17.],
        [6., 22., 52., 19.]]

X = np.arange(4)
plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)
plt.xticks(X + 0.25, ("A", "B", "C", "D"))
plt.show()

data = np.array([[5., 25.,50., 20.],
        [4., 23., 51., 17.],
        [6., 22., 52., 19.]])
color_list = ["b", "g", "r"]
data_label = ["A", "B", "C"]

# 누적 bar 차트
for i in range(3):
    plt.bar(
        X,
        data[i], # 데이터양
        bottom=np.sum(data[:i], axis=0), # i이전까지의 누적합계를 bottom으로 설정
        color=color_list[i],
        label=data_label[i],
    )
plt.legend()
plt.show()

# horizontal bar 차트
women_pop = np.array([5, 30, 45, 22])
men_pop = np.array([5, 25, 50, 20])
X = np.arange(4)

plt.barh(X, women_pop, color="r")
plt.barh(X, -men_pop, color="b")
plt.show()
```
#### 3. seaborn
+ matplotlib를 쉽게 쓸 수 있게 지원해주는 wrapper
+ 기존 matplotlib에 기본 설정을 추가
+ 복잡한 그래프를 간단하게 만들 수 있는 wrapper
+ pandas object를 바로 넣어서 사용이 가능
+ https://seaborn.pydata.org/
+ seaborn의 구성
    + basic plot : 기본적인 플롯
    + multiplot : sub plot을 여러개 추가할 수 있는 것
+ seaborn의 설치
    + !conda install --y seaborn
+ seaborn 종류
    + Viloinplot - boxplot에 distribution을 함께 표현
    + Stripplot - scatter와 category 정보를 함께 표현
    + Swarmplot - 분포와 함께 scatter를 함께 표현
    + Pointplot - category 별로  numeric의 평균, 신뢰구간 표시
    + regplot - scatter + 선형함수를 함께 표시

``` python
# 기본 활용법
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid") # 테마지정
tips = sns.load_dataset("tips") # 레스토랑의 팁 정보 데이터, 라이브러리에서 제공
fmri = sns.load_dataset("fmri") # 

# line plot
sns.set_style("whitegrid")
sns.lineplot(x="timepoint", y="signal", data=fmri) # x, y 평균값과 분포를 그려줌
sns.lineplot(x="timepoint", y="signal", hue="event", data=fmri) # event 필드의 데이터 카테고리 별로 나눠서 보여줌

# scatter plot
sns.scatterplot(x="total_bill", y="tip", data=tips)
sns.regplot(x="total_bill", y="tip", data=tips) #scatter에 선형회귀 라인을 넣어줌

# counter plot
sns.countplot(x="smoker", hue="time", data=tips) # 개수를 세는 막대 그래프

# bar plot
sns.barplot(x="day", y="tip", data=tips) # 상단에 분포를 나타내줌
sns.barplot(x="day", y="tip", data=tips, estimator=len) # 기본 값은 평균임, 계산 기준을 바꾸려면 estimator를 바꿔줌
# len - count
# np.std = 표준편차
# np.sum = 합계

# violinplot - 분포, 표준편차, 평균을 같이 볼 수 있음
sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, palette="muted")

# swamplot - 산전도 같이 갯수를 기준으로 데이터를 볼 수 있음( 데이터가 적을때 사용)
sns.swarmplot(x="day", y="total_bill", hue="smoker", data=tips, palette="muted")

# catplot - 
sns.catplot(x="day", y="total_bill", hue="smoker", kind="swarm", data=tips)

# FacetGrid - 여러 그래프를 한꺼번에 그려ㅑ줌
g = sns.FacetGrid(tips, col="time", row="sex") # 카테고리 별로 그래프가 생성됨
g.map(sns.scatterplot, "total_bill", "tip")

# 다양한 차트를 설정가능
g.map(sns.histplot, x = "total_bill")
```

### [AI Math 7강] 통계학 맛보기
> 모수의 개념과 모수를 추정하는 방법으로 최대가능도 추정법
#### 1. 모수
+ 통계적 모델링은 적절한 가정 위에서 확률분포를 추정(inference)하는 것이 목표
+ 유한한 개수의 데이터만 관찰해서 모집단의 분포를 정확하게 알아낸다는 것은 불가능하므로, 근사적으로 확률분포를 추정해야함
+ 모델을 예측할 때 분포를 정확하게 맞춘다는 것보다는 데이터와 추정방법의 불확실성을 고려한 상태에서 예측의 위험을 최소화하는 방법으로 학습
+ 데이터가 특정 확률분포를 따른다고 가정한 후, 그 분포를 결정하는 모수(parameter)를 추정하는 방법을 모수적(parametric) 방법론이라고 한다.
+ 특정 확률분포를 가정하지 않고 데이터에 따라 모델의 구조 및 모수의 개수가 바뀌면 비모수 방법론이라고 부름
    + 기계학습의 많은 방법론은 비모수(nonparametric) 방법론에 속함
    + 모수가 무한히 많거나, 모수가 데이터에 따라 바뀌는 경우에 비모수 방법론이라고 함
    + 어떤 가정을 부여했는지 아닌지에 따라 구분됨
+ 확률분포를 가정하는 방법 : 히스토그램을 통해 모양을 관찰
    + 데이터가 2개의 값 -> 베르누이 분포
    + 데이터가 n개의 이산적인 값 -> 카테고리분포
    + 데이터가 [0, 1] 사이에서 값을 가지는 경우 -> 베타분포
    + 데이터가 0이상의 값을 가지는 경우 -> 감마분포, 로그정규분포 등
    + 데이터가 R 전체에서 값을 가지는 경우 -> 정규분포, 라플라스분포
+ 기계적으로 확률분포를 가정해서는 안 되며, 데이터를 생성하는 원리를 먼저 고려하는 것이 원칙

#### 2. 데이터로 모수를 추정하는 방법
+ 데이터의 확률분포를 가정했다면 모수를 추정할 수 있음
+ 정규분포의 모스는 평균과 분산으로 이를 추정하는 통계량은 아래와 같음
+ 이는 
+ 표본평균

![CodeCogsEqn (5)](https://user-images.githubusercontent.com/44515744/106232967-b9cf6700-6238-11eb-8327-c8ec75cb5598.gif) 

![CodeCogsEqn (6)](https://user-images.githubusercontent.com/44515744/106233112-17fc4a00-6239-11eb-891c-0dac5eb54914.gif)

+ 표본 분산 
    + 표본분산을 구할 때 N이 아니라 N - 1로 나누는 이유는 불편(unbiased) 추정량을 구하기 위함이다.
    + 표본분산을 구하면 기대 값이 원래 모집단의 분산 시그마 제곱과 일치하게 됨 ( 기대값을 취했을 때 원래 모집단의 통계치와 일치하기 위해 사용 )
    + 컴퓨팅 1/n 써도 상관없는데 estimator를 배우는데 n-1로 해야지 estimator가 나온다고 설명됨
    + 통계량의 확률분포를 표집분포(sampling distribution)라 부르며, 특히 표본평균의 표집분포는 N이 커질수록 정규분포 
    + 표본 평균과 표본 분산의 확률분포를 표집분포(sampling distribution)라 부르며, N(데이터)이 커질수록 정규분포를 따른다.
        + 이를 중심극한정리(Central Limit Theorem)이라 부른다.

![CodeCogsEqn (7)](https://user-images.githubusercontent.com/44515744/106233277-76292d00-6239-11eb-9e4d-b474a07ed344.gif)

![CodeCogsEqn (8)](https://user-images.githubusercontent.com/44515744/106233333-9b1da000-6239-11eb-9369-d80e66c3d231.gif)

#### 3. 최대가능도 추정법
+ 표본평균이나 표본분산은 중요한 통계랑이지만 확률분포마다 사용하는 모수가 다르므로 적절한 통계량이 달라짐
+ 이론적으로 가장 가능성이 높은 모수를 추정하는 방법 중 하나를 최대가능도 추정법(maximum likelihood estimation, MLE)라고 함
+ 가능도(likelihood) 함수는 데이터가 주어져 있는 상황에서 쎄타를 변형시킴에 따라 값이 바뀌는 함수
+ 모수 쎄타를 따르는 분포가 x를 관찰할 가능성을 뜻하지만 확률로 해석하면 안됨

#### 4. 로그가능도를 사용하는 이유?
+ 로그가능도를 최적화하는 모수 세타는 가능도를 최적화하는 MLE가 됨
+ 데이터의 숫자가 적으면 상관없지만 만일 데이터의 숫자가 수억 단위가 되면 컴퓨터의 정확도로는 가능도를 계산하는 것이 불가능
+ 데이터가 독립일 경우, 로그를 사용하면 가능도의 곱셈을 로그가능도의 덧셈으로 바꿔서 컴퓨터로 연산이 가능
    + 곱셈에서 발생하는 데이터 오차를 최소화 할 수 있음
+ 경사하강법으로 가능도를 최적화할 때 미분연산을 사용하게 됨, 로그 가능도를 사용하면 연산량일 O(n^2)에서 O(n)으로 줄여줌
+ 대게의 손실함수의 경우 경사하강법을 사용하므로 음의 로그가능도(negative log-likelihood)를 최적화하게 됨

#### 4. 최대가능도 추정법 예제: 정규분포
+ 정규분포를 따르는 확률변수 X로부터 독립적인 표본 {x1, ..., xn}을 얻었을 때 최대 가능도 추정법을 이용하여 모수를 추정
    + 정규분포이기 때문에 평균과 분산 두개의 파라미터를 가질 수 있음
    + 가능도를 로그 가능도로 변경해서 계산
    + 정규분포의 확률 미도함수에 로그를 씌움, 

#### 5. 카테고리 분포에서 최대가능도 추정법
+ 카테고리 분포 Multinoulli를 따르는 확률변수 X로부터 독립적인 표본을 얻었을 때 최대가능도 추정법을 이용하여 모수를 추정
+ 카테고리 분포의 모수는 다음 제약식을 만족해야함

![CodeCogsEqn (9)](https://user-images.githubusercontent.com/44515744/106241920-2f8ffe80-624a-11eb-829d-c3dd4c27fe43.gif)

#### 6. 딥러닝에서 최대가능도 추정법
+ 최대가능도 추정법을 이용해서 기계학습 모델을 학습할 수 있음
+ 딥러닝 모델의 가중치를 W라 표기했을 때 분류 문제에서 소프트맥스 벡터는 카테고리분포의 모수를 모델링
+ 원핫벡터로 표현한 정답레이블 y을 관찰데이터로 이용해 확률분포인 소프트맥스 벡터의 로그가능도를 최적화할 수 있음
+ 기계학습에서 사용되는 손실함수들은 모델이 학습하는 확률분포와 데이터에서 관찰되는 확률분포의 거리를 통해 유도함
+ 데이터공간에 두 개의 확률분포 P(x), Q(x)가 있을 경우 두 확률분포 사이의 거리(distance)를 계산할 때 다음과 같은 함수들을 이용함
    + 총변동 거리 (Total Variation Distance, TV)
    + 쿨백-라이블러 발산 (Kullback-Leibler Divergence, KL)
    + 바슈타인 거리 (Wasserstein Distance)
+ 분류 문제에서 정답레이블을 P, 모델 예측을 Q라 두면 최대가능도 추정법은 쿨백-라이블러 발산을 최소화하는 것과 동일

![CodeCogsEqn (10)](https://user-images.githubusercontent.com/44515744/106243404-b2b25400-624c-11eb-90a5-236da2407005.gif)

### 추가학습
