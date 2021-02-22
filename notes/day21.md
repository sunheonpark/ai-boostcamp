## [Day 21] 그래프 이론 기초 & 그래프 패턴
### [Graph 1강] 그래프란 무엇이고 왜 중요할까?
#### 1. 그래프란 무엇일까?
+ 그래프(Gpaph)는 정점 집합과 간선 집합으로 이루어진 수학적 구조
+ 하나의 간선은 두 개의 정점을 연결함, 모든 정점 쌍이 반드시 간선으로 직접 연결되는 것은 아님
+ 그래프는 네트워크(Network)로도 불림
+ 정점(Vertex)는 노드(Node)로 간선은 엣지(Edge) 혹은 링크(Link)로도 불림

![캡처](https://user-images.githubusercontent.com/44515744/108650339-adba8a80-7502-11eb-9de2-f38b8d4736d4.PNG)

#### 2. 그래프의 중요성
+ 우리 주변에는 많은 복잡계(Complex System)이 있음
+ 사회는 70억 인구로 이루어진 복잡계, 통신 시스템은 전자 장치로 구성된 복잡계, 신체 역시 복잡계
+ 복잡계를 복잡하게 만드는 건 구성 요소간의 복잡한 상호작용
+ 그래프는 복잡계를 효과적으로 표현하고 분석하기 위한 언어
+ 복잡계는 구성 요소들 간의 상호작용으로 이루어짐
+ 상호작용을 표현하기 위한 수단으로 그래프가 널리 사용됨
+ 복잡계를 이해하고, 복잡계에 대한 정확한 예측을 하기 위해서는 복잡계 이면에 있는 그래프에 대한 이해가 필요
+ 정점 분류(Node Classificiation) 문제


![캡처](https://user-images.githubusercontent.com/44515744/108650696-831d0180-7503-11eb-99f5-5da5315fda4d.PNG)

![캡처](https://user-images.githubusercontent.com/44515744/108650753-a182fd00-7503-11eb-913e-e1a76682e997.PNG)

#### 3. 그래프 관련 인공지능 문제
+ 정점 분류(Node Classification) 문제
    + 트위터에서의 공유(Retweet) 관계를 분석하여, 각 사용자의 정치적 성향을 알 수 있음
    + 단백질의 상호작요을 분석하여 단백질의 역할을 알아낼 수 있음
+ 연결 예측(Link Prediction) 문제
    + 페이스북 소셜 네트워크는 어떻게 진화할까?
+ 추천(Recommendation) 문제
    + 각자에게 필요한 물건은 무엇일까? 어떤 물건을 구매해야 만족도가 높을까?
+ 군집 분석(Community Detection) 문제
    + 연결 관계로부터 사회적 무리(Social Circle)을 찾아낼 수 있을까?
+ 랭킹(Ranking) 및 정보 검색(Information Retrieval) 문제
    + 웹(Web)이라는 거대한 그래프로부터 어떻게 중요한 웹페이지를 찾아낼 수 있을까?
+ 정보 전파(Information Cascading) 및 바이럴 마케팅(Viral Marketing) 문제
    + 정보는 네트워크를 통해 어떻게 전달될까? 어떻게 정보 전달을 최대화할 수 있을까?

#### 4. 그래프의 유형 및 분류
+ 방향이 없는 그래프(Undirected Graph) vs 방향이 있는 그래프(Directed Graph)
+ 간선에 방향이 없는 그래프
    + 협업 관계 그래프
    + 페이스북 친구 그래프

+ 간선에 방향이 있는 그래프
    + 인용 그래프
    + 트위터 팔로우 그래프

![캡처](https://user-images.githubusercontent.com/44515744/108651670-a21c9300-7505-11eb-9edb-9de2e4334440.PNG)

+ 가중치가 없는 그래프(Unweighted Graph) vs 가중치가 있는 그래프(weighted Graph)

![캡처](https://user-images.githubusercontent.com/44515744/108651911-2111cb80-7506-11eb-8ba6-8c677c9090e2.PNG)

+ 동종 그래프(Unpartite Graph) vs 이종 그래프(Bipartite Graph)
+ 동종 그래프는 단일 종류의 정점을 가짐
+ 이종 그래프는 두 종류의 정점을 가짐

![캡처](https://user-images.githubusercontent.com/44515744/108652051-849bf900-7506-11eb-871f-0afaf16e8b22.PNG)

#### 5. 그래프 관련 수학 표현
+ 그래프(Graph)는 정점 집합과 간선 집합으로 이루어진 수학적 구조
+ 보통 정점들의 집합을 V, 간선들의 집합을 E, 그래프를 G = (V,E)로 적음
+ 정점의 이웃(Neighbor)은 그 정점과 연결된 다른 정점을 의미함
+ 정점 v의 이웃들의 집합을 보통 N(v) 혹은 N_{v}로 적음
+ 정점 v에서 간선이 나가는 이웃을(Out-Neighbor)의 집합을 보통 N_{out}(v)로 적음
+ 정점 v로 간선이 들어오는 이웃(In-Neighbor)의 집합을 보통 N_{in}(v)로 적음

#### 6. NetworkX 소개
+ NetworkX를 이용하여, 그래프를 생성, 변경, 시각화할 수 있음
+ 그래프의 구조와 변화를 분석할 수 있음
    + Reference : https://networkx.org/documentation/stable/auto_examples/index.html
    + 참고 snap.py : https://snap.stanford.edu/snappy/
    + NetworkX는 속도가 느리나 사용이 빠름, snap.py는 반대(기능 차이가 많으므로 둘다 아는것을 추천)

``` python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

print("###### Graph Init ######")
G = nx.Graph()  # 방향성이 없는 그래프 초기화
DiGraph = nx.DiGraph()  # 방향성이 있는 그래프 초기화

# 정점을 추가하고, 정점의 수를 세고, 목록을 반환
print("###### Add Node to Graph ######")
print("# Add node 1")
G.add_node(1)
print("Num of nodes in G : " + str(G.number_of_nodes()))
print("Graph : " + str(G.nodes) + "\n")

# 더 많은 정점을 추가
print("# Add vertext 2 ~ 10")
for i in range(1, 11):
    G.add_node(i)
print("Num of nodes in G : " + str(G.number_of_edges()))
print("Graph : " + str(G.nodes) + "\n")

# 간선을 추가하고, 목록을 반환
print("###### Add Egde to Graph ######")
print("#Add edge (1, 2)")
G.add_edge(1, 2)
print("Graph : " + str(G.edges) + "\n")

# 더 많은 간선을 추가
print("#Add edge (1, i) for i = 2 ~ 10")
for i in range(2, 11):
    G.add_edge(1, i)
print("Graph : " + str(G.edges) + "\n")

# 만들어진 그래프를 시각화함
# 정점의 위치 결정
pos = nx.spring_layout(G)
# 정점의 색과 크기를 지정하여 출력
im = nx.draw_networkx_nodes(G, pos, node_color="red", node_size=100)
# 간선 출력
nx.draw_networkx_edges(G, pos)
# 각 정점이 라벨을 출력
nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
plt.show()

# 그래프를 인접 리스트로 저장
nx.to_dict_of_lists(G)

# 그래프를 간선 리스트로 저장
nx.to_edgelist(G)

# 그래프를 인접 행렬(일반 행렬)로 저장
nx.to_numpy_array(G)

# 그래프를 인접 행렬(희소 행렬)로 저장
nx.to_scipy_sparse_matrix(G)

```

#### 7. 그래프의 표현 및 저장
+ 간선 리스트(Edge List): 그래프를 간선들의 리스트로 저장
    + 각 간선은 해당 간선이 연결하는 두 정점들의 순서쌍(Pair)로 저장됨
    + 방향성이 있는 경우에는 (출발점, 도착점) 순서로 저장됨

![캡처](https://user-images.githubusercontent.com/44515744/108654108-5371f780-750b-11eb-9111-a2e1309bc8eb.PNG)

+ 인접 리스트(Adjacent list)
    + 방향성이 없는 경우 : 각 정점의 이웃들을 리스트로 저장
    + 방향성이 있는 경우 : 각 정점의 나가는 이웃들과 들어오는 이웃들을 각각 리스트로 저장

![캡처](https://user-images.githubusercontent.com/44515744/108654440-08a4af80-750c-11eb-8f2c-e9adc8fba7c4.PNG)

+ 인접 행렬(Adjacency Matrix)
    + 정점 수 x 정점 수 크기의 행렬
    + 정점 i와 j 사이에 간선이 있는 경우, 행렬의 i행 j열 (그리고 j행 i열) 원소가 1
    + 정점 i와 j 사이에 간선이 없는 경우, 행렬의 i행 j열 (그리고 j행 i열) 원소가 0
    + 방향성이 없는 경우에는 인접행렬이 대각 행렬이다.
    + 방향성이 있는 경우에는 i에서 j로의 간선이 있는 경우에 1, 아닐 경우에는 0으로 표시한다.
    + 방향성이 있는 경우에는 인접행렬이 대각 행렬이라는 보장이 없다.

![캡처](https://user-images.githubusercontent.com/44515744/108654705-97193100-750c-11eb-8ab5-2200b208edfc.PNG)

+ 행렬별 메모리 요구량 
    + 일반 행렬은 전체 원소를 저장하므로 정점 수의 제곱에 비례하는 저장 공간을 사용
    + 희소 행렬은 0이 아닌 원소만을 저장하므로 간선의 수에 비례하는 저장 공간을 사용
        + 예시) 정점의 수가 10만, 간선의 수가 100만이라면 => 일반행렬은 정점의 수의 제곱 (100억) >> 희소 행렬은 간선의 수 (100만)
    + 희소 행렬은 일반 행렬보다 속도가 느리다는 단점이 존재함

#### 8. 정리
+ 그래프란 무엇이고 왜 중요할까?
    + 그래프는 정점 집합과 간선 집합으로 이루어진 수학적 구조
    + 그래프는 복잡계를 표현하고 분석하기 위한 언어
+ 그래프 관련 인공지능 문제
    + 정점 분류, 연결 예측, 추천, 군집 분석, 랭킹, 정보 검색, 정보 전파, 바이럴 마케팅 등

+ 그래프 관련 필수 기초 개념
    + 방향성이 있는/없는 그래프, 가중치가 있는/없는 그래프, 동종/이종 그래프
    + 나가는/들어가는 이웃

+ (실습) 그래프의 표현 및 저장
    + 파이썬 라이브러리 NetworkX
    + 간선 리스트, 인접 리스트, 인접 행렬

### [Graph 2강] 실제 그래프는 어떻게 생겼을까?
#### 1. 실제 그래프 vs 랜덤 그래프
+ 실제 그래프(Real Graph)란 다양한 복잡계로 부터 얻어진 그래프를 의미
    + 소셜 네트워크, 전자상거래 구매 내역, 인터넷, 웹, 뇌, 단백질 상호작용, 지식 그래프 등
+ 랜덤 그래프(Random Graph)는 확률적 과정을 통해 생성한 그래프를 의미함
    + 에르되스-레니 랜덤 그래프
    + 임의의 두 정점 사이에 간선이 존재하는지 여부는 동일한 확률 분포에 의해 결정됨
    + 에르되스-레니 랜덤그래프 G(n,p)로 표현
        + n개의 정점을 가짐
        + 임의의 두 개의 정점 사이에 간선이 존재할 확률은 
        + 정점 간의 연결은 서로 독립적(Independent)임

+ 랜덤 그래프 예시 (3개의 정점)

![캡처](https://user-images.githubusercontent.com/44515744/108663330-a9e33400-7513-11eb-93d7-9afbbfd96090.PNG)

#### 2. 그래프의 경로와 거리
+ 정점 u와 v의 사이의 경로(Path)는 아래 조건을 만족하는 정점들의 순열(Sequence)
    + u에서 시작해서 v에서 끝나야함
    + 순열에서 연속된 정점은 간선으로 연결되어야 함
    
    ![캡처](https://user-images.githubusercontent.com/44515744/108663547-2d048a00-7514-11eb-9a1a-5a3c31a77906.PNG)

+ 경로의 길이는 해당 경로 상에 높인 간선의 수로 정의됨 (경로 내의 정점의 수 - 1이 길이가 됨)
+ 정점 u와 v의 사이의 거리(Distance)는 u와 v 사이의 최단 경로의 길이를 의미함
+ 그래프의 지름(Diameter)은 정점 간 거리의 최댓값을 의미함
    + 모든 정점들의 거리를 계산하고 이 거리 중 가장 큰 값이 지름이 됨

#### 3. 작은 세상 효과
+ 그래프의 구조적 특성 중 하나
+ 여섯 단계 분리(Six Degrees of Separation) 실험
    + 사회학자 스탠리 밀그램에 의해 1960대에 수행된 실험
    + 오마하와 위치타에서 500명의 사람을 뽑음
    + 그들에게 보스턴에 있는 한 사람에게 편지를 전달하게함 ( 지인을 통해서만 전달 )
    + 25%의 편지만 도착했지만, 평균적으로 6단계만 거침
    + MSN 메신저 그래프에서는 정점 간의 평균 거리는 7정도 밖에 되지 않음(거대 연결 구조만 고려)
+ 이러한 현상을 작은 세상 효과(Small-world Effect)라고 부름(사돈의 팔촌 - 10촌 관계)
+ 작은 세상 효과는 높은 확률로 랜덤 그래프에도 존재함
+ 모든 사람이 100명의 지인이 있다고 가정하면 다섯 단계를 거치면 최대 100억(100^5)명의 사람과 연결될 수 있음
    + 단 실제로는 지인의 중복 때문에 100억 명보다는 적은 사람(많은 사람과 연결될 수 있음)
+ 아래의 그래프 유형에서는 작은 세상 효과가 존재하지 않음

![캡처](https://user-images.githubusercontent.com/44515744/108664404-09dada00-7516-11eb-9dd2-2f878b33a049.PNG)

#### 4. 연결성의 두터운 꼬리 분포
+ 정점의 연결성(Degree)은 그 정점과 연결된 간선의 수를 의미
+ 정점 v의 연결성은 해당 정점의 이웃들의 수와 같다.
+ 정점 v의 연결성은 d(v), d_{v} 혹은 |N(v)|와 같이 표현
+ 방향성이 있는 그래프(Directed Graph)인 경우에는 in, out을 활용해서 연결성 표현함
    + v의 나가는 연결성(Out Degree)은 d_{out}(v) 혹은 |N_{out}(v)|
    + v로 들어오는 연결성(in Degree)은 d_{in}(v) 혹은 |N_{in}(v)|

![캡처](https://user-images.githubusercontent.com/44515744/108664746-c2088280-7516-11eb-8e39-017f4de013b8.PNG)

+ 실제 그래프의 연결성 분포는 두터운 꼬리(Heavy Tail)를 갖음
    + 연결성이 매우 높은 허브(Hub) 정점이 존재함을 의미함
    + BTS의 연결성은 엄청 크지만 내 연결성은 엄청 작다(현실)
+ 랜덤 그래프의 연결성 분포는 높은 확률로 정규 분포와 유사함
    + 연결성이 매우 높은 허브(Hub) 정점이 존재할 가능성은 0에 가까움
    + 키가 10 미터가 넘는 극단적인 예외는 존재하지 않음

![캡처](https://user-images.githubusercontent.com/44515744/108665668-cafa5380-7518-11eb-980b-b60e0b2f16ea.PNG)

#### 5. 거대 연결 요소
+ 연결 요소(Connected Component)는 다음 조건들을 만족하는 정점들의 집합을 의미
    + 연결 요소에 속하는 정점들은 경로로 연결될 수 있음
    + 위 조건을 만족하면서 정점을 추가할 수 없음
    + 예시 그래프에는 3개의 연결 요소가 존재 {1,2,3,4,5}, {6,7,8}, {9}

![캡처](https://user-images.githubusercontent.com/44515744/108666089-aeaae680-7519-11eb-9368-27738debba35.PNG)

+ 실제 그래프에는 거대 연결 요소(Giant Connected Component)가 존재함
    + 거대 연결 요소는 대다수의 정점을 포함함
    + MSN 메신저 그래프에는 99.9%의 정점이 하나의 거대 연결 요소에 포함됨

+ 랜덤 그래프에도 높은 확률로 거대 연결 요소(Giant Connected Component)가 존재함
    + 단, 정점들의 평균 연결성이 1보다 충분히 커야함

![캡처](https://user-images.githubusercontent.com/44515744/108666639-f41be380-751a-11eb-9db3-82b2abab3adb.PNG)

#### 6. 군집 구조
+ 군집(Community)이란 다음 조건들을 만족하는 정점들의 집합
    + 집합에 속하는 정점 사이에는 많은 간선이 존재
    + 집합에 속하는 정점과 그렇지 않은 정점 사이에는 적은 수의 간선이 존재

![캡처](https://user-images.githubusercontent.com/44515744/108667341-56291880-751c-11eb-8b58-5650a9d236e1.PNG)

#### 7. 지역적 군집 계수
+ 지역적 군집 계수(Local Clustering Coefficient)는 한 정점에서 군집의 형성 정도를 측정함
+ 정점 i의 지역적 군집 계수는 점정 i의 이웃 쌍 중 간선으로 직접 연결된 것의 비율을 의미
    + 정점 i의 지역전 군집 계수를 C_{i}로 표현함
    + 전체쌍의 수 (2,3) (2,4) (2,3) (3,4) (4,5) (3,5) => n은 이웃의 수 (n * n-1) / 2 - 조합공식

![캡처](https://user-images.githubusercontent.com/44515744/108668058-bff5f200-751d-11eb-9f93-98ff48181e3d.PNG)

+ 이웃상 사이의 간선이 추가될 경우 지역적 군집 계수가 증가함
+ 참고로 연결성이 0인 정점에서는 지역적 군집 계수가 정의되지 않음

![캡처](https://user-images.githubusercontent.com/44515744/108668506-ad2fed00-751e-11eb-9b7c-04a5d53b0393.PNG)


+ 지역접 군집 계수가 군집이랑 연결되는 원리
    + 정점 i의 지역적 군집 계수가 높다면, i의 이웃들도 높은 확률로 서로 간선으로 직접 연결됨
    + 정점 i의 이웃들도 높은 확률로 서로 간선으로 직접 연결되어 있음
    + 정점 i와 그 이웃들은 높은 확률로 군집을 형성함(서로 많이 연결되어 있으면, 다른 이웃 기준에서도 연결성이 높다고 볼 수 있음)

#### 8. 전역 군집 계수
+ 전역 군집 계수(Global Clustering Coefficient)는 전체 그래프에서 군집의 형성 정도를 측정함
    + 그래프 G의 전역 군집 계수는 각 정점에서의 지역적 군집 계수의 평균(지역적 군집 계수가 정의되지 않는 정점은 제외 - 연결성이 0인 정점)

#### 9. 높은 군집 계수
+ 실제 그래프에서는 군집 계수가 높음. 즉 많은 군집이 존재함
    + 동실성(Homophily) : 서로 유사한 정점끼리 간선으로 연결될 가능성이 높음. 같은 동네에 사는 같은 나이의 아이들이 친구가 되는 경우가 그 예시
    + 전이성(Transitivity) : 공통 이웃이 있는 경우, 공통 이웃이 매개역할을 해줄 수 있음, 친구를 서로에게 소개하는 경우

    ![캡처](https://user-images.githubusercontent.com/44515744/108669265-02b8c980-7520-11eb-80ee-c51affdb15c1.PNG)

#### 10. 랜덤 그래프에서는 지역적 혹은 전역 군집 계수가 높지않음
+ 랜덤 그래프 G(n,p)에서의 군집 계수는 p
+ 랜덤 그래프에서의 간선 연결이 독립적인 것을 고려하면 당연한 경과
+ 공통 이웃의 존재 여부가 간선 연결 확률에 영향을 미치지 않음

#### 11. 실습 : 군집 계수 및 지름 분석
+ 데이터 불러오기
    + 균일 그래프(Regular Graph) : 정점은 균일하게 4개의 다른 정점과 연결됨
    + 작은 세상 그래프(Small-world Graph) : 균일 그래프의 일부 간선을 일부 선택한 간선으로 대체한 것
    + 랜덤 그래프(Random Graph0)
    
    ![캡처](https://user-images.githubusercontent.com/44515744/108669463-70fd8c00-7520-11eb-920c-c1e09c5e0804.PNG)

``` python
regular_graph = nx.Graph()
data = osp.abspath(osp.join(os.getcwd(), 'drive/MyDrive/data/simple/regular.txt'))
f = open(data)
for line in f:
    v1, v2 = map(int, line.split())
    regular_graph.add_edge(v1, v2)
```

+ 군집 계수 계산
``` python
# 주어진 그래프의 전역 군집 계수를 계산하는 함수를 정의
def getGraphAverageClusteringCoefficient(Graph):
    cㅊs = []
    for v in Graph.nodes:
        num_connected_pairs = 0
        for neighbor1 in Graph.neighbors(v):
            for neighbor2 in Graph.neighbors(v):
                if neighbor1 <= neighbor2:
                    continue
                if Graph.has_edge(neighbor1, neighbor2):
                    num_connected_pairs = num_connected_pairs + 1
            cc = num_connected_pairs / (Graph.degree(v) * (Graph.degree(v) -1) / 2)
            cㅊs.append(cc)
        return sum(ccs) / len(ccs)
```

+ 지름 계산
``` python
# 주어진 그래프의 지름(을 계산하는 함수를 정의
def getGraphDiameter(Graph):
    diameter = 0
    for v in Graph.nodes:
        length = nx.single_source_shortest_path_length(Graph, v) # 그래프에서 정점의 최단 거리를 구하는 것(출발점과 다른 정점들 사이의 거리 벡터를 반환)
        max_length = max(length.values())
        if max_length > diameter:
            diameter = max_length
    return diameter
```

+ 비교 분석
    + 실제 그래프는 작은 세상 그래프와 비슷한 특징을 갖고 있음(군집 계수는 크고, 지름은 작다)
    + 수정) 랜덤 그래프의 지름은 작음(잘못된 이미지)

![캡처](https://user-images.githubusercontent.com/44515744/108671745-431a4680-7524-11eb-8f31-d05c209ce4f3.PNG)

#### 12. 정리
+ 실제 그래프 vs 랜덤 그래프 : 실제 그래프는 복잡계로부터 얻어지는 반면, 랜덤 그래프는 확률적 과정을 통해 생성
+ 작은 세상 효과 : 실제 그래프의 정점들은 가깝게 연결됨
+ 연결성의 두터운 꼬리 분포 : 실제 그래프에는 연결성이 매우 높은 허브 정점이 존재함
+ 거대 연결 요소 : 실제 그래프에는 대부분의 정점을 포함하는 거대 연결 요소가 존재
+ 군집 구조 : 실제 그래프에는 군집이 존재하며, 실제 그래프는 군집 계수가 높음