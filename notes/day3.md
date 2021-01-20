## [DAY 3] 파이썬 기초 문법 II
### Python Data Structure
> 자료구조는 어떤 데이터를 저장할 때, 그 데이터에 특징에 따라 컴퓨터에 효율적으로 정리하기 위한 데이터의 저장 및 표현 방식을 이야기합니다. 어떤 데이터는 순서가 있다거나, 그 데이터의 나타내는 ID 값과 쌍을 이룬다던가 하는 등의 특징이 나타나게 됩니다. 일반적으로 사용되는 정수, 문자열등의 변수 타입보다는 이러한 특징들에 잘 맞는 형태로 데이터를 저장하게 된다면 훨씬 효율적으로 컴퓨터의 메모리를 사용하고, 프로그래머가 코드를 작성하기에도 용이하게 해줍니다.
#### 1. 스택 (stack)
+ 나중에 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조
+ Last In First Out (LIFO)
+ Data의 입력을 Push, 출력을 Pop이라고 함
+ Push를 append(), Pop는 pop()을 사용
``` python
# 기본 사용법
a = [1, 2, 3, 4, 5]
a.append(10)
a.append(20)
a.pop()    # 20 출력+마지막 데이터 제거
```
#### 2. 큐 (Queue)
+ 먼저 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조
+ First In First Out (FIFO)
+ Stack과 반대되는 개념
+ 파이썬은 리스트를 사용하여 큐 구조를 활용
+ put은 append(), get을 pop(0)을 활용
    
#### 3. 튜플 (tuple)
+ 값의 변경이 불가능한 리스트
+ 선언시 "[]"가 아닌 "()"를 사용
+ 리스트의 연산, 인덱싱, 슬라이싱 등을 동일하게 사용
+ 프로그램을 작동하는 동안 변경되지 않은 데이터의 저장
+ 함수의 반환 값등 사용자의 실수에 의한 에러를 사전에 방지
``` python
# 기본 사용법
t = ()
t = (1, )    # 값이 하나인 Tuple은 반드시 "," 를 붙여야 함
```
#### 4. 집합 (set)
+ 값을 순서없이 저장, 중복을 불허하는 자료형
+ set 객체 선언을 이용하여 객체 생성
+ 수학에서 활용하는 다양한 집합연산 가능
``` python
# 기본 사용법
s = set([1,2,3,1,2,3])    # set 함수를 사용, 집합 객체 생성
s.add(1)    # 한 원소 1만 추가
s.remove(1)    # 1삭제
s.update([1,4,5])    # [1,4,5] 추가
s.discard(3)    # 3 삭제
s.clear()    #모든 원소 삭제
  
s1 = set([1,2,3,4,5])
s2 = set([3,4,5,6,7])
s1.union(s2)    # s1과 s2의 합집합
s1 | s2    # s1과 s2의 합집합
s1.intersection(s2)    # s1과 s2의 교집합
s1 & s2    # s1과 s2의 교집합
s1.difference(s2)    # s1과 s2의 차집합
s1 - s2    # set([1, 2])
```
#### 5. 사전(dictionary)
+ 데이터를 저장 할 때는 구분 지을 수 있는 값을 함께 저장
+ 구분을 위한 데이터 고유 값을 Identifier 또는 Key라고함
+ Key 값을 활용하여, 데이터 값(Value)를 관리함
+ key와 value를 매칭하여 key로 value를 검색
+ 다른 언어에서는 Hash Table 이라는 용어를 사용
+ VS Code CSV 관련 추천 확장프로그램 : RainbowCSV
``` python
# 기본 사용법
country_code = {}
country_code = {'America': 1, 'Korea': 82, 'China': 86, 'Japan': 81}
  
for dict_items in country_code.items():    # Dict Data 출력
    print(type(dict_items))
  
for k,v in country_code.items():    # Dict Data 출력(언패킹 방식)
    print("Key : ", k)
    print("Value : ", v)
    
    country_code.keys()    # Dict 키 값만 출력
    country_code.values()    # Dict 값만 출력
    country_code["Korea"]    # 키가 "Korea"인 값 출력
    
# 활용 예시 - Python 명령어 사용횟수 계산 코드
import csv
  
def getKey(item):
    return item[1]
  
command_data = []
with open("command_data.csv", "r", encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        command_data.append(row)
      
command_counter = {}
for data in command_data:
    if data[1] in command_counter.keys():
        command_counter[data[1]] += 1
    else:
        command_counter[data[1]] = 1
   
dictlist = []
for key, value in command_counter.items):
    temp = [key, value]
    dictlist.append(temp)
    
sorted_dict = sorted(dictlist, key=getKey, reverse=True)  # Key 값을 기준으로 정렬
print(sorted_dict[:100])  # 상위 100개 출력
```

#### 6. 파이썬 기본 데이터 구조
+ List, Tuple, Dict에 대한 Python Built-in 확장 자료 구조(모듈)
+ 편의성, 실행 효율 등을 사용자에게 제공함
+ 아래의 모듈이 존재합
``` python
from collections import deque
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
```

#### 7. deque
+ Stack과 Queue를 지원하는 모듈
+ List에 비해 효율적인=빠른 자료 저장 방식을 지원함
+ rotate, reverse 등 Linked List의 특성을 지원함
+ 기존 list 형태의 함수를 모두 지원함
``` python
# 기본 사용법
from collections import deque
  
deque_list = deque()
for i in range(5):
    deque_list.append(i)     #우측에 추가
print(deque_list)
deque_list.appendleft(10)    #좌측에 추가
print(deque_list)

deque_list.rotate(1)    #우측으로 하나씩 이동 [10, 0, 1, 2, 3, 4] -> [4, 10, 0, 1, 2, 3]
deque_list.extend([5,6,7])    #리스트를 우측에 붙인다.
deque_list.extendleft([5,6,7])    #리스트를 좌측에 붙인다.
```
> 참고. %timeit [함수명]은 평균적인 실행 소요시간을 반환

#### 8. OrderedDict
+ Dict와 달리, 데이터를 입력한 순서대로 dict를 반환함
+ 그러나 dict도 python 3.6부터 입력한 순서를 보장하여 출력함
    + 최신 버전은 일반 dict를 사용하면 됨
        
#### 9. defaultdict
+ Dicttype의 값에 기본 값을 지정, 신규값 생성시 사용하는 방법
    + 없는 key를 조회시 에러를 내보내는 것보다, 디폴트 값을 반환하게 함
``` python
# 기본 사용법
from collections import defaultdict
d = defaultdict(lamnda : 0) # 기본 값은 함수로 넣어야한다. 0을 return하는 함수를 넣을 수도 있다.

# 활용 예시 - 하나의 지문에서 단어들이 몇개가 있는지 출력
d = defaultdict(labda : 0)
for word in text.split():
    d[word] += 1    # 초기값 지정을 해서 바로 수학적 연산이 가능함

sorted_dict = OrderDict()
for i, v in sorted(d.itmes(), key=get_key, reverse=True):
    sorted_dict[i] = v
print(sorted_dict)
    
```

#### 10. Counter
+ Sequence type의 data element들의 갯수를 dict 형태로 변환
``` python
# 기본 사용법
from collections import Counter
ball_or_strike_list = ["B", "S", "S", "S", "B"]
c = Counter(ball_or_strike_list)    # 출력 Counter({'B':2, 'S':3})

c = Counter({'red': 4, 'blue': 2})
print(list(c.elements()))    # 출력 ['red','red','red','red','blue','blue']    리스트로 변환

c = Counter(a=4, b=2, c=0, d=-2)  #생성자에 파라미터를 이런 방식으로 추가할 수 있음
d = Counter(a=1, b=2, c=3, d=4)
c.subtract(d)  # 두 카운터 간 빼기
c + d  # 더하기
c & d  # 교집합
c | d  # 유니온(or)

# 활용 예시 - 하나의 지문에서 단어들이 몇개가 있는지 출력
sorted(Counter(text).items(), key=lambda t : t[1], reverse=True)
```

#### 11. namedtuple
+ Tuple 형태로 Data 구조체를 저장하는 방법
+ 저장되는 data의 variable을 사전에 지정해서 저장함
+ 파이썬에서는 class를 많이 
``` python
# 기본 사용법    
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(x=11, y=22)
x, y = p # 언패킹 가능
print(x, y)
print(p.x + p.y) # 데이터의 체계를 하나로 묶을 수 있음

# 활용 예시 - csv 구조화
from collections import namedtuple
import csv
f = open("users.csv", "r")
next(f)
reader = csv.reader(f)
student_list = []
for row in reader:
    student_list.append(row)
    print(row)

coloumns = ["user_id", "integration_id", "etc"] # 컬럼 목록
Student = namedtuple('Student', " ".join(coloumns))
student_namedtupe_list = []
for row in student_List:
    student = Student(*row)
    student_namedtupe_list.append(student)
print(student_namedtupe_list)
print(student_namedtupe_list[0].full_name)
```
----------------------------
### Pythonic code
> pythonic code 는 앞서 우리가 살펴보았던 데이터 구조와 달리 특별히 모듈이나 함수가 존재하는 것은 아닙니다. 단지 앞에서 배운 str 이나 다양한 모듈들을 활용하여 파이썬 특유의 문법을 표현하는 것입니다. 파이썬 문법의 가장 큰 특징은 짧고 이해하기 편하다는 것 입니다. 코드의 수를 줄여서 비록 컴퓨터의 시간은 증가할 수 있지만, 사람의 시간은 아낄 수 있다는 장점이 있습니다.

#### 1. Pythonic code 정의
+ 파이썬 스타일의 코딩 기법
+ 파이썬 특유의 문법을 활용하여 효율적으로 코드를 표현함
+ 고급 코드를 작성 할 수록 더 많이 필요해짐

#### 2. Pythonic Code를 사용하는 이유
+ 남 코드에 대한 이해도 : 많은 개발자들이 python 스타일로 코딩한다.
+ 효율 : 단순 for loop append 보다 list가 조금 더 빠르다
+ 간지 : 쓰면 왠지 코드 잘 짜는 거처럼 보인다.


#### 3. split 함수
+ string type의 값을 "기준값"으로 나눠서 List 형태로 변환
``` python
example = 'python,java,javascript'   # ","를 기준으로 문자열 나누기
example.split(",")    # list로 반환 
a, b, c = example.split(",")    # a, b, c 변수로 unpacking
example = 'teamlab.technology.io'
subdomain, domain, tld = example.split('.')    #문자열 나누기
```

#### 4. join 함수
+ list를 특정 문자를 사용해서 문자열 형태로 반환
``` python
# 기본 활용법
colors = ["red", "blue", "green", "yellow"]
"-".join(colors)    # red-blue-green-yellow 를 반환
```

#### 5. list Comprehension
+ 기존 List 사용하여 간단히 다른 List를 만드는 기법 (포괄적인 List, 포함되는 리스트, 리스트 내포)
+ 일반적으로 for + append 보다 속도가 빠름
``` python
# 기본 활용법
for i in range(10):
    result.append(i)  #일반 스타일
        
result = [i for i in range(10)]
result = [i for i in range(10) if i % 2 == 0]  # 조건 추가가 가능
           
# nested for loop
word_1 = "Hello"
word_2 = "World"
result = [i+j for i in word_1 for j in word_2]  # 중첩 반복
result = [i+j for i in word_1 for j in word_2 if not(i==j)]  # 필터(조건) 추가가 가능
result = [i+j if not(i==j) else i for i in word_1 for j in word_2] # 삼항 연산으로 필터를 쓸 경우에는 중간에 활용
stuff = [[w.upper(), w.lower(), len(w)] for w in words]  # 2차원 리스트로 반환도 가능
result = [[i+j for i in case_1] for j in case_2]  # 뒤에 for문이 먼저 작동함 [['AD', 'BD', 'CD'], ['AE', 'BE', 'CE'] ... ] 형태로 생성가능

my_str = "ABCD"
{v : i for i , v in enumerate(my_str)}  
```

#### 6. Enumerate
+ list의 element를 추출할 때 번호를 붙여서 추출
``` python
# 기본 활용법
for i, v in enumberate("ABC"):
    print("{0} \t {1}".format(i, v))
        
# 참고 - list 중복제거
set_text = list(set(text.split()))
{v : v.lower() for i , v in enumerate(my_str)}  # dict형태로도 반환
```
+ 참고 - print문 end값 설정
    + print(city, end="\t")
    + print(city, end=" ")

#### 7. zip
+ 두 개의 list의 값을 병렬적으로 추출
``` python
# 기본 활용법
alist = ["a1", "a2", "a3"]
blist = ["b1", "b2", "b3"]

[[a,b] for a, b in zip(alist, blist)] # list로 반환
[ c for c in zip(alist, blist)] # 반환 값은 기본적으로 튜플

# 활용 예시 - 평균
math = (100, 90, 80)
kor = (90, 90, 70)
eng = (90, 80, 70)
[sum(value) / 3 for value in zip(math, kor, eng)]

# 활용 예시 - enumerate & zip
for i, values in enumerate(zip(alist,blist))    # 값들을 번호별로 묶을 수 있음
    print(i, values)
    
#### 8. lambda
+ 함수 이름 없이, 함수처럼 쓸 수 있는 익명함수 ( 수학의 람다 대수에서 유래 )
+ Python 3부터는 권장하지 않으나 여전히 많이 쓰임
    + 어려운 문법
    + 테스트의 어려움
    + 문서화 docstring 지원 미비
    + 코드 해석의 어려움
    + 이름이 존재하지 않는 함수의 출현
``` python
# 일반 적인 코드
def f(x, y):
    return x + y  
    
print(f(1, 4))

# 람다 형태의 코드
f = (lambda x, y: x + y)
print(f(1, 4))
(lambda x, y: x + y)(1, 4) # 이런 형태로도 사용 가능

# 활용 예시
up_low = lambda x : x.upper() + x.lower()
up_low = lambda x : '-'.join(x.split())
```

#### 9. map function
+ 두 개 이상의 list에도 적용 가능함, if filter도 사용가능
+ 시퀀스 데이터에 함수를 매핑하는 기능
+ python3는 iteration 을 생성 -> list을 붙여줘야함
+ 실행시점의 값을 생성, 메모리 효율적
+ 사용을 권장하지 않음
``` python
# 기본 활용법
ex = [1,2,3,4,5]
f = lambda x: x ** 2
list(map(f, ex))  # python3 부터는 list를 붙여 줘야함

f = lambda x, y: x + y
list(map(f, ex, ex))  # 함수에 따라 여러개 파라미터 추가

# 비권장하는 이유
def f(x):
    return x + 5
ex = [1,2,3,4,5]
result = list(map(f, ex))

# 보다 아래가 훨씬 간단함
[f(value) for value in ex]
```

#### 10. reduce function
+ map function과 달리 list에 똑같은 함수를 적용해서 통합
+ 대용량의 데이터를 다루는데 활용
+ Lambda, map, reduce는 간단한 코드로 다양한 기능을 제공
+ 코드의 직관성이 떨어져서 lambda나 reduce는 python3에서 권장하지 않음
``` python
# 기본 활용법
from functools import reduce
print(reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]))   # 1+2 , 3+3, 6+4 같이 앞의 결과에 또 함수를 적용
```

#### 11. iterable object
+ sequence형 자료형에서 데이터를 순서대로 추출하는 object
+ 내부적으로 __iter__와 __next__가 사용됨
+ iter()와 next() 함수를 iterable 객체를 iterator object로 사용
``` python
# 기본 활용법
memory_address_cities = iter(cities)
memory_Address_cities  # 값의 주소를 반환
next(memory_Address_cities)   # 값을 반환, 다음 주소 값을 갖고있음
```

#### 12. generator
+ iterable object를 특수한 형태로 사용하는 함수
+ element가 사용되는 시점에 값을 메모리에 반환 : yield를 사용해 한번에 하나의 element만 반환함
+ 메모리 주소를 절약할 수 있음(필요할때만 사용)
    + 대용량 데이터를 불러올 때는 이것을 사용
+ list 타입의 데이터를 반환해주는 함수는 generator로 만들어라
    + 읽기 쉬운 장점, 중간 과정에서 loop가 중단될 수 있을 때
+ 큰 데이터를 처리할 때는 generator expression을 고려하라
    + 데이터가 커도 처리의 어려움이 없음
+ 파일 데이터를 처리할 때도 generator를 쓰자
``` python
# 기본 활용법
def general_list(value):
    result = []
    for i in range(value):
        result.append(i)
    return result

import sys
result = general_list(50)
sys.getsizeof(result)   #객체의 사이즈를 측정

def general_list(value):
    result = []
    for i in range(value):
        yield i  #평소에는 메모리에 없다가 호출할때만 값을 전달, 메모리 주소 절약이 가능
        
list(general_list(50))  # 이 방식으로 generator의 값을 리스트로 사용 가능        
```

#### 12. generator comprehension
+ list comprehension과 유사한 형태로 generator 형태의 list 생성
+ generator expression 이라는 이름으로도 부름
+ [] 대신 ()를 사용하여 표현
``` python
# 기본 활용법
gen_ex = (n*n for n in range(500))
print(type(gen_ex))

list(gen_ex) # 이때 메모리에 올라온다.
```
#### 13. Function Passing Arguments
+ Keyword arguments
    + 함수에 입력되는 parameter의 변수명을 사용, arguments를 넘김
    + 파라미터 변수명을 지정하기 때문에 순서를 신경 안써도 된다.
``` python
# 기본 활용법
def print_somthing(my_name, your_name, third_name):
    print('Hello {0}, My name is {1}".format(your_name, my_name))
    
print_something(third_name="abc", my_name="Sungchul", your_name="TEAMLAB")
```

+ Default arguments
    + parameter의 기본 값을 사용, 입력하지 않을 경우 기본값 출력
``` python
# 기본 활용법
def print_somthing(my_name, your_name="TEAMLAB"):
    print('Hello {0}, My name is {1}".format(your_name, my_name))
    
print_something("Sungchul", "TEAMLAB")
print_something("Sungchul")
```

+ variable-length asterisk (파라미터 가변인자)
    + 개수가 정해지지 않은 변수를 함수의 parameter로 사용하는 법
    + Keyword arguments와 함께, argument 추가가 가능
    + Asterisk(*) 기호를 사용하여 함수의 parameter를 표시함
    + 입력된 값은 tuple type으로 사용할 수 있음
    + 가변인자는 오직 한 개만 맨 마지막 parameter 위치에 사용가능
    + 파라미터는 *arges로 사용
``` python
# 기본 활용법
def asterisk_test(a, b, *args):
    return a+b+sum(args)
    
asterisk_test(1,2,3,4,5)  # (3,4,5)처럼 튜플로 저장
```

+ keyword variable-length (키워드 가변인자)
    + Parameter 이름을 따로 지정하지 않고 입력하는 방법
    + asterisk(*) 두개를 사용하여 함수의 parameter를 표시함
    + 입력된 값은 dict type으로 사용할 수 있음
    + 가변인자는 오직 한 개만 기존 가변인자 다음에 사용

``` python
# 기본 활용법
def kwargs_test_1(**kwargs):
    print(kwargs)    # {'first':3, 'second':4, 'third':5} 형태로 dict 반환
    print(type(kwargs))    # <class 'ditc'>

kwargs_test_1(first=3, second=4, third=5)

def kwargs_test_3(one, two, *args, **kwargs):  # 순서가 중요하다 
    print(one+two+sum(args))
    print(kwargs)
    
kwrags_test_3(3,4,5,6,7,8,9, first=3, second=4, third=5) # one - 3, two - 4, args = (5,6,7,8,9), kwargs = {'first':3, 'second':4, 'third':5)
# 키워드 형태로 값을 넣기 시작하면 계속 키워드 형태만 넣어야함
```

+ asterisk
    + 흔히 알고 있는 *를 의미함
    + 단순 곱셈, 제곱연산, 가변 인자 활용 등 다양하게 사용됨

+ asterisk - unpacking a container
    + tuple, dict 등 자료형에 들어가 있는 값을 unpacking
    + 함수의 입력값, zip 등에 유용하게 사용가능
``` python
# 기본 활용법
def asterisk_test(a, *args): # 파라미터 가변인자 선언( 여기서 *는 여러개의 가변인자를 받는 용도로 사용 )
    print(a, *args) # 여러개로 들어온 파라미터를 언패킹, 1 2 3 4 5 6
    print(a, args)  # 1 (2, 3, 4, 5, 6)
    print(type(args))  # <class 'tuple'>

test = (2,3,4,5,6)
asterisk_test(1, *test)  # 풀어서 여러개의 가변인자로 전달, *을 안하면 튜플 형태로 파라미터 전달 ( 여기서 *는 언패킹을 하는 용도로 사용 )

# 언패킹 예시
a, b, c = ([1, 2], [3, 4], [5, 6])
print (a, b, c)

data = ([1, 2], [3, 4], [5, 6]) # 위 아래 두 코드는 같다.
print(*data)

# 키워드 언패킹
def asterisk_test(a, b, c, d):
    print(a, b, c, d)
data = {"b":1 , "c":2, "d":3}
asterist_test(10, **data) # b=1, c=2, d=3의 형태로 언패킹한다.

ex = ([1, 2], [3, 4], [5, 6], [5, 6], [5, 6])
for value in zip(*ex):  # (1, 3, 5, 5, 5)   (2, 4, 6, 6, 6) 언패킹으로 같은 값으로 묶음
    print(value)
```
