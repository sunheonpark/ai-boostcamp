## [DAY 5] 파이썬으로 데이터 다루기
### File / Exception / Log Handling
> 예상치 못한 오류를 해결하기 위해 특정 상황에서 발생할 수 있는 예외를 포괄적으로 지정해서 대비하는 예외처리와 파일을 다루는 방법 그리고 프로그램을 진행하면서 기록을 남기는 로깅에 대한 내용
#### 1. Exception 정의
+ 예상 가능한 예외
    + 발생 여부를 사전에 인지할 수 있는 예외
    + 사용자의 잘못된 입력, 파일 호출 시 파일 없음
    + 개발자가 반드시 명시적으로 정의 해야함
        + 조건문을 활용해서 예외처리를 한다.
+ 예상이 불가능한 예외
    + 인터프리터 과정에서 발생하는 예외, 개발자 실수
    + 리스트의 범위를 넘어가는 값 호출 <= 정수 0으로 나눔
    + 수행 불가시 인터프리터가 자동 호출
        + Exception Handling을 통해 처리한다.

#### 2. Exception Handling
+ try ~ except 문법
+ except가 발생하면 다시 try로 올라가서 코드를 수행함
+ except가 없다면 프로그램이 종료됨
+ Built-in Exception : 기본적으로 제공하는 예외
+ Exception Handling이 많을 경우 다른 사용자가 Exception 원인을 찾기 힘들어짐
+ 파일이 비어있을 경우, 


Exception 이름 | 내용
---- | ----
IndexError | List의 Index 범위를 넘어갈 때
NameError | 존재하지 않은 변수를 호출 할 때
ZeroDivisionError | 0으로 숫자를 나눌 떄
ValueError | 변환할 수 없는 문자/숫자를 변환할 때
FileNotFoundError | 존재하지 않는 파일을 호출할때

``` python
# 기본 활용법
for i in range(10):
    try:
        print(i, 10 // i )
    except ZeroDivisionError:
        print("Error")
        print("Not divided by 0")
    except IndexError as e: # 에러에 대한 상세정보 파악가능
        print(e)
    except Exception as e:
        print(e)
    else: # Exception이 발생하지 않을 경우 else 구문이 실행
        print("No Exceiption")
    finally:
        print("End") # 예외랑 상관없이 무조건 실행
```

+ raise 구문 : 필요에 따라 강제로 Exception을 발생
    + 장기간 처리되는 데이터가 중간에 의도한 것과 달리 다르게 처리되는 경우 raise 처리
``` python
# 기본 활용법
    while True:
        value = input("변환할 정수 값을 입력해주세요")
        for digit in value:
            if digit not in "0123456789":
                raise ValueError("숫자값을 입력하지 않으셨습니다")
        print("정수값으로 변환된 숫자 -", int(value))
```

+ assert 구문
``` python
# 기본 활용법
def get_binary_number(decimal_number):
    assert isinstance(decimal_number, int) # Fasle 일때 exception 에러를 발생시켜줌(요즘엔 주로 이런 용도로 Type Hint를 사용)
    return bin(decimal_number)

print(get_binary_number(10))
```

#### 3. File 유형
+ 컴퓨터는 text 파일을 처리하기 위해 binary 파일로 변환시킴
+ 모든 text파일도 실제는 binary 파일, ASCII/Unicode 문자열 집합으로 저장되어 사람이 읽을 수 있음
+ 파일의 종류
    + Text 파일 : 인간도 이해할 수 있는 형태인 문자열 형태로 저장된 파일
        + 메모장으로 열면 내용 확인 가능
    + Binary 파일 : 컴퓨터만 이해할 수 있는 형태인 이진(법)형식으로 저장된 파일
        + 메모장으로 열면 데이터가 깨져 보인다

#### 4. Python File I/O
+ 파이썬은 파일 처리를 위해 "open"키워드를 사용함
``` python
f = open("<파일이름>", "접근 모드") # 파일을 읽을 수 있는 주소를 지정
f.close()
```

파일열기모드 | 설명 
---- | ----
r | 읽기모드
w | 쓰기모드
a | 추가모드

``` python
# 기본 활용법
f = open("i_have_a_dream.txt", "r")
contents = f.read() # 모든 내용을 가져온다.
contents = f.readlines() # \n을 기준으로 다 잘라서 list로 가져온다.
contenst = f.readline() # 데이터가 많아 한번에 메모리에 올릴 수 없다면 readline을 활용
print(contents)
f.close()

#with 구문과 함께 사용하기 - close() 안해도 됨
with open("i_have_a_dream.txt", "r") as f:
    contents = f.read()
    print(contents)

#encoding 지정
# 한글과 동아시아 utf-8, widnow에서는 cp949라는 포맷을 사용, utf-8로 데이터 포맷을 맞춰서 저장해야함
with open("i_have_a_dream.txt", "w", encoding="utf8") as f 

# a는 파일 맨끝에 데이터를 추가한다.
with open("counter_log.tex", "a", encoding="utf8") as f:
    for i in range(11, 21):
        data = "%d번째 줄입니다.\n" % i
        f.write(data)
```

#### 5. Pyhton Directory
+ os 모듈을 사용하여 Directory 다루기
``` python
# 폴더 생성
import os
os.mkdir("log")

# 폴더 생성 예외처리
try:
    os.mkdir("abc")
except FileExistsError as e:
    print("Already created")

# 폴더 존재여부 체크
os.path.exists("abc") # 참일 경우 True
os.path.isfile("file.ipynb")

# shutil - 폴더 처리 관련 모듈
import shutil
source = "i_have_a_dream.txt"
dest = os.path.join("abc", "sungchucl.txt") # 폴더를 상대 경로로 지정해주기 위해 join을 활용 => ('abc\\sungchul.txt')
shutil.copy(source, dest)

# pathlib - 폴더 처리 관련 모듈2 : 객체처럼 다룰 수 있어서 훨씬 수월함
import pathlib
pathlib.path.cwd() # 현재 파일 경로
cwd.parent # 상위 폴더로 이동 
cwd.parent
cwd.parent.parents # 노드간의 이동이 편리함

# log 파일 생성 예시
import os
if not os.path.exists("log"):
    os.mkdir("log")

TARGET_FILE_PATH = os.path.join("log", "count_log.txt")
if not os.path.exists(TARGET_FILE_PATH):
    f = open("log/count_log.txt", "w", encoding="utf8")
    f.write("기록이 시작됩니다\n")
    f.close()

with open(TARGET_FILE_PATH, 'a', encoding="utf8") as f:
    import random, datetime
    for i in range(1, 11):
        stamp = str(datetime.datetime.now())
        value = random.random() * 100000  # 0.0에서부터 1.0 사이의 실수(float)를 반환
        log_line = stamp + "\t" + str(value) + "값이 생서되었습니다\n"
        f.write(log_line)
```

#### 6. Pickle
+ 파이썬 객체를 영속화(persistence)하는 built-in 객체
+ 데이터, object 등 실행중 정보(계산 결과_모델)를 저장 
+ 바이너리 파일로 저장, 읽어야함
``` python
import pickle

# 피클 저장
f = open("list.pickle", "wb")
test = [1, 2, 3, 4, 5]
pickle.dump(test, f) # 변수를 f에 저장
f.close()

# 객체를 삭제
del test

# 피클 로드
f = open("list.pickle", "rb")
test_pickle = pickle.load(f)
test_pickle
f.close()

# 객체도 저장이 가능하다.
import pickle

class Multiply(object):
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def multiply(self, number):
        return number * self.multiplier

multiply = Multiply(5)
f = open("multiply_object.pickle", "wb")
pickle.dump(multiply, f)
f.close()

# 객체 호출
f = open("multiply_object.pickle", "rb")
multiply_pickle = pickle.load(f)
```

#### 7. Logging Handling
+ 프로그램이 실행되는 동안 일어나는 정보를 기록에 남기기
+ 유저의 접근, 프로그램의 Exceiption, 특정 함수의 사용
+ Console 화면에 출력, 파일에 남기기, DB에 남기기
+ 실행시점에서 남겨야 하는 기록(유저), 개발시점에서 남겨야할 기록(에러)

#### 8. print vs logging
+ Console 창에만 남기는 기록은 분석시 사용불가
+ 모듈별로 별도의 logging을 남길 필요가 있음

#### 9. logging 모듈
+ python의 기본 Log 관리 모듈
+ 프로그램 진행 사황에 따라 다른 Level의 Log를 출력함
+ Log 관리시 가장 기본이 되는 설정 정보
+ DEBUG > INFO > WARNING > ERROR > CRITICAL (Warning 부터 사용자 레벨)
+ 기본 Logging Level은 WARNING 레벨로 설정되어 있음(Operation level이 기준)
``` python
# 기본 활용법
import logging
logging.debug("틀렸잖아!")
logging.info("확인해")
logging.warning("조심해")
logging.error("에러났어")
logging.critical("망했다") # 완전히 종료됐을 때
```

+ logging level
Level | 개요
---- | ----
debug | 개발시 처리 기록을 남겨야하는 로그 정보를 남김
info | 처리가 진행되는 동안의 정보를 알림
warning | 사용자가 잘못 입력한 정보나 처리는 가능하나 원래 개발시 의도치 않는 정보가 들어왔을 때 알림
error | 잘못된 처리로 인해 에러가 났으나, 프로그램을 동작할 수 있음을 알림
critical | 잘못된 처리로 데이터 손실이나 더이상 프로그램이 시작할 수 없음을 알림

``` python
# 레벨 정하는 예시 코드
import logging

logger = logging.getLogger("main")  # Logger 선언

logger.basicConfig(level=logging.DEBUG) # 로그 레벨을 DEBUG로 설정
logger.basicConfig(level=logging.CRITICAL) # 로그 레벨을 CRITICAL로 설정

stream_handler = logging.FileHandler("my.log", mode="w", encoding="utf8") # Logger의 output 방법 선언
logger.addHandler(stream_handler) # Logger의 output 등록

# Log 결과 값의 포맷 설정
formatter = logging.Formatter('%(asctime)s %(levelname)s %(process)d %(message)s') 

#Config 파일 설정
logging.config.fileConfig('example.conf')
logger = logging.getLogger()
```

#### 10. log 파일 설정을 정하는 방법 = configparser
+ 프로그램의 실행 설정을 file에 저장함
+ Section, Key, Value 값의 형태로 설정된 설정 파일을 사용
+ 설정 파일을 Dict Type으로 호출 후 사용
``` python
# 기본 활용법
""" config File 예시 - 파일명 : example.conf
[SectionOne]  # 대괄호로 Section 구분
Status: Single  # 속성 - Key : Value
Name : Derek
Value : Yes

[SectionTwo]
FavoriteColor: Green
"""

import Configparser

#설정 파일 호출
config = configparser.ConfigParser()
config.read('example.cfg')
print(config.sections())

#설정 파일 값 조회
print(config['SectionOne'])
for key in config['SectionOne']:
    value = config['SectionOne'][key]
```

#### 12. log 파일 설정을 정하는 방법 = argparser
+ Console 창에서 프로그램 실행시 Setting 정보를 저장함
+ 일반적으로 활용하는 방식
+ Command-Line Option 이라고 부름
+ main 함수 안에다가 argument 정보를 선언
``` python
import argparse
parser = argparse.ArgumentParse(description="Sum tow integers.")

# 짧은 이름, 긴 이름, 표시명, Help 설명, Argument Type
parser.add_argument('-a', "--a_value", dest= "A_value", help="A integers", type= int)
parser.add_argument('-b', "--b_value", dest= "B_value", help="B integers", type= int)

args = parser.parse_args()
print(arges)
print(arges.a)

"""
python arg_sum.py -a 19 -b 10 형태로 Console에서 입력
"""
```
-----------------------------
### Python data handling
> CSV, 웹, XML, JSON 네 가지 데이터 타입과 정규표현식에 대한 내용
#### 1. CSV
+ CSV(Comma Separate Value)은 필드를 쉼표(,)로 구분한 텍스트 파일
+ 엑셀 양식의 데이터를 프로그램에 상관없이 쓰기 위한 데이터 형식
+ 엑셀에서는 "다른 이름 저장" 기능으로 사용 가능
+ 판다스로 주로 csv를 핸들링함
``` python
# 기본 활용법
# 파일처럼 처리하는 방식
line_count = 0
data_header = []
customer_list = []

with open("customers.csv") as customer_data:
    while True:
        data = customer_data.readline()
        if not data: break
        if line_counter == 0:
            data_header = data.split(",")
        else:
            customer_list.append(data.split(","))
        line_counter += 1

# 파일처럼 처리할 경우에는 문장 내의 ","에 대해 전처리를 해야함
# 간단히 처리하기 위한 csv 모듈을 사용하는 방식
import csv
header = []
rownum = 0
result = []

# csv 읽기
with open("customers.csv", "r", encoding="cp949") as f:
     reader = csv.reader(f)
     for row in reader:
        if rownum == 0:
            header = row
        location = row[7]
        if location.find(u"성남시") != -1: #u는 유니코드의 약자
            reuslt.append(row)
        rownum += 1
# csv 쓰기
with open("result_customver.csv", "w", encoding="utf8") as sf:
    writer = csv.writer(sf, delimiter='\t', quotechar="'", quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerow(header)
    for row in seoung_nam_data:
        writer.writerow(row)

```

+ quotechar은 데이터 내에 포함된 쉼표에 대한 처리를 위해 삽입
+ Widnow는 lineterminator가 \n이나 Mac은 \r\n이다.

Attribute | Default | Meaning
---- | ---- | ---- 
delimiter | , | 글자를 나누는 기준
lineterminator | \r\n | 줄 바꿈 기준
quotechar | " | 문자열을 둘러싸는 신호 문자
quoting | QUOTE_MINIMAL | 데이터 나누는 기준을 quoterchar에 의해 둘러싸인 레벨로 설정


#### 2. Web
+ World Wide Web(WWW), 줄여서 웹이라고 부름
+ 우리가 쓰는 인터넷 공간의 정식 명칭
+ 데이터 송수신을 위한 HTTML 프로토콜을 사용
+ 데이터를 표시하기 위해 HTML 형식을 사용
    + 1. 요청 : 웹주소, Form, Header 등
    + 2. 처리 : Database 처리 등 요청 대응
    + 3. 응답 : HTML, XML 등으로 결과 반환
    + 4. 렌더링 : HTML, XML 표시

#### 3. HTML
+ 웹 상의 정보를 구조적으로 표현하기 위한 언어
+ 제목, 단락, 링크 등 요소 표시를 위해 Tag를 사용
+ 모든 요소들은 꺾쇠 괄호 안에 둘러 쌓여있음
+ 모든 HTML은 트리 모양의 포함 관계를 가짐
+ 일반적으로 웹 페이지의 HTML 소스파일은 컴퓨터가 다운로드 받은 후 웹 브라우저가 해석/표시
+ 정보의 보고, 많은 데이터들이 웹을 통해 공유됨
+ HTML도 일종의 프로그램, 페이지 생성 규칙이 있으므로 이를 분석하여 데이터 추출이 가능
+ 추출한 데이터를 바탕으로 하여 다양한 분석이 가능

#### 4. 정규식
+ 정규 표현식, regexp 또는 regex로 불림
+ 복잡한 문자열 패턴을 정의하는 문자 표현 공식
+ 특정한 규칙을 가진 문자열의 집합을 추출

기호 | 설명 
---- | ---- 
[ ] | [ 와 ] 사이의 문자들과 매치
\- | 문자의 범위 지정
\. | 줄바꿈 문자인 \n를 제외한 모든 문자와 매치
\* | 앞에 있는 글자를 반복해서 나올 수 있음
\+ | 앞에 있는 글자를 최소 1회 이상 반복
{m,n} | 반복 횟수를 지정
? | 앞 문자가 있을 수도 없을 수도 있다.
\| | or 조건
^ | not 조건
( ) | 패턴을 하나로 묶는다 - 캡쳐로 활용

#### 5. 정규식 in 파이썬
+ re 모듈을 import 하여 사용
+ 함수 search : 한개만 찾기
+ 함수 findall - 전체 찾기
+ 추출된 패턴은 tuple로 반환됨
+ 정규식을 활용하면 사이트의 특정 코드를 추출할 수 있음
``` python
# 기본 활용법
import re
import urllib.request

url = "https://www.naver.com"
html = urllib.request.urlopen(url)
html_contents = str(html.read())
id_results = re.findall(r"(A-Za-z0-9]+\*\*\*)", html_contents)
for result in id_results
    print(result)
```

#### 6. XML이란
+ 데이터의 구조와 의미를 설명하는 TAG(MarkUp)를 사용하여 표시하는 언어
+ TAG와 TAG사이에 값이 표시되고, 구조적인 정보를 표현
+ JSON 이전엔 XML을 사용해서 데이터를 주고 받음
+ HTML과 동일하게 정규식을 사용해서 Parsing이 가능함

``` xml
<books>
    <book>
    </book>
</books>
```

#### 7. BeautifulSoup
+ HTML, XML 등 Markup 언어 SCraping을 위한 대표적인 도구
+ laxml과 html5lib과 같은 Parser를 사용함
+ 속도는 느리나 간편하게 사용이 가능 (lxml이 가장 빠름)
+ conda install lxml 설치
+ conda install beautifulsoup4 설치

``` python
# 기본 활용법
# 모듈 호출
from bs4 import BeautifulSoup

# 객체 생성
soup = BeautifulSoup(books_xml, "lxml")

# 태그 정보 탐색
soup.find_all("book") # 모든 태그 탐색
soup.find("book") # 하나의 태그만 탐색

for book_info in soup.find_all("book")
    print(book_info.get_text())  # 태그의 텍스트를 가져옴

A_tag = soup.find("BBB")
B_tag = A.tag.find("CCC") # 태그 내 탐색도 가능함
```

#### 8. JSON
+ JSON(Javscript Object Notation)은 웹 언어인 Java Script의 데이터 객체 표현 방식
+ 간결성으로 기계/인간이 모두 이해하기 편함
+ 데이터 용량이 적고, Code로의 전환이 쉬움(XML 대체제)
+ Python의 Dict Type과 유사 key:value 쌍으로 데이터 표시
+ 데이터 저장 및 읽기는 dict type과 상호 호환이 가능
``` python
# 기본 활용법
import json

# json 데이터 불러오기
with open("json_example.json", "r", encoding="utf8") as f:
    contents = f.read()
    json_data = json.loads(contents) # dict 타입으로 불어옴
    print(json_data["employees"])

# json 파일 쓰기
dict_data = {'name':'sunheon', 'age':'28'}
with open("data.json", "w") as f:
    json.dump(dict_data, f)
```
----------------------
## 추가 학습
### 알고리즘
#### 1. 복잡도
> 복잡도는 알고리즘의 성능을 나타내는 척도
+ 시간 복잡도 : 알고리즘을 위해 필요한 연산의 횟수
    + 시간 복잡도를 표현할 때는 빅오(Big-O) 표기법을 사용
    + 시간 복잡도를 O(N) 형태로 표기 N은 데이터의 양이다.
    + 빅오 표기법은 차수가 가장 큰 항만 남기지만 상수 값이 미치는 영향이 더 클 수 있으므로 고려해야한다.
    + 2중 반복문의 경우에는 O(N^2)이다.
    + 연산 횟수가 1000000000억을 넘어가면 C 언어를 기준으로 통상 1초 이상의 시간이 소요된다. 보통 5초 이내에 풀어야한다.
    + 시간 복잡도 판

빅오 표기법 | 명칭
---- | ----
O(1) | 상수 시간
O(logN) | 로그 시간
O(N) | 선형 시간
O(NlogN) | 로그 선형 시간
O(N^2) | 이차 시간
O(N^3) | 삼차 시간
O(2^n) | 지수 시간

+ 공간 복잡도 : 알고리즘을 위해 필요한 메모리의 양
    + 일반적인 메모리 사용량 기준은 MB단위
    + 메모리 사용량은 128 ~ 512MB 정도로 제한

#### 2. 알고리즘 유형
+ 그리디 알고리즘 : 현재 상황에서 당장 좋은 것만 고르는 방법
+ 완전 탐색 : 모든 경우의 수를 주저 없이 다 계산하는 해결 방법
+ 시뮬레이션 : 문제에서 제시한 알고리즘을 한 단계씩 차례대로 직접 수행하는 방식
