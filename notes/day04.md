## [DAY 4] 파이썬 기초 문법 III
### Python Object Oriented Programming
> 객체 지향 프로그래밍 언어, Object Oriented Programming(OOP)는 프로그래밍 언어를 배우는 데 있어서 매우 중요한 개념입니다. 파이썬 자체도 OOP 형태로 구성되어 있기도 하지만, 파이썬 나오기 전에 대세 언어들이였던 자바, C++, C# 같은 언어들이 모두 OOP 기반의 언어들입니다. OOP를 배우는 것은 이전에 우리가 if 문이나 loop문을 배우듯이 프로그래밍 언어를 배우는 데 있어 가장 기본적인 개념이 되었습니다.
#### 1. 절자 지향 vs 객체 지향 예시
+ 절차 지향 :  수강신청이 시작부터 끝까지 순서대로 작성
+ 객체 지향 : 수강신청 관련 주체들(교수, 학생, 관리자)의 행동(수강신청, 과목 입력)과 데이터(수강과목, 강의 과목)들을 중심으로 프로그램 작성 후 연결

#### 2. 객체지향 프로그래밍 개요
+ Object-Oriented Programming, OOP
+ 객체 : 실생활에서 일종의 물건 속성(Attribute)와 행동(Action)을 가짐
+ OOP는 이러한 객체 개념을 프로그램으로 표현, 속성은 변수(variable), 행동은 함수(method)로 표현됨
+ 파이썬 역시 객체 지향 프로그램 언어
+ 붕어빵 틀인 클래스(Class)와 실제 구현체(붕어빵)인 인스턴스(instance)로 나눔

#### 3. class 선언하기
+ class 선언, object는 python3에서 자동 상속
+ class SoccerPlayer(object):  # python3에서는 object는 안 적어도 자동으로 상속됨

#### 4. Python naming rule
+ snake_case : 띄어쓰기 부분에 "_"를 추가 뱀 처럼 늘여쓰기, 파이썬 함수/변수명에 사용
+ CamelCase : 띄어쓰기 부분에 대문자 낙타의 등 모양, 파이썬 Class명에 사용

#### 5. ATtribute 추가하기
+ Attribute 추가는 __init__ , self와 함께
    + __init__은 객체 초기화 예약 함수

``` python
# 기본 활용법
class SoccerPlayer(object):
    def __init__(self, name, position, back_number)
        self.name = name
        self.position = position
        self.back_number = back_number

abc = SoccerPlayer("son", "FW", 7)
park = SoccerPlayer("park", "WF", 13)
```

#### 6. 파이썬에서 __ 의미
+ __는 특수한 예약 함수나 변수 그리고 함수명 변경(맹글링)으로 사용
    + 관련 함수로는 __main__ , __add__ , __str__ , __eq__ 등이 있음
``` python
# 기본 활용법
def __str__(self):  # 객체를 출력할 때 아래의 내용이 출력됨
    return "Hello, My name is %s, I play in %s in center " % (self.name, self.position)
    
print(abc) # 원하는 형태로 출력

def __add__(self, other):
    return self.name + other.name

abc + park # + 연산에 대해 정의가 됨

if __name__ == '__main__':
    #파일이 메인일 때만 실행하는 코드 추가
    
{{함수명}}.__doc__   # 함수 선언시 입력한 docstring 출력
    
```

#### 7. method 구현하기
+ method(Action) 추가는 기존 함수와 같으나, 반드시 self를 추가해야만 class 함수로 인정된다.
+ 이때 self는 생성된 instance 자신을 의미한다.
``` python
# 기본 활용법
def change_back_number(self, new_number):
    print("선수의 등번호를 변경합니다 : From %d to %d" % \
        (self.back_number, new_number))

abc.change_back_number(7)
abc.back_number = 7   # 직접 변경도 가능하나 권장하는 방식은 아님
```

#### 8. OOP(Object-oriented programming) Characteristics
+ 객체 지향 언어는 실제 세상을 모델링하는 것
+ Inheritance : 상속
    + 부모클래스로 부터 속성과 Method를 물려받은 자식 클래스를 생성하는 것
``` python
# 기본 활용법
class Person(object):  # 초기 상속은 object를 입력, python3에서는 없어도 됨
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender

    def about_me(self):
        print("저의 이름은 ", self.name, "이구요, 제 나이는 ", str(self.age), "살 입니다.")

class Employee(Person): # 상속할 Class 입력
    def __init__(self, name, age, gender, salary, hire_date):
        super().__init__(name, age, gender) #부모객체 사용
        self.salary = salary
        self.hire_date = hire_date # 속성값 추가

    def do_work(self):
        print("열심히 일합니다.")

    def about_me(self):    # 부모 클래스 함수 재정의
        super().about_me()    # 부모 클래스 함수 사용
        print("제 급여는 ", self.salary, "원 이구요, 제 입사일은 ", self.hire_date, " 입니다.")

myPerson = Person("John": 34, "Male")
myEmployee = Employee("Daeho", 34, "Male", 300000, "2012/03/01")
myEmployee.about_me()
```

+ Polymorphism : 다형성
    + 같은 이름 메소드의 내부 로직을 다르게 작성
    + Dynamic Typing 특성으로 파이썬에서는 같은 부모클래스의 상속에서 주로 발생함
    + 중요한 OOP의 개념이나 깊에 알 필요는 없음
    + 같은 부모를 상속받는 클래스에서 동일한 성격의 method를 만들 경우 같은 함수명으로 다른 기능을 구현하여 편하게 사용할 수 있다.

``` python
# 기본 활용법
class Cat(Animal):
    def talk(self):
        return 'Meow!'

class Dog(Animal):
    def talk(self):
        return 'Woof! Woof!'
```

+ Visibility : 가시성
    + 객체의 정보를 볼 수 있는 레벨을 조절하는 것
    + 캡슐화 또는 정보 은닉 (Information Hiding)
    + Class를 설계할 때, 클래스 간 산섭/정보공유의 최소화
    + 누구나 객체 안의 모든 변수를 볼 필요가 없음
        + 객체를 사용하는 사용자가 임의로 정보 수정
        + 필요 없는 정보에는 접근 할 필요가 없음
        + 만약 제품으로 판한다면? 소스의 보호
``` python
# 기본 활용법
class Product(obejct):
    pass


class Inventory(object):
    def __init__(self):
        self.__items = []  # 앞에 __를 붙여서 Private 변수로 선언, 타 객체가 접근을 못하게 한다.

    def add_new_item(self, product):
        if type(product) == Product:  # 자료형 타입 검사 구문
            self.__items.append(product)
            print("new item added")
        else:
            raise ValueError("Invalid Item")

    def get_number_of_items(self):
        return len(self.__items)

    @property  # Property라는 decorator를 함수에 붙이면 숨겨진 변수를 반환하게 해준다. 함수명을 변수처럼 쓸 수 있게함
    def items(self):
        return self.__items


my_inventory = Inventory()
my_inventory.add_new_item(Product())
my_inventory.add_new_item(Product())

items = my_inventory.items  # Proprty decorator로 추가한 함수를 변수처럼 호출한다.
```

#### 9. First-class objects
+ 일등함수 또는 일급 객체
+ 변수나 데이터 구조에 할당이 가능한 객체
+ 함수를 파라미터로 전달이 가능 + 리턴 값으로 사용
+ 파이썬의 함수는 일급함수

``` python
# 기본 활용법
def square(x):
    return x * x

def cube(x):
    return x * x * x

f = square  # 함수를 변수로 사용
f(5)

def formula(method, argument_list):  # 전달받은 메소드를 함수 내에서 
    return [method(value) for value in argument_list]
```

#### 10. Inner function
+ 함수 내에 또 다른 함수가 존재
``` python
# 기본 활용법
def print_msg(msg):
    def printer():
        print(msg)
    return printer

another = print_msg("Hello, Python")
another() # 함수가 리턴됐기 떄문에 Inner 함수가 할당됨
```
#### 11. decorator function
+ closures : inner function을 return값으로 반환
    + 비슷한 목적의 다양하게 변형된 함수들을 만드는 것
``` python
#closure example
def star(func)
    def inner(*args, **kwargs): # 전달한 매개변수가 그대로 들어옴
        print(args[1] * 30)
        func(*args, **kwargs) # 전달한 매개변수가 그대로 들어옴
        print(args[1] * 30)
    return inner

@star  # 아래 함수가 star 함수에 매개변수로 들어감
def printer(msg, mark):
    print(msg)
printer("Hello", "*")

@star
@percent
def printer(msg):  # 클로져 함수를 여러게 사용 가능 @percent -> @star 순으로 실행
    print(msg)
printer("Hello")

'''
    ****
    %%%%
    Hello
    %%%%
    ****
'''

# decorator function에 인자를 넘기는 예시
def generater_pow(exponent):  # decorator function의 argument가 exponent에 들어감
    def wrapper(f): # 전달된 함수가 f에 추가 (매개변수를 전달할떄는 wrapper function을 넣어야함)
        def inner(*args):  # 실행함수
            result = f(*args)
            return exponent ** result
        return inner
    return wrapper

@generate_power(2)
def raise_two(n):
    return n**2
    
print(raise_two(7))
```
--------------------------
### Module and Project
#### 1. 모듈의 정의
+ 어떤 대상의 부분 혹은 조각
+ 프로그램에서는 작은 프로그램 조각들, 모듈들을 모아서 하나의 큰 프로그램을 개발함
+ 프로그램을 모듈화 시키면 다른 프로그램이 사용하기 쉬움
+ 모듈을 모아놓은 단위, 하나의 프로그램을 패키지라고 함

#### 2. 모듈 만들기
+ 파이썬의 Module == py 파일을 의미한다.
+ 같은 폴더에 Module에 해당하는 .py 파일과 사용하는 .py을 저장한 후
+ import 문을 사용해서 module을 호출한다.
    + import 시에는 인터프리터가 파일을 미리 컴파일을 해놓음 __pycache__
``` python
#기본 활용법
# fah_converter.py
def A():
    pass

import fah_converter  # 파일에 있는 모든 코드를 메모리에 로드
fah_converter.A()  # 모듈이름을 객체로 함수를 호출할 수 있음
```

#### 3. namespace
+ 모듈을 호출할 때 범위를 정하는 방법
+ 모듈 안에는 함수와 클래스 등이 존재 가능
+ 필요한 내용만 골라서 호출 할 수 있음
+ from 과 import 키워드를 사용함
``` python
#기본 활용법
## alias 활용
import fah_converter as fah
fah.A() 

## 모듈에서 특정 함수 또는 클래스만 호출
from fah_converter import A
A()

## 모듈에서 모든 함수 또는 클래스 호출
from fah_converter import *
A()
```

#### 3. built-in module
+ 파이썬이 기본 제공하는 라이브러리
+ 문자처리, 웹, 수학 등 다양한 모듈이 제공
+ 별다른 조치없이 import 문으로 활용 가능
``` python
#난수
import random
print(random.randint(0, 100))
print(random.random())

#시간
print(time.localtime())

#웹
import urllib.request
response = urllib.request.urlopen("http://thetemlab.io")
print(response.read())
```

#### 4. Package 정의
+ 하나의 대형 프로젝트를 만드는 코드의 묶음
+ 다양한 모듈들의 합, 폴더로 연결됨
+ __init__ , __main__ 등 키워드 파일명이 사용됨
+ 다양한 오픈 소스들이 모두 패키지로 관리됨
+ 기능들을 세부적으로 폴더로 나눈다.
+ 패키지 생성 방법
    + 1) 기능을 폴더를 기준으로 세분화
    + 2) 각 폴더별로 필요한 모듈을 구현함
    + 3) 1차 Test - Python Shell
    + 4) 폴더별로 __init__.py 구성하기
        + 현재 폴더가 패키지임을 알리는 초기화 스크립트
        + 없을 경우 패키지로 간주하지 않음 (3.3+ 부터는 X)
        + 하위 폴더와 py 파일(모듈)을 모두 포함한
        + import와 __all__ keyword 사용
    + 5) __main__.py을 생성(파이썬을 폴더채로 실행이 가능)



``` python
# 메인 폴더 __init__.py 예시
__all__ = ["image", "sound", "stage"]

from . import image # 사용할 모듈, 패키지를 명시함
from . import sound
from . import stage


# 서브 폴더
__all__ = ["main", "sub"] # 사용할 모듈, 패키지를 명시함

from . import main
from . import sub


# 메인 폴더 __main__.py 예시
if __name__ == '__main__':  # 다른 모듈도 import해서 main에서 사용 가능
    print("Hello Game")
```

#### 5. Package의 참조
+ Package 내에서 다른 폴더의 모듈을 부를 때 상대 참조로 호출하는 방법
``` python
from game.graphic.render import render_test # 절대참조
from .render import render_test # 상대참조
from ..sound.echo import echo_test # 부모 디텍토리 기준

```

#### 6. 가상환경 설정하기
+ 프로젝트 진행 시 필요한 패키지만 설치하는 환경
+ 기본 인터프리터 + 프로젝트 종류별 패키지 설치
+ 다양한 패키지 관리 도구를 사용함
    + virtualenv + pip : 가장 대표적인 가상환경 관리 도구, 레퍼런스 + 패키지 개수
    + conda : 상용 가상환경도구, 설치의 용이성 (windows에서 장점)

#### 7. 가상환경 관련 명령어
+ 가상환경 생성 : conda create -n my_project python=3.8
+ 가상환경 활성화 : conda activate my_project
+ 가상환경 비활성화 : conda deactivate
+ 패키지 설치 : conda install {패키지명} # 패키지 설치_패키지에 있는 비 python 파일들을 자동으로 컴파일 해줌(window에서 편함)
    + matplotlib
        + 대표적인 파이썬 그래프 관리 패키지
        + 엑셀과 같은 그래프들을 화면에 표시함
        + 다양한 데이터 분석 도구들과 함께 사용됨
    + tqdm
        + for loop을 돌릴때 loop가 얼마나 남았는지 알려줌

        
