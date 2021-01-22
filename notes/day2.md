## [DAY 2] 파이썬 기초 문법
### Variable & List
#### 1. 변수의 개요
+ 데이터(값)을 저장하기 위한 메모리 공간의 프로그래밍상 이름
+ 프로그래밍에서 변수는 값을 저장하는 장소
+ 선언 되는 순간 메모리 특정영역에 물리적인 공간이 할당됨
+ camper 변수에 sunheon park이라는 값을 넣어라
``` python
  camper = "sunheon park"'
```
+ 변수는 메모리 주소를 가지고 있고 변수에 들어가는 값은 메모리 주소에 할당됨
+ [참고] 컴퓨터의 구조 - 폰 노이만(아키텍처) : 사용자가 컴퓨터에 값을 입력하거나 프로그램을 실행할 경우, 그 정보를 먼저 메모리에 저장시키고 CPU가 순차적으로 그 정보를 해석하고 계산한다.
#### 2. 변수명의 작성법
+ 알파벳, 숫자, 언더스코어(_)로 선언 가능
+ 변수명은 의미 있는 단어로 표기하는 것이 좋다.
+ 변수명은 대소문자가 구분된다.
+ 특별한 의미가 있는 예약어는 쓰지 않는다. ex) for, if, else

#### 3. 기본 자료형(Primitive data type)
+ data type : 파이썬이 처리할 수 있는 데이터 유형

유형 | 설명 | 예시 | 선언형태
---- | ---- | ---- | ----
integer | 양/음의 정수 | 1, 2, 3, 100, -9 | data = 1
float | 소수점이 포함된 실수 | 10.2, -9.3, 9.0 | data = 9.0
string | 따옴표 (' / ")에 들어가 있는 문자형 | abc,a20abc | data = 'abc'
boolean | 참 또는 거짓 | True, False | data = True
+ 데이터 타입마다 차지하는 메모리 공간의 크기가 다름 ( integer - 32bit, float - 64bit )
+ 코드 실행시점에 데이터의 Type이 결정됨 - Dynamic Typing => 인터프리터는 속도가 느린 단점을 갖고 있음

#### 4. 연산자(Operator)와 피연산자(operand)
+ +, -, * 같은 기호들을 연산자라고 칭함
+ 연산의 순서는 수학에서 연산 순서와 같음
+ 문자간에도 + 연산도 가능함 => concatenate
+ "**"는 제곱승에 대한 연산
+ "%"는 나머지에 대한 연산
+ "/"는 정확히 나눠지지 않는 경우 소수점 15자리까지 표현
+ "//"는 나누고 정수부분만 반환한다.
+ 증가 또는 감소 연산
  + a += 1는 a = a + 1과 같은 의미로 증가연산
  + a -= 1는 a = a - 1과 같은 의미로 감소연산
  + 변수는 좌변에 있을 때는 저장공간, 우변에 있을 때는 값으로 볼 수 있음
  + 파이썬에서 "a++", "a--" 구문은 사용이 불가능

#### 5. 데이터 형 변환: 정수형, 실수형
+ type() : 변수 타입을 구하는 함수 
``` pyton
  a = 12
  type(a)
```
+ float() : 실수형으로 데이터를 변환
``` python
  float("31.2")
```
+ int() : 정수형으로 데이터를 변환 ( 실수형을 정수로 형변환할 경우에는 소수점 이하 내림 )
``` python
  int("31.6")  # Output : 31
```
+ 컴퓨터의 반올림 오차
  + 단순한 실수도 이진수로 변환하면 무한소수가 됨
  + 반올림 오차는 충분히 작아 반올림 하여 일반적으로 문제가 되지 안흥ㅁ
+ [참고] 컴퓨터는 실리콘이라는 재료로 만든 반도체 구성
  + 반도체는 특정 자극을 줬을 때 전기를 통할 수 있게 하는 물질 (전류의 흐름의 제어가 가능) 
  + 전류가 흐를 때 1, 흐르지 않을 때는 0으로만 숫자를 표현할 수 있음

#### 6. list
+ 시퀀스 자료형, 여러 데이터들의 집합
+ int, float 같은 다양한 데이터 타입 포함 (리스트는 메모리의 주소를 참조하기 때문)
+ 인덱싱
  + list에 있는 값들은 주소(offset)를 가짐 ( 0부터 offset을 사용 )
  + 주소를 사용해 할당된 값을 호출
  + len 함수를 사용해서 list의 길이를 계산
+ 슬라이싱
  + list의 값들을 잘라서 쓰는 것이 슬라이싱
  + list의 주소 값을 기반으로 부분 값을 반환
``` python
  cities[5:10] # index 5부터 9까지를 반환
  cities[:] # 처음부터 끝까지
  cities[-9: # : 마지막 index : -1 , 그 앞 index : -2  - index를 끝에서부터 접근할 수 있음
  cities[::2] # 2칸 단위로 슬라이싱
  cities[::-1] # 역으로 슬라이싱
```
+ 리스트의 연산
  + 문자열A + 문자열B - 두 리스트가 합쳐진다.
  + 문자열A * 2 - 동일한 내용의 리스트가 합쳐진다. (concatenate)
  + "blue" in color - 문자열 포함 확인
  + color.append("white") - 리스트에 값을 추가한다.
  + color.extend(["black", "purple"]) - 리스트에 리스트를 추가한다.
  + color.remove("white") - 리스트에 값을 제거한다.
  + del color[0] - 리스트에 값을 제거하는데 메모리를 날리는 차이점이 존재
+ 리스트 메모리 저장 장식
``` python
  a = [5, 4, 3, 2, 1]
  b = [1, 2, 3, 4, 5]
  a = b # a, b가 같은 곳을 가리킴
  a.sort() # 리스트의 순서과 바뀜(sort 리스트 값이 바뀌는 함수)
  sorted(a) # 기존 리스트를 변경하지 않고 정렬된 값을 반환
  b # b의 결과 값도 순서가 바뀐 것을 알 수 있다.
```
+ 패킹과 언패킹
  + 패킹 : 한 변수에 여러 개의 데이터를 넣는 것
  + 언패킹 : 한 변수의 데이터를 각각의 변수로 반환
``` python
  t = [1, 2, 3,]
  a, b, c = t #리스트의 값이 각각 저장된다
  a #1
  b #2
  c #3
```
+ 이차원 리스트
  + 리스트 안에 리스트를 만들어 행렬 생성
  + midterm_copy = copy.deepcopy(midterm_score) # 같은 저장공간을 가리키는게 아니라 값을 복사하여 별도의 저장공간에 저장
----------------------
### Function and Console I/O
#### 1. 함수의 개요
+ 어떤 일을 수행하는 코드의 덩어리
``` python
  # 사각형의 넓이를 구하는 함수
  def calculate_rectangle_area(x, y):
    return x * y
```
+ 반복적인 수행을 1회만 작성 후 호출
+ 코드를 논리적인 단위로 분리
+ 캡슐화 : 인터페이스만 알면 타인의 코드를 사용

#### 2. 함수 선언 문법
+ 함수 이름, Parameter, indentation, return value(optional)으로 구성됨

#### 3. 함수 수행 순서
+ 함수 부분을 제외한 메인 프로그램부터 시작
+ 함수 호출시 함수 부분을 수행 후 되돌아 옴

#### 4. Parameter VS argument의 차이
+ parameter : 함수의 입력 값 인터페이스
+ argument : 실제 Parameter에 대입된 값

#### 5. 함수 형태
+ parameter 유무, 반환 값(return value) 유무에 따라 함수의 형태가 다름

#### 6. Console I/O 개요
+ input() : 콘솔창에서 입력 값을 전달받는 함수 (String 타입)
``` python
  typing_word = input() # 콘솔창의 입력 값을 받을 수 있음
```
+ print() : 콘솔창에 값을 출력하는 함수
``` python
  print("Hello World!!", "Hello Again!!") #콤바를 구분으로 다른 데이터를 동시 출력이 가능(중간에 공백이 추가)
```
#### 7. Print Formatting
프린트 문은 기본적인 출력 외에 출력의 양식을 지정 가능
``` python
  print(1,2,3)
  print("a" + " " + "b" + " " + "c")
  
  #% String 방식(%5d, %8.2f 등의 출력 포맷팅을 할 수 있음)
  print("%d %d %d" % (1,2,3))    
  
  #format 함수 사용
  print("{0} {1} {2}".format("a","b","c"))
  print("Product: {0:>10s}, Price per unit: {1:10.5f}.".format("Apple", 5.243))    #format 함수 사용 + Padding ( > 오른쪽으로 정렬을 의미함 )
  print("Product: {name:>10s}, Price per unit: {price:10.5f}.".format(name="Apple", price=5.243))    #format 함수 사용 + Padding + Naming
  
  #fstring 방식
  name = "SunHeon"
  age = 30
  print(f"value is {value}")
  print(f'{name:20}')    #20칸, 좌측 정렬
  print(f'{name:*>20}')    #20칸, 나머지 별표, 우측 정렬
  print(f'{name:*%20}')    #20칸, 가운데 정렬, 나머지 별포
  
  number = 3.141592653589793
  print(f'{number:10.2f}')
```
#### 8. 참고자료
+ PEP-8 NAMING CONVENTIONS : https://www.youtube.com/watch?v=Sm0wwmEwqpI

----------------------
### Conditionals and Loops
#### 1. 조건문 기본
+ 프로그램 작성 시, 조건에 따른 판단과 반복은 필수
+ 조건에 따라 특정한 동작을 하게하는 명령어
+ 조건문은 조건을 나타내는 기준과 실행해야 할 명령으로 구성됨
+ 파이썬은 조건문으로 if, else, elif 등의 예약어를 사용함
``` python
  if <조건>:  #if를 쓰고 조건 삽입 후 " : "입력
    <수행 명령1-1> #들여쓰기 후 수행할 명령 입력
    <수행 명령1-2> #같은 조건하에 실행일 경우 들여쓰기 유지
  else: #조건이 불일치할 경우 수행할 명령 block
    <수행 명령2-1> #조건 불일치 시 수행할 명령 입력
    <수행 명령2-2> #조건 불일치 시 수행할 명령 들여쓰기 유지
```
#### 2. 조건 판단 방법
+ if 다음에 조건을 표기하여 참 또는 거짓을 판단함
+ 참/거짓의 구분을 위해서 비교 연산자를 활용
+ is는 메모리 주소를 기반으로 조건 검사를 함
  + -5~256는 정적 메모리를 사용함, 나머지 범위의 값은 동적 메모리를 할당하기 때문에 is를 쓰면 값이 달라진다.
+ 컴퓨터는 값이 존재하면 참, 없으면 거짓으로 판단함
  
비교연산자 | 비교상태 | 설명
---- | ---- | ----
x < y | ~보다 작음 | x가 y보다 작은지 검사
x > y | ~보다 큼 | x가 y보다 큰지 검사
x == y or x is y | 같음 | x와 y가 같은지 검사(값과 메모리 주소)
x != y or x is not y | 같지 않음 | x와 y가 다른지 검사(값과 메모리 주소)
x >= y | 크거나 같음 | x가 y보다 이상인지 검사
x <= y | 작거나 같음 | x가 y보다 이하인지 검사

#### 3. 논리 키워드 사용: and, or, not
+ 조건문을 표현할 때 집합의 논리 키워드를 함께 사용하여 참과 거짓을 판단
``` python
  if not (4 > 3):    #not으로 인해 거짓 반환
  
  boolean_list = [True, False, True]
  all(boolean_list)    #and 조건(아이템)
  all([20 <= age, age <= 26])    #좌측 형태로도 조건 설정이 가능
  any(boolean_list)    #or 조건(아이템)
```

#### 4. 삼항 연산자
``` python
  value = 12
  is_even = True if value % 2 == 0 else False
```
#### 5. 반복문이란?
+ 정해진 동작을 반복적으로 수행하게 하는 명령문
+ 반복문은 반복 시작 조건, 종료 조건, 수행 명령으로 구성됨
+ 반복문 역시 반복 구문은 들여쓰기와 block으로 구분됨
+ 파이썬은 반복문으로 for, while 등의 명령 키워드를 사용함
``` python
  for looper in [1,2,3,4,5]:
    print(f"{looper} : hello")
    
  for looper in range(0,5)    #range(0,5) = range(5), 0부터 4까지의 리스트를 만들어 줌
```
#### 6. 반복문 상식
+ 임시적인 반복 변수는 대부분 i, j, k로 정함
+ 반복문은 대부분 0부터 반복을 시작(2진수가 0부터 시작하기 때문에 - 관례)
+ 무한 loop
  + 반복 명령이 끝나지 않는 프로그램 오류
  + CPU와 메모리 등 컴퓨터의 리소스를 과다하게 점유

#### 7. for문의 다양한 반복문 조건 표현
``` python
#문자열을 한자씩 리스트로 처리 - 시퀀스형 자료형
for i in "abcdefg":
  print(i)
  
#각각의 문자열 리스트로 처리
for i in ["americano", "latte", "frafuchino"]:
  print(i)

#간격을 두고 세기
for i in range(1, 10, 2)
  # 1부터 10까지 2씩 증가시키면서 반복문 수행
  print(i)
  
#역순으로 반복문 수행
for i in range(10, 1, -1):
  # 10부터 1까지 -1씩 감소시키면서 반복문 수행
  print(i)
```
#### 8. while문
+ 조건이 만족하는 동안 반복 명령문을 수행
+ for문은 while문으로 변환 가능
  + 반복 실행횟수가 명확할 때는 for문, 명확하지 않을 떄는 while문을 주로 사용한다.
  
#### 9. 반복의 제어 - break, continue
+ break : 특정 조건에서 반복 종료
+ continue : 특정 조건에서 남은 반복 명령 skip
``` python
  for i in range(1,10)
    if i == 5:
      break
    print(i)
```
+ else : 반복 조건이 만족하지 않을 경우 반복 종료 시 1회 수행
  + break로 종료된 반복문은 else block이 수행되지 않음
``` python
  for i in range(*10):
    print(i,)
  else:
    print("EOP")
```
+ Debugging Loop : loop 내 많은 print문 활용 권장
``` python
  print("input deciaml number: *,)
  decimal = int(input())
  result = ""
  loop_counter = 0
  while (decimal > 0):
    temp_decimal_input = decimal
    temp_result_input = result
    
    remainder = decimal % 2
    decimal = decimal // 2
    result = str(remainder) + result
    
    print("---------",loop_counter, "loop value check ---------- ")
    print("Initial decimal:", temp_decimal_input, ", Remainder:", remainder, ", Initial result", temp_result_input)
    print("Output decimal:", decimal, "Output result:", result)
    print("----------------------------------------------------- ")
    print("")
    
    loop_counter += 1
print("Binary number is", result)
```
#### 10. 가변적인 중첩 반복문 (Variable nested loop)
+ 실제 프로그램에서는 반복문은 사용자의 입력에 따라 가변적으로 반복되고 하나의 반복이 아닌 중복되어 반복이 일어남
``` python
  #사람 별 평균을 구하는 코드
  student_score = [0,0,0,0,0]
  i = 0
  for subject in midterm_score:
    for score in subject:
      student_score[i] += score
      i += 1
    i = 0
  else:
    a, b, c, d, e = student_score    #학생 별 점수를 Unpacking
    student_average = [a/3, b/3, c/3, d/3, e/3]
    print(student_average)
```
#### 11. Debugging
+ 코드의 오류를 발견하여 수정하는 과정
+ 문법적 에러를 찾기 위한 에러메시지 분석
  + 들여쓰기
  + 오탈자
  + 대소문자 구분 안 함
+ 논리적 에러를 찾기 위한 테스트도 중요
+ 모든 문제는 Google + Stack Overflow로 해결 가능

----------------------
### String and advanced function concept
#### 1. 문자열(String)
+ 시퀀스 자료형으로 문자형 data를 메모리에 저장
+ 영문자 한 글자는 1byte의 메모리 공간을 사용
+ string은 1byte 크기로 한 글자씩 메모리 공간이 할당됨

#### 2. 1Byte의 메모리 공간?
+ 컴퓨터는 2진수로 데이터를 저장
+ 이진수 한 자릿수는 1bit로 저장됨
+ 즉 1bit는 0 또는 1
+ 1 byte = 8 bit = 2^8 = 256까지 저장 가능
+ 컴퓨터는 문자를 직접적으로 인식 X
  + 모든 데이터는 2진수로 인식
+ 이를 위해 2진수를 문자로 변환하는 표준 규칙을 정함 (아스키코드)

#### 3. 프로그램 언어에서 데이터 타입
+ 각 타입 별로 메모리 공간을 할당 받은 크기가 다름
+ 데이터 타입은 메모리의 효율적 활용을 위해 매우 중요

종류 | 타입 | 크기 | 표현 범위(32bit)
---- | ---- | ---- | ----
정수형 | int | 4바이트 | -2^31~2^31-`
정수형 | long | 무제한 | 무제한
실수형 | float | 8바이트 | 약 10^-308 ~ 10^+308

#### 4. 문자열 특징 
+ 인덱싱 (Indexing)
  + 문자열의 각 문자는 개별 주소(offset)을 갖고 있음
  + List와 같은 형태로 데이터를 처리함
+ 슬라이싱 (Slicing)
  + 문자열의 주소값을 기반으로 문자열의 부분값을 반황
+ 문자열 연산 및 포함여부 검사
  + 덧셈으로 a와 b 변수를 연결할 수 있음
  + 곱하기로 반복 연산이 가능
  + 'A' in a와 같은 구문으로 포함여부 확인 가능

#### 5. 문자열 함수

함수명 | 기능
---- | ----
len(a) | 문자열의 문자 개수를 반환
a.upper() | 대문자로 변환
a.lower() | 소문자로 변환
a.capitalize() | 첫 문자를 대문자로 변환, "i love you -> I love you"
a.title() | 제목 형태로 변환, "i love you -> I Love You"
a.count('abc') | 문자열 a에 'abc'가 들어간 횟수 반환
a.find('abc') | 문자열 a에 'abc'가 들어간 위치(오프셋) 반환
a.rfind('abc') | 문자열 a에 'abc'가 들어간 위치(오프셋) 반환 - 우측부터 탐색
a.startswith('abc') | 문자열 a가 'abc'로 시작하는 문자열여부 반환 
a.endswith('abc') | 문자열 a가 'abc'로 끝나는 문자열여부 반환
a.strip() | 좌우 공백을 없앰
a.rstrip() | 오른쪽 공백을 없앰
a.lstrip() | 왼쪽 공백을 없앰
a.split() | 공백을 기준으로 나눠 리스트로 반환
a.split('abc') | abc를 기준으로 나눠 리스트로 반환
a.isdigit() | 문자열이 숫자인지 여부 확인
a.islower() | 문자열이 소문자인지 여부 확인
a.isupper() | 문자열이 대문자인지 여부 확인

#### 6. 다양한 문자열 표현
+ 문자열 선언은 큰따옴표("")나 작은 따옴표 ('')를 활용
+ \'는 문자열 구분자가 아닌 출력 문자로 처리
+ a = "it's ok." -> 큰 따옴표와 작은 따옴표를 혼합하여 사용
+ 두줄 이상 저장하는 방법
  + \n 을 추가하여 줄 바꿈
  + 큰따옴표 또는 작은 따옴표를 세 번 연속 사용

``` python
  a = """ It's Ok
          I'm Happy.
          See you. """
```
+ raw_string을 활용하면 입력 문자열 그대로 활용하는 것이 가능하다.
``` python
  raw_string = r"테스트 입니다. \n 하하"
  print(raw_string)
```

#### 7. 함수 호출 방식 개요
+ 함수에서 Parameter를 전달하는 방식
  + 값에 의한 호출 (Call by Value) : 함수에 인자를 넘길 때 값만 넘김.
  + 참조에 의한 호출 (Call by Reference) : 함수에 인자를 넘길 때 메모리 주소를 넘김. 함수 내에 인자 값 변경 시, 호출자의 값도 변경됨
  + 객체 참조에 의한 호출 (Call by Object Reference) : 객체의 주소가 함수로 전달되는 방식 (파이썬의 방식)
    + 전달된 객체를 참조하여 변경 시 호출자에게 영향을 주나, 새로운 객체를 만들 경우 호출자에게 영향을 주지 않음.
    + 일반적인 상황에서는 값이 전달되면 복사를 하여 사용하는 것이 좋음
``` python
  def spam(eggs):
    eggs.append(1)  # 기존 객체의 주소값에 [1] 추가
    eggs = [2, 3]   # 새로운 객체 생성
  ham = [0]
  spam(ham)
  print(ham)    # [ 0, 1 ]
``` 

#### 8. 변수의 범위 (Scoping Rule)
+ 변수가 사용되는 범위 ( 함수 또는 메인 프로그램)
+ 지역 변수(Local Variable) : 함수내에서만 사용
+ 전역 변수(Global Variable) : 프로그램 전체에서 사용
+ 같은 이름의 변수를 사용하더라도, 실제 변수는 범위에 따라 다르다.
  + 전역변수는 함수에서 사용가능하다. 함수 내에 전역 변수와 같은 이름의 변수를 선언하면 새로운 지역 변수가 생긴다.
+ 함수 내에서 전역변수 사용 방법 - global 변수
``` python
def f():
  global s
  s = "I Love London!"
  print(s)

s = "I love Paris!'
f()
print(s)
```

#### 9. 재귀함수 (recursive function)
+ 자기자신을 호출하는 함수
+ 점화식과 같은 재귀적 수학 모형을 표현할 때 사용
+ 재귀 종료 조건 존재, 종료 조건까지 함수호출 반복

``` python
  def factorial(n):
    if n == 1:
      return 1
    else:
      return n * factorial(n-1)
  print(factorial(int(input("Input Number for Factorial Calculation: "))))
```

#### 10. Function Type Hint
+ 파이썬의 dynamic typing 기능은 처음 함수를 사용하는 사용자가 interface를 알기 어렵다는 단점이 있음
+ python 3.5 버전 이후로는 type hints 기능 제공
``` python
  def type_hint_example(name: str) -> str:
    return f"Hello, {name}"
```
+ 장점
  + 사용자에게 인터페이스를 명확히 알려줄 수 있다.
  + 함수의 문서화시 Parameter에 대한 정보를 명확히 알 수 있다.
  + 시스템 전체적인 안정성을 확보할 수 있다.
  
#### 11. function docstring
+ 파이썬에 대한 상세스펙을 사전에 작성 -> 함수 사용자의 이행도 UP
+ 세개의 따옴표로 docstring 영역 표시(함수명 아래)
+ VS Code에서 "Python Docstring Generator"라는 라이브러리를 생성할 수 있음
  + CTRL+Shift+P를 클릭해서 VS Code 기능들을 사용할 수 있음
  + type hint를 준 상태에서 Docstring을 호출
``` python
  def kos_root():
    """Return the pathname of the KOS root directiory."""
    global _kos_root
    if _kos_root: return _kos_root
    #...
```

#### 12. 함수 작성 가이드 라인
+ 함수는 가능하면 짧게 작성할 것 (줄 수를 줄일 것)
+ 함수 이름에 함수의 역할, 의도가 명확히 들어낼 것 ( V, O 형태 )
+ 하나의 함수에는 유사한 역할을 하는 코드만 포함
+ 인자로 받은 값 자체를 바꾸진 말 것 (임시변수 선언)
+ 함수는 언제 만드는 가?
  + 공통 코드는 함수로 작성
  + 복잡한 수식, 조건은 함수로

#### 13. 파이썬 코딩 컨벤션(코딩 규칙)
+ 명확한 규칙은 없음
+ 때로는 팀마다, 프로젝트마다 따로
+ 중요한 건 일관성!!!
+ 읽기 좋은 코드가 좋은 코드
+ 들여쓰기는 Tab or 4 Space 논쟁! (일반적으로 4 Space를 권장)
+ 한 줄은 최대 79자까지
+ 불필요한 공백은 피함
+ 연산자는 1칸 이상 안 띄움
+ 주석은 항상 갱신, 불필요한 주석은 삭제
+ 코드의 마지막에는 항상 한 줄 추가
+ 대문자 O, 소문자 l, 대문자 I금지
+ 함수명은 소문자로 구성, 필요하면 밑줄로 나눔
+ flake8로 코드 컨벤션을 검토할 수 있음
  + 설치 명령어 : conda install -c anaconda flake8
  + 검사 명어 : flake8 {{파일명.py}}
+ black 모듈을 활용하여 거의 pep8에 근접한 수준으로 수정( 파일을 내용을 직접 변경 )
  + black {{파일명.py}}

### 추가 학습
#### 1. git 설치
+ 설치 URL : https://git-scm.com/
+ git 명령어
  + git --version : 버전 확인
  + git config --global user.name sunheonpark : 이름 설정
  + git config --global user.email. sunheonpark@gmail.com : 이메일 설정
  + 원하는 경로로 이동 -> git clone {{git URL}} : git에 있는 내용을 다운로드
  + git add {{파일명}} : 파일을 git에 추가
  + git status : 추가된 git 현황을 확인
  + git commit -m "Add Text File [{{파일명}}]" : 설명을 추가해서 커밋
  + git push : commit한 내용을 git에 적용

#### 2. unnitest 명령어
+ python -m unittest test_*.py
