---
layout: page
title: Python Numpy Tutorial
permalink: /python-numpy-tutorial/
---

<!--
Python:
  Simple data types
    integer, float, string
  Compound data types
    tuple, list, dictionary, set
  Flow control
    if, while, for, try, with
  Comprehensions, generators
  Functions
  Classes
  Standard library
    json, collections, itertools

Numpy
-->

이 튜토리얼은 [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) 에 의해 작성되었습니다.

cs231n 수업의 모든 과제에서는 프로그래밍 언어로 파이썬을 사용할 것입니다.
파이썬은 그 자체만으로도 훌륭한 범용 프로그래밍 언어이지만, 몇몇 라이브러리(numpy, scipy, matplotlib)의 도움으로
계산과학 분야에서 강력한 개발 환경을 갖추게 됩니다.  

많은 분들이 파이썬과 numpy를 경험 해보셨을 거라고 생각합니다. 경험하지 못했을지라도 이 문서를 통해  
'프로그래밍 언어로서의 파이썬'과 '파이썬을 계산과학에 활용하는 법'을 빠르게 훑을 수 있습니다.

만약 Matlab을 사용해보셨다면, [Matlab 사용자를 위한 numpy](http://wiki.scipy.org/NumPy_for_Matlab_Users) 페이지를 추천해 드립니다.

또한 [CS 228](http://cs.stanford.edu/~ermon/cs228/index.html) 수업을 위해 [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) 와 [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335) 가 만든 [이 튜토리얼의 IPython notebook 버전](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb) 도 참조할 수 있습니다.

목차:

- [파이썬](#python)
  - [기본 자료형](#python-basic)
  - [컨테이너](#python-containers)
      - [리스트](#python-lists)
      - [딕셔너리](#python-dicts)
      - [집합](#python-sets)
      - [튜플](#python-tuples)
  - [함수](#python-functions)
  - [클래스](#python-classes)
- [Numpy](#numpy)
  - [배열](#numpy-arrays)
  - [배열 인덱싱](#numpy-array-indexing)
  - [데이터 타입](#numpy-datatypes)
  - [배열 연산](#numpy-math)
  - [브로드캐스팅](#numpy-broadcasting)
- [SciPy](#scipy)
  - [이미지 작업](#scipy-image)
  - [MATLAB 파일](#scipy-matlab)
  - [두 점 사이의 거리](#scipy-dist)
- [Matplotlib](#matplotlib)
  - [Plotting](#matplotlib-plotting)
  - [Subplots](#matplotlib-subplots)
  - [이미지](#matplotlib-images)

<a name='python'></a>

## 파이썬
파이썬은 고급 프로그래밍 언어(사람이 이해하기 쉽게 작성된 언어)이며, 다중패러다임을 지원하는 동적 프로그래밍 언어입니다.
짧지만 가독성 높은 코드 몇 줄로 수준 높은 아이디어들을 표현할 수 있기에 파이썬 코드는 거의 수도코드처럼 보인다고도 합니다.
아래는 quicksort 알고리즘을 파이썬으로 구현한 예시입니다:

~~~python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) / 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print quicksort([3,6,8,10,1,2,1])
# 출력 "[1, 1, 2, 3, 6, 8, 10]"
~~~

### 파이썬 버전
현재 파이썬에는 두 가지 버전이 있습니다. 파이썬 2.7 그리고 파이썬 3.4입니다.
혼란스럽게도, 파이썬3은 기존 파이썬2와 호환되지 않게 변경된 부분이 있습니다.
그러므로 파이썬 2.7로 쓰여진 코드는 3.4환경에서 동작하지 않고 그 반대도 마찬가지입니다.
이 수업에선 파이썬 2.7을 사용합니다.

커맨드라인에 아래의 명령어를 입력해서 현재 설치된 파이썬 버전을 확인할 수 있습니다.
`python --version`.

<a name='python-basic'></a>

### 기본 자료형

다른 프로그래밍 언어들처럼, 파이썬에는 정수, 실수, 불리언, 문자열 같은 기본 자료형이 있습니다.
파이썬 기본 자료형 역시 다른 프로그래밍 언어와 유사합니다.

**숫자:** 다른 언어와 마찬가지로 파이썬의 정수형(Integers)과 실수형(floats) 데이터 타입 역시 동일한 역할을 합니다 :

~~~python
x = 3
print type(x) # 출력 "<type 'int'>"
print x       # 출력 "3"
print x + 1   # 덧셈; 출력 "4"
print x - 1   # 뺄셈; 출력 "2"
print x * 2   # 곱셈; 출력 "6"
print x ** 2  # 제곱; 출력 "9"
x += 1
print x  # 출력 "4"
x *= 2
print x  # 출력 "8"
y = 2.5
print type(y) # 출력 "<type 'float'>"
print y, y + 1, y * 2, y ** 2 # 출력 "2.5 3.5 5.0 6.25"
~~~
다른 언어들과는 달리, 파이썬에는 증감 단항연산자(`x++`, `x--`)가 없습니다.

파이썬 역시 long 정수형과 복소수 데이터 타입이 구현되어 있습니다.
자세한 사항은 [문서](https://docs.python.org/2/library/stdtypes.html#numeric-types-int-float-long-complex)에서 찾아볼 수 있습니다.

**불리언(Booleans):** 파이썬에는 논리 자료형의 모든 연산자가 구현되어 있습니다.
그렇지만 기호(`&&`, `||`, 등.) 대신 영어 단어로 구현되어 있습니다 :

~~~python
t = True
f = False
print type(t) # 출력 "<type 'bool'>"
print t and f # 논리 AND; 출력 "False"
print t or f  # 논리 OR; 출력 "True"
print not t   # 논리 NOT; 출력 "False"
print t != f  # 논리 XOR; 출력 "True"
~~~

**문자열:** 파이썬은 문자열과 연관된 다양한 기능을 지원합니다:

~~~python
hello = 'hello'   # String 문자열을 표현할 땐 따옴표나
world = "world"   # 쌍따옴표가 사용됩니다; 어떤 걸 써도 상관없습니다.
print hello       # 출력 "hello"
print len(hello)  # 문자열 길이; 출력 "5"
hw = hello + ' ' + world  # 문자열 연결
print hw  # 출력 "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf 방식의 문자열 서식 지정
print hw12  # 출력 "hello world 12"
~~~

문자열 객체에는 유용한 메소드들이 많습니다; 예를 들어:

~~~python
s = "hello"
print s.capitalize()  # 문자열을 대문자로 시작하게 함; 출력 "Hello"
print s.upper()       # 모든 문자를 대문자로 바꿈; 출력 "HELLO"
print s.rjust(7)      # 문자열 오른쪽 정렬, 빈공간은 여백으로 채움; 출력 "  hello"
print s.center(7)     # 문자열 가운데 정렬, 빈공간은 여백으로 채움; 출력 " hello "
print s.replace('l', '(ell)')  # 첫 번째 인자로 온 문자열을 두 번째 인자 문자열로 바꿈;
                               # 출력 "he(ell)(ell)o"
print '  world '.strip()  # 문자열 앞뒤 공백 제거; 출력 "world"
~~~
모든 문자열 메소드는 [문서](https://docs.python.org/2/library/stdtypes.html#string-methods)에서 찾아볼 수 있습니다.

<a name='python-containers'></a>

### 컨테이너
파이썬은 다음과 같은 컨테이너 타입이 구현되어 있습니다: 리스트, 딕셔너리, 집합, 튜플

<a name='python-lists'></a>

#### 리스트
리스트는 파이썬에서 배열 같은 존재입니다. 그렇지만 배열과 달리 크기 변경이 가능하고
서로 다른 자료형일지라도 하나의 리스트에 저장될 수 있습니다:

~~~python
xs = [3, 1, 2]   # 리스트 생성
print xs, xs[2]  # 출력 "[3, 1, 2] 2"
print xs[-1]     # 인덱스가 음수일 경우 리스트의 끝에서부터 세어짐; 출력 "2"
xs[2] = 'foo'    # 리스트는 자료형이 다른 요소들을 저장할 수 있습니다
print xs         # 출력 "[3, 1, 'foo']"
xs.append('bar') # 리스트의 끝에 새 요소 추가
print xs         # 출력 "[3, 1, 'foo', 'bar']"
x = xs.pop()     # 리스트의 마지막 요소 삭제하고 반환
print x, xs      # 출력 "bar [3, 1, 'foo']"
~~~
마찬가지로, 리스트에 대해 자세한 사항은 [문서](https://docs.python.org/2/tutorial/datastructures.html#more-on-lists)에서 찾아볼 수 있습니다.

**슬라이싱:**
리스트의 요소로 한 번에 접근하는 것 이외에도, 파이썬은 리스트의 일부분에만 접근하는 간결한 문법을 제공합니다;
이를 *슬라이싱*이라고 합니다:

~~~python
nums = range(5)    # range는 파이썬에 구현되어 있는 함수이며 정수들로 구성된 리스트를 만듭니다
print nums         # 출력 "[0, 1, 2, 3, 4]"
print nums[2:4]    # 인덱스 2에서 4(제외)까지 슬라이싱; 출력 "[2, 3]"
print nums[2:]     # 인덱스 2에서 끝까지 슬라이싱; 출력 "[2, 3, 4]"
print nums[:2]     # 처음부터 인덱스 2(제외)까지 슬라이싱; 출력 "[0, 1]"
print nums[:]      # 전체 리스트 슬라이싱; 출력 ["0, 1, 2, 3, 4]"
print nums[:-1]    # 슬라이싱 인덱스는 음수도 가능; 출력 ["0, 1, 2, 3]"
nums[2:4] = [8, 9] # 슬라이스된 리스트에 새로운 리스트 할당
print nums         # 출력 "[0, 1, 8, 9, 4]"
~~~
numpy 배열 부분에서 다시 슬라이싱을 보게 될 것입니다.

**반복문:** 아래와 같이 리스트의 요소들을 반복해서 조회할 수 있습니다:

~~~python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print animal
# 출력 "cat", "dog", "monkey", 한 줄에 하나씩 출력.
~~~

만약 반복문 내에서 리스트 각 요소의 인덱스에 접근하고 싶다면, 'enumerate' 함수를 사용하세요:

~~~python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)
# 출력 "#1: cat", "#2: dog", "#3: monkey", 한 줄에 하나씩 출력.
~~~

**리스트 comprehensions:**
프로그래밍을 하다 보면, 자료형을 변환해야 하는 경우가 자주 있습니다.
간단한 예를 들자면, 숫자의 제곱을 계산하는 다음의 코드를 보세요:


~~~python
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print squares   # 출력 [0, 1, 4, 9, 16]
~~~

**리스트 comprehension**을 이용해 이 코드를 더 간단하게 만들 수 있습니다:

~~~python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print squares   # 출력 [0, 1, 4, 9, 16]
~~~

리스트 comprehensions에 조건을 추가할 수도 있습니다:

~~~python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print even_squares  # 출력 "[0, 4, 16]"
~~~

<a name='python-dicts'></a>

#### 딕셔너리
자바의 '맵', 자바스크립트의 '오브젝트'와 유사하게, 파이썬의 '딕셔너리'는 (열쇠, 값) 쌍을 저장합니다.
아래와 같은 방식으로 딕셔너리를 사용할 수 있습니다:

~~~python
d = {'cat': 'cute', 'dog': 'furry'}  # 새로운 딕셔너리를 만듭니다
print d['cat']       # 딕셔너리의 값을 받음; 출력 "cute"
print 'cat' in d     # 딕셔너리가 주어진 열쇠를 가졌는지 확인; 출력 "True"
d['fish'] = 'wet'    # 딕셔너리의 값을 지정
print d['fish']      # 출력 "wet"
# print d['monkey']  # KeyError: 'monkey' not a key of d
print d.get('monkey', 'N/A')  # 딕셔너리의 값을 받음. 존재하지 않는 다면 'N/A'; 출력 "N/A"
print d.get('fish', 'N/A')    # 딕셔너리의 값을 받음. 존재하지 않는 다면 'N/A'; 출력 "wet"
del d['fish']        # 딕셔너리에 저장된 요소 삭제
print d.get('fish', 'N/A') # "fish"는 더 이상 열쇠가 아님; 출력 "N/A"
~~~
딕셔너리에 관해 더 알고 싶다면 [문서](https://docs.python.org/2/library/stdtypes.html#dict)를 참조하세요.

**반복문:** 딕셔너리의 열쇠는 쉽게 반복될 수 있습니다:

~~~python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print 'A %s has %d legs' % (animal, legs)
# 출력 "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs", 한 줄에 하나씩 출력.
~~~

만약 열쇠와, 그에 상응하는 값에 접근하고 싶다면, 'iteritems' 메소드를 사용하세요:

~~~python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.iteritems():
    print 'A %s has %d legs' % (animal, legs)
# 출력 "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs", 한 줄에 하나씩 출력.
~~~

**딕셔너리 comprehensions:**
리스트 comprehensions과 유사한 딕셔너리 comprehensions을 통해 손쉽게 딕셔너리를 만들 수 있습니다.
예시:

~~~python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print even_num_to_square  # 출력 "{0: 0, 2: 4, 4: 16}"
~~~

<a name='python-sets'></a>

#### 집합
집합은 순서 구분이 없고 서로 다른 요소 간의 모임입니다. 예시:

~~~python
animals = {'cat', 'dog'}
print 'cat' in animals   # 요소가 집합에 포함되어 있는지 확인; 출력 "True"
print 'fish' in animals  # 출력 "False"
animals.add('fish')      # 요소를 집합에 추가
print 'fish' in animals  # 출력 "True"
print len(animals)       # 집합에 포함된 요소의 수; 출력 "3"
animals.add('cat')       # 이미 포함되어있는 요소를 추가할 경우 아무 변화 없음
print len(animals)       # 출력 "3"
animals.remove('cat')    # Remove an element from a set
print len(animals)       # 출력 "2"
~~~

마찬가지로, 집합에 관해 더 알고 싶다면 [문서](https://docs.python.org/2/library/sets.html#set-objects)를 참조하세요.

**반복문:**
집합을 반복하는 구문은 리스트 반복 구문과 동일합니다;
그러나 집합은 순서가 없어서, 어떤 순서로 반복될지 추측할 순 없습니다:

~~~python
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)
# 출력 "#1: fish", "#2: dog", "#3: cat", 한 줄에 하나씩 출력.
~~~

**집합 comprehensions:**
리스트, 딕셔너리와 마찬가지로 집합 comprehensions을 통해 손쉽게 집합을 만들수 있습니다.

~~~python
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print nums  # 출력 "set([0, 1, 2, 3, 4, 5])"
~~~

<a name='python-tuples'></a>

#### 튜플
튜플은 요소 간 순서가 있으며 값이 변하지 않는 리스트입니다.
튜플은 많은 면에서 리스트와 유사합니다; 가장 중요한 차이점은 튜플은 '딕셔너리의 열쇠'와 '집합의 요소'가 될 수 있지만, 리스트는 불가능하다는 점입니다.
여기 간단한 예시가 있습니다:

~~~python
d = {(x, x + 1): x for x in range(10)}  # 튜플을 열쇠로 하는 딕셔너리 생성
t = (5, 6)       # 튜플 생성
print type(t)    # 출력 "<type 'tuple'>"
print d[t]       # 출력 "5"
print d[(1, 2)]  # 출력 "1"
~~~
[문서](https://docs.python.org/2/tutorial/datastructures.html#tuples-and-sequences)에 튜플에 관한 더 많은 정보가 있습니다.

<a name='python-functions'></a>

### 함수
파이썬 함수는 'def' 키워드를 통해 정의됩니다. 예시:

~~~python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print sign(x)
# 출력 "negative", "zero", "positive", 한 줄에 하나씩 출력.
~~~

가끔은 아래처럼 선택적으로 인자를 받는 함수를 정의할 때도 있습니다:

~~~python
def hello(name, loud=False):
    if loud:
        print 'HELLO, %s!' % name.upper()
    else:
        print 'Hello, %s' % name

hello('Bob') # 출력 "Hello, Bob"
hello('Fred', loud=True)  # 출력 "HELLO, FRED!"
~~~
파이썬 함수에 관한 더 많은 정보는 [문서](https://docs.python.org/2/tutorial/controlflow.html#defining-functions)를 참조하세요.

<a name='python-classes'></a>

### 클래스

파이썬에서 클래스를 정의하는 구문은 복잡하지 않습니다:

~~~python
class Greeter(object):

    # 생성자
    def __init__(self, name):
        self.name = name  # 인스턴스 변수 선언

    # 인스턴스 메소드
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name

g = Greeter('Fred')  # Greeter 클래스의 인스턴스 생성
g.greet()            # 인스턴스 메소드 호출; 출력 "Hello, Fred"
g.greet(loud=True)   # 인스턴스 메소드 호출; 출력 "HELLO, FRED!"
~~~
파이썬 클래스에 관한 더 많은 정보는 [문서](https://docs.python.org/2/tutorial/classes.html)를 참조하세요.

<a name='numpy'></a>

## Numpy

[Numpy](http://www.numpy.org/)는 파이썬이 계산과학분야에 이용될때 핵심 역할을 하는 라이브러리입니다.
Numpy는 고성능의 다차원 배열 객체와 이를 다룰 도구를 제공합니다. 만약 MATLAB에 익숙한 분이라면 Numpy 학습을 시작하는데 있어
[이 튜토리얼](http://wiki.scipy.org/NumPy_for_Matlab_Users)이 유용할 것입니다.

<a name='numpy-arrays'></a>

### 배열
Numpy 배열은 동일한 자료형을 가지는 값들이 격자판 형태로 있는 것입니다. 각각의 값들은 튜플(이때 튜플은 양의 정수만을 요소값으로 갖습니다.) 형태로 색인 됩니다.
*rank*는 배열이 몇 차원인지를 의미합니다; *shape*는 는 각 차원의 크기를 알려주는 정수들이 모인 튜플입니다.

파이썬의 리스트를 중첩해 Numpy 배열을 초기화 할 수 있고, 대괄호를 통해 각 요소에 접근할 수 있습니다:

~~~python
import numpy as np

a = np.array([1, 2, 3])  # rank가 1인 배열 생성
print type(a)            # 출력 "<type 'numpy.ndarray'>"
print a.shape            # 출력 "(3,)"
print a[0], a[1], a[2]   # 출력 "1 2 3"
a[0] = 5                 # 요소를 변경
print a                  # 출력 "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])   # rank가 2인 배열 생성
print b.shape                     # 출력 "(2, 3)"
print b[0, 0], b[0, 1], b[1, 0]   # 출력 "1 2 4"
~~~

리스트의 중첩이 아니더라도 Numpy는 배열을 만들기 위한 다양한 함수를 제공합니다.

~~~python
import numpy as np

a = np.zeros((2,2))  # 모든 값이 0인 배열 생성
print a              # 출력 "[[ 0.  0.]
                     #       [ 0.  0.]]"

b = np.ones((1,2))   # 모든 값이 1인 배열 생성
print b              # 출력 "[[ 1.  1.]]"

c = np.full((2,2), 7) # 모든 값이 특정 상수인 배열 생성
print c               # 출력 "[[ 7.  7.]
                      #       [ 7.  7.]]"

d = np.eye(2)        # 2x2 단위행렬 생성
print d              # 출력 "[[ 1.  0.]
                     #       [ 0.  1.]]"

e = np.random.random((2,2)) # 임의의 값으로 채워진 배열 생성
print e                     # 임의의 값 출력 "[[ 0.91940167  0.08143941]
                            #                  [ 0.68744134  0.87236687]]"
~~~
배열 생성에 관한 다른 방법들은 [문서](http://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation)를 참조하세요.

<a name='numpy-array-indexing'></a>

### 배열 인덱싱
Numpy는 배열을 인덱싱하는 몇 가지 방법을 제공합니다.

**슬라이싱:**
파이썬 리스트와 유사하게, Numpy 배열도 슬라이싱이 가능합니다. Numpy 배열은 다차원인 경우가 많기에, 각 차원별로 어떻게 슬라이스할건지 명확히 해야 합니다:

~~~python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# 슬라이싱을 이용하여 첫 두 행과 1열, 2열로 이루어진 부분배열을 만들어 봅시다;
# b는 shape가 (2,2)인 배열이 됩니다:
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# 슬라이싱된 배열은 원본 배열과 같은 데이터를 참조합니다, 즉 슬라이싱된 배열을 수정하면
# 원본 배열 역시 수정됩니다.
print a[0, 1]   # 출력 "2"
b[0, 0] = 77    # b[0, 0]은 a[0, 1]과 같은 데이터입니다
print a[0, 1]   # 출력 "77"
~~~

정수를 이용한 인덱싱과 슬라이싱을 혼합하여 사용할 수 있습니다.
하지만 이렇게 할 경우, 기존의 배열보다 낮은 rank의 배열이 얻어집니다.
이는 MATLAB이 배열을 다루는 방식과 차이가 있습니다.

슬라이싱:

~~~python
import numpy as np

# 아래와 같은 요소를 가지는 rank가 2이고 shape가 (3, 4)인 배열 생성
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# 배열의 중간 행에 접근하는 두 가지 방법이 있습니다.
# 정수 인덱싱과 슬라이싱을 혼합해서 사용하면 낮은 rank의 배열이 생성되지만,
# 슬라이싱만 사용하면 원본 배열과 동일한 rank의 배열이 생성됩니다.
row_r1 = a[1, :]    # 배열a의 두 번째 행을 rank가 1인 배열로
row_r2 = a[1:2, :]  # 배열a의 두 번째 행을 rank가 2인 배열로
print row_r1, row_r1.shape  # 출력 "[5 6 7 8] (4,)"
print row_r2, row_r2.shape  # 출력 "[[5 6 7 8]] (1, 4)"

# 행이 아닌 열의 경우에도 마찬가지입니다:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print col_r1, col_r1.shape  # 출력 "[ 2  6 10] (3,)"
print col_r2, col_r2.shape  # 출력 "[[ 2]
                            #       [ 6]
                            #       [10]] (3, 1)"
~~~

**정수 배열 인덱싱:**
Numpy 배열을 슬라이싱하면, 결과로 얻어지는 배열은 언제나 원본 배열의 부분 배열입니다.
그러나 정수 배열 인덱싱을 한다면, 원본과 다른 배열을 만들 수 있습니다.
여기에 예시가 있습니다:

~~~python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# 정수 배열 인덱싱의 예.
# 반환되는 배열의 shape는 (3,)
print a[[0, 1, 2], [0, 1, 0]]  # 출력 "[1 4 5]"

# 위에서 본 정수 배열 인덱싱 예제는 다음과 동일합니다:
print np.array([a[0, 0], a[1, 1], a[2, 0]])  # 출력 "[1 4 5]"

# 정수 배열 인덱싱을 사용할 때,
# 원본 배열의 같은 요소를 재사용할 수 있습니다:
print a[[0, 0], [1, 1]]  # 출력 "[2 2]"

# 위 예제는 다음과 동일합니다
print np.array([a[0, 1], a[0, 1]])  # 출력 "[2 2]"
~~~

정수 배열 인덱싱을 유용하게 사용하는 방법 중 하나는 행렬의 각 행에서 하나의 요소를 선택하거나 바꾸는 것입니다:

~~~python
import numpy as np

# 요소를 선택할 새로운 배열 생성
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print a  # 출력 "array([[ 1,  2,  3],
         #             [ 4,  5,  6],
         #             [ 7,  8,  9],
         #             [10, 11, 12]])"

# 인덱스를 저장할 배열 생성
b = np.array([0, 2, 0, 1])


# b에 저장된 인덱스를 이용해 각 행에서 하나의 요소를 선택합니다
print a[np.arange(4), b]  # 출력 "[ 1  6  7 11]"

# b에 저장된 인덱스를 이용해 각 행에서 하나의 요소를 변경합니다
a[np.arange(4), b] += 10

print a  # 출력 "array([[11,  2,  3],
         #             [ 4,  5, 16],
         #             [17,  8,  9],
         #             [10, 21, 12]])
~~~

**불리언 배열 인덱싱:**
불리언 배열 인덱싱을 통해 배열 속 요소를 취사선택할 수 있습니다.
불리언 배열 인덱싱은 특정 조건을 만족하게 하는 요소만 선택하고자 할 때 자주 사용됩니다.
다음은 그 예시입니다:

~~~python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)  # 2보다 큰 a의 요소를 찾습니다;
                    # 이 코드는 a와 shape가 같고 불리언 자료형을 요소로 하는 numpy 배열을 반환합니다,
                    # bool_idx의 각 요소는 동일한 위치에 있는 a의
                    # 요소가 2보다 큰지를 말해줍니다.

print bool_idx      # 출력 "[[False False]
                    #       [ True  True]
                    #       [ True  True]]"

# 불리언 배열 인덱싱을 통해 bool_idx에서
# 참 값을 가지는 요소로 구성되는
# rank 1인 배열을 구성할 수 있습니다.
print a[bool_idx]  # 출력 "[3 4 5 6]"

# 위에서 한 모든것을 한 문장으로 할 수 있습니다:
print a[a > 2]     # 출력 "[3 4 5 6]"
~~~

튜토리얼을 간결히 하고자 numpy 배열 인덱싱에 관한 많은 내용을 생략했습니다.
조금 더 알고싶다면 [문서](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)를 참조하세요.

<a name='numpy-datatypes'></a>

### 자료형
Numpy 배열은 동일한 자료형을 가지는 값들이 격자판 형태로 있는 것입니다.
Numpy에선 배열을 구성하는 데 사용할 수 있는 다양한 숫자 자료형을 제공합니다.
Numpy는 배열이 생성될 때 자료형을 스스로 추측합니다, 그러나 배열을 생성할 때 명시적으로 특정 자료형을 지정할 수도 있습니다. 예시:

~~~python
import numpy as np

x = np.array([1, 2])  # Numpy가 자료형을 추측해서 선택
print x.dtype         # 출력 "int64"

x = np.array([1.0, 2.0])  # Numpy가 자료형을 추측해서 선택
print x.dtype             # 출력 "float64"

x = np.array([1, 2], dtype=np.int64)  # 특정 자료형을 명시적으로 지정
print x.dtype                         # 출력 "int64"
~~~
Numpy 자료형에 관한 자세한 사항은 [문서](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)를 참조하세요.

<a name='numpy-math'></a>

### 배열 연산
기본적인 수학함수는 배열의 각 요소별로 동작하며 연산자를 통해 동작하거나 numpy 함수모듈을 통해 동작합니다:

~~~python
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# 요소별 합; 둘 다 다음의 배열을 만듭니다
# [[ 6.0  8.0]
#  [10.0 12.0]]
print x + y
print np.add(x, y)

# 요소별 차; 둘 다 다음의 배열을 만듭니다
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print x - y
print np.subtract(x, y)

# 요소별 곱; 둘 다 다음의 배열을 만듭니다
# [[ 5.0 12.0]
#  [21.0 32.0]]
print x * y
print np.multiply(x, y)

# 요소별 나눗셈; 둘 다 다음의 배열을 만듭니다
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print x / y
print np.divide(x, y)

# 요소별 제곱근; 다음의 배열을 만듭니다
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print np.sqrt(x)
~~~

MATLAB과 달리, '*'은 행렬 곱이 아니라 요소별 곱입니다. Numpy에선 벡터의 내적, 벡터와 행렬의 곱, 행렬곱을 위해 '*'대신 'dot'함수를 사용합니다. 'dot'은 Numpy 모듈 함수로서도 배열 객체의 인스턴스 메소드로서도 이용 가능한 합수입니다:

~~~python
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# 벡터의 내적; 둘 다 결과는 219
print v.dot(w)
print np.dot(v, w)

# 행렬과 벡터의 곱; 둘 다 결과는 rank 1인 배열 [29 67]
print x.dot(v)
print np.dot(x, v)

# 행렬곱; 둘 다 결과는 rank 2인 배열
# [[19 22]
#  [43 50]]
print x.dot(y)
print np.dot(x, y)
~~~

Numpy는 배열 연산에 유용하게 쓰이는 많은 함수를 제공합니다. 가장 유용한 건 'sum'입니다:

~~~python
import numpy as np

x = np.array([[1,2],[3,4]])

print np.sum(x)  # 모든 요소를 합한 값을 연산; 출력 "10"
print np.sum(x, axis=0)  # 각 열에 대한 합을 연산; 출력 "[4 6]"
print np.sum(x, axis=1)  # 각 행에 대한 합을 연산; 출력 "[3 7]"
~~~
Numpy가 제공하는 모든 수학함수의 목록은 [문서](http://docs.scipy.org/doc/numpy/reference/routines.math.html)를 참조하세요.

배열연산을 하지 않더라도, 종종 배열의 모양을 바꾸거나 데이터를 처리해야 할 때가 있습니다.
가장 간단한 예는 행렬의 주 대각선을 기준으로 대칭되는 요소끼리 뒤바꾸는 것입니다; 이를 전치라고 하며 행렬을 전치하기 위해선, 간단하게 배열 객체의 'T' 속성을 사용하면 됩니다:

~~~python
import numpy as np

x = np.array([[1,2], [3,4]])
print x    # 출력 "[[1 2]
           #          [3 4]]"
print x.T  # 출력 "[[1 3]
           #          [2 4]]"

# rank 1인 배열을 전치할 경우 아무 일도 일어나지 않습니다:
v = np.array([1,2,3])
print v    # 출력 "[1 2 3]"
print v.T  # 출력 "[1 2 3]"
~~~
Numpy는 배열을 다루는 다양한 함수들을 제공합니다; 이러한 함수의 전체 목록은 [문서](http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)를 참조하세요.


<a name='numpy-broadcasting'></a>

### 브로드캐스팅
브로트캐스팅은 Numpy에서 shape가 다른 배열 간에도 산술 연산이 가능하게 하는 메커니즘입니다. 종종 작은 배열과 큰 배열이 있을 때, 큰 배열을 대상으로 작은 배열을 여러 번 연산하고자 할 때가 있습니다. 예를 들어, 행렬의 각 행에 상수 벡터를 더하는 걸 생각해보세요. 이는 다음과 같은 방식으로 처리될 수 있습니다:

~~~python
import numpy as np

# 행렬 x의 각 행에 벡터 v를 더한 뒤,
# 그 결과를 행렬 y에 저장하고자 합니다
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # x와 동일한 shape를 가지며 비어있는 행렬 생성

# 명시적 반복문을 통해 행렬 x의 각 행에 벡터 v를 더하는 방법
for i in range(4):
    y[i, :] = x[i, :] + v

# 이제 y는 다음과 같습니다
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print y
~~~

위의 방식대로 하면 됩니다; 그러나 'x'가 매우 큰 행렬이라면, 파이썬의 명시적 반복문을 이용한 위 코드는 매우 느려질 수 있습니다. 벡터 'v'를 행렬 'x'의 각 행에 더하는 것은 'v'를 여러 개 복사해 수직으로 쌓은 행렬 'vv'를 만들고 이 'vv'를 'x'에 더하는것과 동일합니다. 이 과정을 아래의 코드로 구현할 수 있습니다:

~~~python
import numpy as np

# 벡터 v를 행렬 x의 각 행에 더한 뒤,
# 그 결과를 행렬 y에 저장하고자 합니다
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))  # v의 복사본 4개를 위로 차곡차곡 쌓은 것이 vv
print vv                 # 출력 "[[1 0 1]
                         #       [1 0 1]
                         #       [1 0 1]
                         #       [1 0 1]]"
y = x + vv  # x와 vv의 요소별 합
print y  # 출력 "[[ 2  2  4
         #       [ 5  5  7]
         #       [ 8  8 10]
         #       [11 11 13]]"
~~~

Numpy 브로드캐스팅을 이용한다면 이렇게 v의 복사본을 여러 개 만들지 않아도 동일한 연산을 할 수 있습니다.
아래는 브로드캐스팅을 이용한 예시 코드입니다:

~~~python
import numpy as np

# 벡터 v를 행렬 x의 각 행에 더한 뒤,
# 그 결과를 행렬 y에 저장하고자 합니다
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # 브로드캐스팅을 이용하여 v를 x의 각 행에 더하기
print y  # 출력 "[[ 2  2  4]
         #       [ 5  5  7]
         #       [ 8  8 10]
         #       [11 11 13]]"
~~~

`x`의 shape가 `(4, 3)`이고 `v`의 shape가 `(3,)`라도 브로드캐스팅으로 인해 `y = x + v`는 문제없이 수행됩니다;
이때 'v'는 'v'의 복사본이 차곡차곡 쌓인 shape `(4, 3)`처럼 간주되어 'x'와 동일한 shape가 되며 이들 간의 요소별 덧셈연산이 y에 저장됩니다.

두 배열의 브로드캐스팅은 아래의 규칙을 따릅니다:

1. 두 배열이 동일한 rank를 가지고 있지 않다면, 낮은 rank의 1차원 배열이 높은 rank 배열의 shape로 간주합니다.
2. 특정 차원에서 두 배열이 동일한 크기를 갖거나, 두 배열 중 하나의 크기가 1이라면 그 두 배열은 특정 차원에서 *compatible*하다고 여겨집니다.
3. 두 행렬이 모든 차원에서 compatible하다면, 브로드캐스팅이 가능합니다.
4. 브로드캐스팅이 이뤄지면, 각 배열 shape의 요소별 최소공배수로 이루어진 shape가 두 배열의 shape로 간주합니다.
5. 차원에 상관없이 크기가 1인 배열과 1보다 큰 배열이 있을 때, 크기가 1인 배열은 자신의 차원 수만큼 복사되어 쌓인 것처럼 간주합니다.

설명이 이해하기 부족하다면 [scipy문서](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)나 [scipy위키](http://wiki.scipy.org/EricsBroadcastingDoc)를 참조하세요.

브로드캐스팅을 지원하는 함수를 *universal functions*라고 합니다.
*universal functions* 목록은 [문서](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs)를 참조하세요.

브로드캐스팅을 응용한 예시들입니다:

~~~python
import numpy as np

# 벡터의 외적을 계산
v = np.array([1,2,3])  # v의 shape는 (3,)
w = np.array([4,5])    # w의 shape는 (2,)
# 외적을 계산하기 위해, 먼저 v를 shape가 (3,1)인 행벡터로 바꿔야 합니다;
# 그다음 이것을 w에 맞춰 브로드캐스팅한뒤 결과물로 shape가 (3,2)인 행렬을 얻습니다,
# 이 행렬은 v와 w 외적의 결과입니다:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print np.reshape(v, (3, 1)) * w

# 벡터를 행렬의 각 행에 더하기
x = np.array([[1,2,3], [4,5,6]])
# x는 shape가 (2, 3)이고 v는 shape가 (3,)이므로 이 둘을 브로드캐스팅하면 shape가 (2, 3)인
# 아래와 같은 행렬이 나옵니다:
# [[2 4 6]
#  [5 7 9]]
print x + v

# 벡터를 행렬의 각 행에 더하기
# x는 shape가 (2, 3)이고 w는 shape가 (2,)입니다.
# x의 전치행렬은 shape가 (3,2)이며 이는 w와 브로드캐스팅이 가능하고 결과로 shape가 (3,2)인 행렬이 생깁니다;
# 이 행렬을 전치하면 shape가 (2,3)인 행렬이 나오며
# 이는 행렬 x의 각 열에 벡터 w을 더한 결과와 동일합니다.
# 아래의 행렬입니다:
# [[ 5  6  7]
#  [ 9 10 11]]
print (x.T + w).T
# 다른 방법은 w를 shape가 (2,1)인 열벡터로 변환하는 것입니다;
# 그런 다음 이를 바로 x에 브로드캐스팅해 더하면
# 동일한 결과가 나옵니다.
print x + np.reshape(w, (2, 1))

# 행렬의 스칼라배:
# x 의 shape는 (2, 3)입니다. Numpy는 스칼라를 shape가 ()인 배열로 취급합니다;
# 그렇기에 스칼라 값은 (2,3) shape로 브로드캐스트 될 수 있고,
# 아래와 같은 결과를 만들어 냅니다:
# [[ 2  4  6]
#  [ 8 10 12]]
print x * 2
~~~

브로드캐스팅은 보통 코드를 간결하고 빠르게 해줍니다, 그러니 가능한 많이 사용하세요.

### Numpy Documentation
이 문서는 여러분이 numpy에 대해 알아야할 많은 중요한 사항들을 다루지만 완벽하진 않습니다.
numpy에 관한 더 많은 사항은 [numpy 레퍼런스](http://docs.scipy.org/doc/numpy/reference/)를 참조하세요.

<a name='scipy'></a>

## SciPy

Numpy는 고성능의 다차원 배열 객체와 이를 다룰 도구를 제공합니다.
numpy를 바탕으로 만들어진 [SciPy](http://docs.scipy.org/doc/scipy/reference/)는,
numpy 배열을 다루는 많은 함수를 제공하며 다양한 과학, 공학분야에서 유용하게 사용됩니다.

SciPy에 익숙해지는 최고의 방법은 [SciPy 공식 문서](http://docs.scipy.org/doc/scipy/reference/index.html)를 보는 것입니다.
이 문서에서는 scipy중 cs231n 수업에서 유용하게 쓰일 일부분만을 소개할것입니다.

<a name='scipy-image'></a>

### 이미지 작업
SciPy는 이미지를 다룰 기본적인 함수들을 제공합니다.
예를들자면, 디스크에 저장된 이미지를 numpy 배열로 읽어 들이는 함수가 있으며,
numpy 배열을 디스크에 이미지로 저장하는 함수도 있고, 이미지의 크기를 바꾸는 함수도 있습니다.
이 함수들의 간단한 사용 예시입니다:

~~~python
from scipy.misc import imread, imsave, imresize

# JPEG 이미지를 numpy 배열로 읽어들이기
img = imread('assets/cat.jpg')
print img.dtype, img.shape  # 출력 "uint8 (400, 248, 3)"

# 각각의 색깔 채널을 다른 상수값으로 스칼라배함으로써
# 이미지의 색을 변화시킬 수 있습니다.
# 이미지의 shape는 (400, 248, 3)입니다;
# 여기에 shape가 (3,)인 배열 [1, 0.95, 0.9]를 곱합니다;
# numpy 브로드캐스팅에 의해 이 배열이 곱해지며 붉은색 채널은 변하지 않으며,
# 초록색, 파란색 채널에는 각각 0.95, 0.9가 곱해집니다
img_tinted = img * [1, 0.95, 0.9]

# 색변경 이미지를 300x300픽셀로 크기 조절.
img_tinted = imresize(img_tinted, (300, 300))

# 색변경 이미지를 디스크에 기록하기
imsave('assets/cat_tinted.jpg', img_tinted)
~~~

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/cat.jpg'>
  <img src='{{site.baseurl}}/assets/cat_tinted.jpg'>
  <div class='figcaption'>
    Left: 원본 이미지.
    Right: 색변경 & 크기변경 이미지.
  </div>
</div>

<a name='scipy-matlab'></a>

### MATLAB 파일
`scipy.io.loadmat` 와 `scipy.io.savemat`함수를 통해
matlab 파일을 읽고 쓸 수 있습니다.
[문서](http://docs.scipy.org/doc/scipy/reference/io.html)를 참조하세요.

<a name='scipy-dist'></a>

### 두 점 사이의 거리
SciPy에는 점들간의 거리를 계산하기 위한 유용한 함수들이 정의되어 있습니다.

`scipy.spatial.distance.pdist`함수는 주어진 점들 사이의 모든 거리를 계산합니다:

~~~python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# 각 행이 2차원 공간에서의 한 점을 의미하는 행렬을 생성:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print x

# x가 나타내는 모든 점 사이의 유클리디안 거리를 계산.
# d[i, j]는 x[i, :]와 x[j, :]사이의 유클리디안 거리를 의미하며,   
# d는 아래의 행렬입니다:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print d
~~~
이 함수에 대한 자세한 사항은 [pidst 공식 문서](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)를 참조하세요.

`scipy.spatial.distance.cdist`도 위와 유사하게 점들 사이의 거리를 계산합니다. 자세한 사항은 [cdist 공식 문서](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)를 참조하세요.

<a name='matplotlib'></a>

## Matplotlib
[Matplotlib](http://matplotlib.org/)는 plotting 라이브러리입니다.
이번에는 MATLAB의 plotting 시스템과 유사한 기능을 제공하는
`matplotlib.pyplot` 모듈에 관한 간략한 소개가 있겠습니다.,

<a name='matplotlib-plot'></a>

### Plotting
matplotlib에서 가장 중요한 함수는 2차원 데이터를 그릴 수 있게 해주는 `plot`입니다.
여기 간단한 예시가 있습니다:

~~~python
import numpy as np
import matplotlib.pyplot as plt

# 사인과 코사인 곡선의 x,y 좌표를 계산
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# matplotlib를 이용해 점들을 그리기
plt.plot(x, y)
plt.show()  # 그래프를 나타나게 하기 위해선 plt.show()함수를 호출해야만 합니다.
~~~

이 코드를 실행하면 아래의 그래프가 생성됩니다:

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/sine.png'>
</div>

약간의 몇 가지 추가적인 작업을 통해 여러 개의 그래프와 제목, 범주, 축 이름을 한 번에 쉽게 나타낼 수 있습니다:

~~~python
import numpy as np
import matplotlib.pyplot as plt

# 사인과 코사인 곡선의 x,y 좌표를 계산
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# matplotlib를 이용해 점들을 그리기
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
~~~
<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/sine_cosine.png'>
</div>

`plot`함수에 관한 더 많은 내용은 [문서](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)를 참조하세요.

<a name='matplotlib-subplots'></a>

### Subplots

'subplot'함수를 통해 다른 내용도 동일한 그림 위에 나타낼 수 있습니다.
여기 간단한 예시가 있습니다:

~~~python
import numpy as np
import matplotlib.pyplot as plt

# 사인과 코사인 곡선의 x,y 좌표를 계산
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 높이가 2이고 너비가 1인 subplot 구획을 설정하고,
# 첫 번째 구획을 활성화.
plt.subplot(2, 1, 1)

# 첫 번째 그리기
plt.plot(x, y_sin)
plt.title('Sine')

# 두 번째 subplot 구획을 활성화 하고 그리기
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# 그림 보이기.
plt.show()
~~~

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/sine_cosine_subplot.png'>
</div>

`subplot`함수에 관한 더 많은 내용은
[문서](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot)를 참조하세요.

<a name='matplotlib-images'></a>

### 이미지
`imshow`함수를 사용해 이미지를 나타낼 수 있습니다. 여기 예시가 있습니다:

~~~python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# 원본 이미지 나타내기
plt.subplot(1, 2, 1)
plt.imshow(img)

# 색변화된 이미지 나타내기
plt.subplot(1, 2, 2)

# imshow를 이용하며 주의할 점은 데이터의 자료형이
# uint8이 아니라면 이상한 결과를 보여줄 수도 있다는 것입니다.
# 그러므로 이미지를 나타내기 전에 명시적으로 자료형을 uint8로 형변환 해줍니다.

plt.imshow(np.uint8(img_tinted))
plt.show()
~~~

<div class='fig figcenter fighighlight'>
  <img src='{{site.baseurl}}/assets/cat_tinted_imshow.png'>
</div>

---
<p style="text-align:right"><b>
번역: 강상훈 <a href="https://github.com/sanghunka" style="color:black">(sanghunka)</a>
</b></p>
