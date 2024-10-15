---
title: "[자료구조]Tree & Heap(힙)"

categories: 
  - DataStructure

toc: true
toc_sticky: true

date: 2022-12-09
last_modified_at: 2022-12-09
---

## 1. Tree(트리)
### 1) Tree(트리)란?
연결리스트나 힙, 스택등의 앞서 공부한 자료구조들은 모두 선형 자료구조(Linear Data Structure)이다. 

반면, 트리는 <span style = "color:green">부모(parents)-자식(child) 관계를 **계층적**으로 표현</span>한 **비선형 자료구조(Nonlinear Data Structure)**이다.


<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206685637-e37173d4-ed51-4931-b595-498f496f5f4d.png">
</p>

### 2) Tree 구조에서 쓰이는 용어

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/206690396-257ca115-c14e-46df-af4f-75d745d1ef43.png">
</p>

### 3) 이진 트리(Binary Tree)
이진 트리(Binary Tree)란, 모든 노드의 **자식 노드가 2개**를 넘지 않는 트리를 말한다.  
대부분의 실제 사용되는 트리 구조는 이진 트리 구조이다.

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/111734605/206693671-eb9ea74d-83bd-4e53-a896-685cc417b82d.png">
</p>

### 4) 이진 트리의 표현법
이진 트리를 표현하는 방법으로는 크게 **리스트**를 이용하는 방법과 **연결 리스트를 클래스를 정의**하는 방법이 있다.

#### (1) 하나의 리스트로 표현  

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206695097-ab5dc200-e65e-454b-acd4-9ba5fab26e7c.png">
</p>  
**레벨 0부터** 차례대로, **왼쪽에서 오른쪽 순서**로 작성한다. 자식노드가 없는 경우는 None으로 작성한다.  

`장점`
이렇게 트리를 표현하면, `상수시간의 연산`으로 자식노드와 부모노드를 찾을 수 있다.

왜냐하면 현재 노드의 인덱스번호를 알고 있다면, 자식노드와 부모노드의 인덱스번호를 계산할 수 있기 때문이다.
* 자식노드
  - 왼쪽 자식노드: $$(Index)  \times  2 + 1$$
  - 오른쪽 자식노드: $$(Index)  \times  2 + 2$$ 
* 부모 노드  
  - $$\frac{(Index) -1}{2}$$의 **몫**
  - (Index - 1)//2

```
부모 노드의 인덱스가 0 (a 노드)
이때 왼쪽 자식노드는 A[0*2 + 1] = A[1]이다. (b노드)

부모 노드의 인덱스가 3일때 (노드 c)
오른쪽 자식노드는 A[2*2+2] = A[6]이다. (f노드)
```

`단점`  
불필요한 `메모리 낭비`가 발생한다. 
- 연산 시간 ⇋ 메모리  `Trade-off`

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/206702907-46470b0e-a3a0-4287-860d-67a68536d3a9.png">
</p>  

노드가 실제로는 비어있지만, 하나의 리스트로 표현해야 하기에, 그 빈 노드에 `None`을 채워넣게 되고  
그에따라 차지하는 메모리는 증가한다.

#### (2) 리스트를 중복해서 표현

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206698717-8d1e749c-ae71-443b-a9e1-58507cadd18c.png">
</p>  

[루트, [루트의 **왼쪽** 부트리], [루트의 오른쪽 부트리]]형식으로 재귀적으로 정의

#### (3) 연결 리스트로 표현

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206700248-d884b578-da17-4ef8-a7dc-fb15ca0c4369.png">
</p>  
각 노드가 key, parent, left, right 에 대한 정보를 가진다. 단 루트 노드는 제외다.(루트 노드는 부모 노드가 없다.)


## 2. Heap(힙)
### 1) 힙의 개념
모양 성질과 힙 성질을 만족하는 리스트에 저장된 값의 시퀀스이다.

```
모양 성질: 
        1. 마지막 레벨을 제외한 각 레벨의 노드가 모두 채워져 있어야 한다.
        2. 마지막 레벨에선 노드들이 왼쪽부터 채워져야한다.

힙 성질:
        1. 루트 노드를 제외한 모든 노드에 저장된 값(key)은 자신의 부모노드의 
           보다 크면 안된다.
```
 <span style = "color:green">보통 리스트의 데이터는 힙성질, 모양성질을 갖춘 데이터가 주어지지 않는다.</span>  
 따라서 힙 성질을 갖춘 데이터가 되도록 리스트의 데이터들을 **재정렬**해 주어야 한다.
 - makeheep() 함수 이용

### 2) 힙의 그래프로 만드는 과정

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/206703693-6b12933d-f927-4e57-8a86-fc03ab51a72c.png">
</p> 

* <span style = "color:green">**힙 성질(Heap Property): 모든 부모노드의 key 값은 자식노드의 key값보다 작지 않다.**</span>

<p align="center">
<img width="1000" alt="1" src="https://user-images.githubusercontent.com/111734605/206704155-dbc771c9-bf9e-488d-a5cb-459c3bf0bd03.png">
</p> 

### 3) 힙(Heap)에서 쓰이는 연산
- `make_heap()`
- `heapify_down`
- `heapify_up`
- `insert`
- `finde_max`
- `delete_max`

#### (1) Heap 클래스

```python
class Heap:
    def __init__(self, L): #L은 리스트
        self.A = L
        self.make_heap()

    def __str__(self):
        return str(self.A)
```

#### (2) make_heap, heapify_down 함수  
* **make-heap : Heap 성질을 만족하도록 리스트를 재배치**
  - <span style = "color:green">heapify-dwon 이라는 연산을 반복 수행</span>해야함

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206708227-42bf0fae-602f-47bf-8d62-3e6d6518f186.png">
</p> 

* List의 마지막에 저장된 값(None 제외하고)은 힙의 가장 마지막 레벨에 가장 오른쪽
 에 저장됨.
* 이걸 밑으로 내려가면서, 자식 노드들로 내려가면서 힙 성질을 만족 시키도록 움직임.
* 그 다음 값도 반복
* 참고로 리프 노드는 볼 필요 없음. 자식 노드가 없기에 힙 성질 만족함.
* A에서 11 -> 12 -> 3 -> 15 -> 10 까지는 리프 노드이므로 skip
* 그 다음이 1인데, A[3] = 임, why? 루트 노드 = 0, 루트 노드 2배 + 1 =1, A[1] = 8
* A[1]의 왼쪽 자식이므로 1 = A[1*2 + 1] = A[3] 
* A[3]의 왼쪽, 오른쪽 자식 노드의 인덱스를 알아야 함
* 왼쪽 자식 = A[7], 오른쪽 자식 = A[8] 
* 근데 자식 노드가 1보다 크니깐 바꿔야됨. 뭐랑? 12 !!

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206709018-7137d6a8-1032-45d5-b06f-9f5e9e63be55.png">
</p> 

- k = Heap 리스트의 인덱스  
- n = Heap 원소의 개수   

```python
H = [2,8,6,1,10,15,3,12,11]
def makeheap(self):
    n = len(self) 
    #데이터의 각 요소에 대해서, 마지막 요소부터 heapify_down함수를 실행
    for k in range(n-1, -1, -1):
        self.heapify_down(k,n) #매개변수로 대상 노드의 인덱스와 전체 데이터의 길이가 주어진다.

def heapify_down(self, k,n):
    #두 자식노드의 값이 자기자신보다 작을거나 리프노드에 도달할때까지 반복
    while 2*k + 1 < n:
        #왼쪽 자식노드와 오른쪽 자식노드의 인덱스번호 계산
        L, R = 2 * k + 1, 2 * k + 2
        # 부모노드와 자식노드들 중 가장 큰 값의 인덱스 찾기
        if self.A[L] > self.A[k]:
            m = L
        else:
            m = k

        if R < n and self.A[R] > self.A[m]:
            m = R
        
        if m != k: #현재 k가 heap 성질 위배하는 경우
            self.A[k], self.A[m] = self.A[m], self.A[k]
            k = m #k에 m값을 줌으로써 m인덱스를 부모노드로하는 heap성질 검증 실행한다.
        else:
            break #현재 노드가 heap 성질을 만족한다면 break건다. 
            왜냐하면 makeheap()에 의해 k가 작아지면서 윗 노드에서 heap성질을 위반하면 
            아래 노드들에 대해 알아서 검증해주기 때문이다.
```

* make_heap & heapify_down의 수행 시간

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206711881-3e4d3e08-4b1b-44c9-851a-4be70083554e.png">
</p> 

* make_heap
  - k 번의 for 루프를 도는데, k에 대해서 1번씩 총 n 번 부르게 됨.
  - O(n × t) = O(n × h)
  - t는 heapify_down의 수행 시간이다.
* hepify_down
  - 루트 노드에서 밑으로 내려가면서 최악의 경우 리프 노드까지 도달
  - 최악의 경우를 예로
    - 1) A[k], A[L], A[R] 세 개 비교해야하고 
    - 2) A[k]와 m을 비교해야하고 
    - 3) A[k]와 m 을 swap해야함
  - 하지만 이 세 가지 연산 모두 상수시간내에 됨
  - 최악의 경우는 root node - leaf node 까지, Height !!
  - Big $$O(h)$$, h = Height

#### (3) insert(삽입 연산)

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206713261-04de9695-8e7e-44dd-85a9-c62163daae68.png">
</p>

* insert(14) = A.append(14) -> heap 성질 불만족
  - Step 1) 부모 노드랑 비교해보니 부모 노드 키 값이 더 작다
  - Step 2) 부모 노드랑 자리 change
  - Step 3) 다시 바뀐 위치에서 부모 노드랑 키 값 비교, 부모 노드가 더 작음
  - Step 4) 부모 노드랑 자리 change
  - Step 5) 부모 노드랑 키 값 비교, 부모 노드가 더 큼 -> heap 성질 만족  

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/111734605/206713749-c6083d96-9bc6-4358-a314-876c502ba39a.png">
</p>

heapifyup함수는 insert연산을 위한 함수이다. insert연산은 힙 리스트의 마지막에 값을 추가하고, 이 값이 힙 성질을 가지도록  
부모노드들을 타고 올라가면서 정렬을해야한다. 따라서 시간복잡도는 **$$O(logN)$$**이다.

```python
def heapify_up(self, k):
    #k가 양수이고, 현재노드의 부모노드의 값 < 현재노드일때 실행
    while k > 0 and self.A[(k-1)//2] < self.A[k]:
        #부모노드와 현재노드의 위치를 바꾸고
        self.A[k], self.A[(k-1)//2] = self.A[(k-1)//2], self.A[k]
        #k에 부모노드의 인덱스번호를 주고 while문 반복
        k = (k-1) // 2
    
def insert(self, key):
    #힙 리스트의 마지막에 값을 추가한다.
    self.A.append(key)
    #현재 리스트가 4개 -> len(A) = 4, indexnum = 3 이므로 -1해준다.
    self.heapify_up(len(self.A) - 1)
```
#### (4) find_max & delete_max

<p align="center">
<img width="1200" alt="1" src="https://user-images.githubusercontent.com/111734605/206717073-169e4ec6-8ba7-4503-b784-0713aea3955d.png">
</p>

* find_max는 무조건 루트 노드르르 리턴한다.

* delete_max는 현재 힙 리스트의 값 중 가장 큰 값을 삭제하고 리턴한다.  
  삭제하고 리턴 -> pop연산을 해야하므로 A[0]을 A[n-1]의 값으로 대체하고 pop,  
  이후 0번 인덱스 노드부터 재배치한다-> heapify_down(0, len(A))

```python 
def delete_max(self):
    if len(self.A) == 0:
        return None
    #리턴할 max값 저장
    key = self.A[0]
    #대체
    self.A[0], self.A[len(self.A) - 1] = self.A[len(self.A) - 1],self.A[0]
    #삭제
    self.A.pop() 
    self.heapify_down(0,len(self.A))
    return key
```
#### (5) heap_sort  
정렬알고리즘이다. 주어진 데이터를 make_heap으로 힙 정렬을 한다. heap정렬의 특징은 가장 큰 값이 root노드가 된다는 것이므로, `힙 리스트의 첫값을 마지막값과 대체, 그리고 마지막 값을 제외한 리스트에 대해 heapify_down으로 정렬` 을 반복한다.  

* step 1) n개의 숫자를 입력으로 받음
* step 2) make_heap 을 불러서 heap으로 구성 (O(nlogn))
* step 3) n 번을 delete_max를 하면 됨.
  (단, 마지막 값은 pop하지말고 진행)

* make_heap => O(N)  
* heapify_down => O(logN)   

따라서  
* **heap_sort => O(NlogN)**  

```python
def heap_sort(self):
    n = len(self.A)
    #현재 리스트의 길이의 인덱스번호를 큰값부터 -1부여
    for k in range(len(self.A) - 1, -1, -1)
        #대체
        self.A[0], self.A[k] = self.A[k],self.A[0]
        
        #뒤로 대체된 값을 제외한 값들에 대해 heap정렬
        n = n-1
        heapify_down(0,n)
```

### 4) 연산시간 정리

![image](https://user-images.githubusercontent.com/111734605/206718509-2bff6adb-aa17-4195-b074-d7d97f8afddf.png)

* search 연산은 왜 없어?
  - 찾는 값이 자식 노드의 왼쪽인지 오른쪽인지 구분 불가능
  - heap 은 search 에 부적합
  - 큰 값들을 찾거나 지우는 연산에 heap 을 쓰는 것이 좋음

* find_min, delete_min은 왜 없어?
  - 우리가 만든 heap은 max_heap임
  - 따라서 heap 성질을 바꿔서 부모 노드가 더 작은 값이 오게 만들면 됨
  - min_heap 은 두 함수를 구현할 수 있음

<p align="center">
<img width="1200" alt="1" src="https://user-images.githubusercontent.com/111734605/206719678-e91c8077-4693-4718-b1e0-54eb316e725d.png">
</p>

## Reference
[신찬수 교수님 강의 자료](https://www.youtube.com/c/ChanSuShin/featured)  
