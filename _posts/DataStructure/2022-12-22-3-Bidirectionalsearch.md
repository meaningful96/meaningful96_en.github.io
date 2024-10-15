---
title: "[자료구조]Bidirectional Search(양방향 탐색)"

categories: 
  - DataStructure

toc: true
toc_sticky: true

date: 2022-12-22
last_modified_at: 2022-12-22
---

##  1. 양방향 탐색이란?
### 1) 기존의 탐색 방식(DFS,BFS)
기존에 공부했던 탐색 방법은 두가지이다. 너비 우선 탐색(BFS)과 깊이 우선 탐색 방법(DFS)이다. 

- BFS(Breadth-First-Search)는 너비 우선 탐색 방법이다. 
  1. 루트 노드 혹은 다른 노드에서 시작해서 인접한 노드를 먼저 탐색하는 방법이다.
  2. 두 노드 사이의 최단 경로 혹은 임의의 경로를 찾고 싶을 때 사용한다.

- DFS(Depth-First Search)는 깊이 우선 탐색이다. 
  1. 루트 노드 혹은 다른 임의의 노드에서 시작해서 다은 분기(branch)로 넘어가기 저에 해당 분기를 완벽하게 탐색하는 방법입니다.
  2. 모든 노드를 방문하고자 하는 경우에 사용한다.

### 2) 양방향 탐색의 정의
두 방법 모드 일종의 source vertex에서 goal vertex방향으로 search하는 방법들이다. Bidirectional search는 이렇게 특정 방향으로의 탐색이 아닌 양방향 탐색을 하는 방법이다.
따라서 Bidirectional search에는 두 가지 매커니즘이 존재한다.
- Mechanism
  1. **Forward search from source/initial vertex toward goal vertex**
  2. **Backward search from goal/target vertex toward source vertex**

즉 시작 노드인 source 노드에서 최종 목표인 goal 노드방향으로 가는 걸 Forward Search, 그리고 그 반대 방향으로 탐색을 진행하는 것을 Backward Search라고 한다.
두 search가 동시에 진행되며 **교차(Intersecstion)**되는 순간 연산이 종료된다.




### 3) 양방향 탐색의 예시
```
Bidirectional search replaces single search graph(which is likely to grow exponentially) with two smaller sub graphs 
– one starting from initial vertex and other starting from goal vertex. The search terminates when two graphs intersect.
Just like A* algorithm, bidirectional search can be guided by a heuristic estimate of remaining distance from source to 
goal and vice versa for finding shortest path possible.
```
<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/111734605/208996458-1306ca6e-a645-4277-a89f-a05c969505df.png">
</p>
0 노드에서 14 노드까지 search 연산을 진행한다고 하자. 이 때, 0 노드가 source node가 되는 것이고, 14가 goal 노드가 되는 것이다. 두 연산이 동시에 진행될 때, 7 node(Vertex)
에서 교차하게되고, 이 때 search연산이 종료된다. 이를 통해 불필요한 exploration을 피한것을 볼 수 있다.

## 2. 양방향 탐색의 목적
### 1) Why bidirectional approach?
많은 경우, <span style = "color:green">양방향 탐색의 앞선 두 탐색 방법보다 빠르며, 불필요한 탐색 경로를 줄일 수 있다.</span>

트리를 생각해볼때, b가 branching factor이고 목표 vertex까지의 거리가 d인경우 **BFS와 DFS 탐색의 complexity**는 $$ O(bd) $$이다.

반면에, 양방향 탐색을 진행하는경우 하나의 operation을 진행하는 complexity는 $$O(\frac{bd}{2})$$이다. 따라서 전체 complexity는 $$O(\frac{bd}{2} + \frac{bd}{2} )$$이고,
이는 $$O(bd)$$보다 작다.

### 2) When to use bidirectional approach?
양방향 탐색은 탐색의 시작과 끝 지점이 명확하게 정의되어있고, 유니크한 경우에 사용하는 것이 적절하다. Branching factor는 두 operation(direction)이 정확하게 일치한다.

### 3) 성능 평가

- Completeness : Bidirectional search is complete if BFS is used in both searches.
- Optimality   : It is optimal if BFS is used for search and paths have uniform cost.
- Time and Space Complexity : Time and space complexity is $$O(\frac{bd}{2})$$


## Reference
[Bidirectional Search](https://www.geeksforgeeks.org/bidirectional-search/)
