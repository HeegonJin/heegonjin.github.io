---
title: "GPNN"
category:
    Deep Learning
tag:
    Paper Review
author_profile: true
toc : true
use_math : true
comments: true
---


# Learning Human-Object Interactions by Graph Parsing Neural Networks

# Abstract

논문에서 해결하고자 하는 테스크(task)는, 이미지와 비디오에 대해서 사람-객체 상호작용(Human-object Interactions)을 감지하고(detecting) 인식(recognition) 하는 것이다.

이를 위해 논문에서는 Graph Parsing Neural Network라는 프레임워크를 제안하는데, 이는
structured knowledge(structured-rnn 논문 참고)를 통합시키면서 end-to-end로 미분가능하다. 
주어진 scene에 대하여, GPNN은 parse graph를 inference하는데, 구체적으로는 다음과 같다.
1)  HOI graph structure을 표현하는 인접행렬(adjacency matrix)

2) 각 노드들의 라벨

Message Passing inference framework(Message passing Neural Network 참고)를 이용하여, 
GPNN은 iteratively 하게 인접행렬과 노드 라벨을 계산하게 된다.

이 모델은 3가지의 HOI detection 벤치마크를 이미지와 비디오에 대해서 성능평가 하였다

HICO-DET, V-COCO, CAD-120 dataset에서  SOTA를 달성했고, 이를 통해 GPNN이 큰 데이터셋과 
spatial-temporal(시공간적인, 비디오 데이터에 대해서) 세팅에서도 적용된다.

# Introduction

Human-object inteeraction (HOI) understanding 문제는 "자전거를 타다" "자전거를 닦는다"와 같이사람과 객체간의 관계를 추론하는 것이다. 개체 각각에 대한 traditional visual recognition 방법들        (pose estimation, action recognition, object detection) 을 넘어, HOI를 인식하는 것은 이미지에 대한 깊은 맥락적인 이해가 필요하다. 

최근 딥러닝 방법이 위에 언급한 instance recognition의 개별 task에 대해서 인상적인 발전을 보여주고 있지만, HOI recognition과 관련해서는 알려진 방법들이 적은 상태이다. 이러한 이유는 HOI가 단순한 '인식'의 문제가 아니라 사람과 객체, 그리고 그들의 복잡한 관계에 대한 정보를 통합하여 '추론(reasoning)'이 필요하기 때문이다.

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled.png)

---

GPNN은 이미지와 비디오에 모두 적용되는, generic한 특성을 가지는 HOI representation을 제공한다. 그래프 모델과 신경망 모델을 통합함에 따라, GPNN은 반복적으로 그래프 구조와 메시지 전달을 학습 및 추론할 수 있다. 위 그림에서 (vii) final parse grap는 주어진 scene에 대해서 그래프 구조(사람과 칼 사이의 링크), 그리고 노드 라벨(햝다)로 설명한다. 그림에서 두꺼운 엣지일 수록 그 노드들 간에 더 강한 정보흐름이 일어난다는 것을 의미한다.

이 논문의 컨트리뷰션은 크게 세가지로 볼 수 있다.

먼저, structural knowledge (graph model)과 DNN을 통합하여 end-to-end로 학습가능하다는 것

둘쨰, 잘 정의된 모듈러 함수들을 통해, 그래프 구조 추론과 메시지 전달을 동시에 수행한다는 것

셋째, 다양한 큰 데이터셋들에 대해 scalable하며, 이미지, 비디오 모두에 적용가능한 generic representation이 가능하다는 것이다.

# GPNN for HOI

## Formulation

HOI understanding을 위해서, 사람과 객체는 노드들로 표현되며, 그들의 관계는 엣지로 표현된다.

사람과 객체간에 가능한 모든 관계를 포함하는 완전한 HOI 그래프가 주어졌을 때, 
우리는 이 그래프에 대해서 의미있는 엣지들을 유지하고 노드에 라벨을 할당하여 Parse Graph를 
자동적으로 추론하는 것을 목표로 한다.

### Illustration of the forward pass of GPNN

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%201.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%201.png)

GPNN은 노드와 엣지 피처를 입력으로 받고 message passing 양식으로 Parse graph를 추론한다.

parse graph의 구조는 soft adjacency matrix로 주어진다. 여기서 soft adjacency matrix는 피처(히든 노드 스테이트) 기반 *link function*으로 계산된다. 위 그림에서 더 adjacency matrix가 더 어두운 색을 띌 수록, 연결성이 더 강하다는 것을 의미한다. 그리고 나서 *message functions가*  각 노드들의 incoming message들을 다른 노드들로부터의 메시지들의 weighted sum으로 계산한다.

그림에서 두꺼운 엣지는 더 큰 정보의 흐름을 의미한다. *update functions* 는 (내재된) 히든 스테이트의 각 노드들의 상태를 업데이트한다. 위 프로세스는 여러 번 반복되고, 그래프 구조 계산과 메시지 전달을 함께 학습한다. 마지막으로, 각 노드에 대해서, *readout function*이 HOI action이나 객체의 라벨을 히든 노드 스테이트로부터 출력한다.

G = (V, E, Y) 를 완전한 HOI 그래프라고 하자. parse graph g=(V_g, E_g, Y_g) 는 G의 subgraph이다. 

우리는 주어진 노드 피처 $\gamma$_v 와 엣지 피처 $\gamma$_E 를 이용하여 데이터의 확률분포 p를 따르는 데이터를 가장 잘 설명하는 최적의 parse graph g*를 추론해야 한다.

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%202.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%202.png)

위 식에서 앞단은 parse graph의 노드들에 대한 labeling probability를 의미하고, 뒷단은 그래프 구조에 대한 확률을 의미한다. GPNN의 네가지 함수들은 각각 개별적인 모듈로서 GPNN의 forward pass를 수행한다. 위에서 언급한 link function, message function, update function, readout function 이다.

위 그림을 다시 보면, 링크 함수는 엣지 피처(Feature Matrix F)를 입력받아 노드의 연결관계를 추론한다. 그 결과 soft adjacency matrix (Adjacency Matrix A)가 만들어지고, 이것은 노드들 간의   메시지 패싱의 가중치들로 사용된다. 각 노드의 들어오는 메시지들은 메시지 함수로 요약되고, 히든 임베딩 상태의 노드들은 이 메시지들로부터 업데이트 함수에 의해 갱신된다.  

### Link Function

링크 함수 L은 노드 피쳐 $\gamma$_v 와 엣지 피처 $\gamma$_E 를 입력으로 받고 adjacency matrix A를 출력한다.

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%203.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%203.png)

위 그림에서, A_vw는 인접행렬 A의 (v, w) entry를 의미하고, $\gamma$_vw 는 v와 w를 잇는 엣지 피처를 의미한다. 이 인접행렬은 parse graph g 의 형태를 근사화하고, parse graph에서 message propagation을 할때, 이 soft adjacency matrix 가 엣지 사이에 전달되는 정보량을 조절한다.

### Message and Update Function

학습된 그래프 구조를 바탕으로, 메시지 전달 알고리즘을 이용해 노드 라벨을 추론한다. 
**[Belief Propagation](https://tastyprogramming.tistory.com/7)** 과정에서, 히든 스테이트의 노드들은 다른 노드들과 상호작용하며 반복적으로 갱신된다.  

---

**[Belief Propagation](https://tastyprogramming.tistory.com/7)** : 베이시안 네트워크 상의 모든 확률 변수들의 사후 분포 계산은 NP-hard이기 때문에, 근사해를 추정하는 기법 중에 하나이다. 그래프 모델에서 관측된 일부 확률변수의 분포(Evidence)가 주어졌을 때, 그로부터 직간접적으로 영향을 받는 모든 관측되지 않는 확률변수의 분포를 추정하는 것.

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%204.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%204.png)

---

각 반복 단계 s 마다, 두 함수들은 다음을 계산하게 된다.

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%205.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%205.png)

위에 m_v는 노드 v에 대해서 들어오는 메시지 주요 정보, h_v는 노드 v의 히든 스테이트를 의미한다.

(첫 이터레이션 s=0 에서 히든 스테이트 h는 노드 피쳐로 초기화한다)

노드간 연결관계를 표현하는 A는 parse graph에서의 노드간의 정보 흐름을 일어나도록 한다.
(즉, 완전 그래프로부터 parse graph를 형성한다.)

U는 업데이트 함수로서 히든 노드 스테이트들을 들어오는 메시지에 따라 갱신한다.

이 메시지 전달 페이즈는 수렴할 때까지 S번 반복하게 된다.

### Readout Function

마지막으로, readout function은 각 노드들의 히든 스테이트을 입력으로 받고 라벨을 출력한다.

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%206.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%206.png)

readout function R이 노드 v의 히든 스테이트 h_v를 activation하여 출력 y를 계산한다.

### Iterative Parsing

A를 학습 시작 단계에서만 학습하는 것이 아니라, A를 업데이트 된 노드정보와 엣지피처를 이용한다. 따라서 

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%203.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%203.png)

위 식을 아래와 같이 iterations s 에서 일반화할 수 있다.

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%207.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%207.png)

![Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%208.png](Learning%20Human-Object%20Interactions%20by%20Graph%20Parsin%20f5fc95ac92e547618cd581710719b85b/Untitled%208.png)

이를 통해, 그래프 구조와 메시지 업데이트는 동시에 이루어지고, 통일된 프레임워크로 반복 학습된다.