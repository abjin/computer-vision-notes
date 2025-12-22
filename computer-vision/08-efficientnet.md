# EfficientNet이란

- 제안: Google
- 목표: 더 나은 정확도 + 더 적은 파라미터 수

- 핵심아이디어: 복합 스케일링
  - 깊이, 너비, 해상도를 수동으로 조절하는 대신, 모든 것을 함께 이상적인 비율로 스케일링
  - 복합 스케일링을 통해 가장 성능을 높이는 방법으로 모델을 키움


- 스케일링 공식
  - 깊이 (Depth): $d=\alpha^{\phi}$
  - 너비 (Width): $w=\beta^{\phi}$
  - 해상도 (Resolution): $r=\gamma^{\phi}$

- 목표
  - $\max_{d, w, r} \text{Accuracy}(\mathcal{N}(d, w, r))$
- 제약 조건
  - $\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2$  ($\phi$ 를 1 씩 늘릴때마다 계산 비용약 약 2배씩 증가)
  - 제약 조건: $\text{Memory}(\mathcal{N}) \le \text{target memory}$, $\text{FLOPs}(\mathcal{N}) \le \text{target flops}$


# EfficientNet 아키텍쳐 개요

- MBConv
- SE block
- Swish

# MBConv 구조


- 구조 : 역병목 구조
  - **Expansion**: 1x1 합성곱을 사용해서 채널 크기 증가 -> 특징 표현 능력 보충 (1x1x채널크기)
  - **깊이별 합성곱**(3x3): 공간 특징 추출
  - **SE 블록**: 피처맵 재조정. 정보 풍부한 채널 강화, 덜 유용한 채널 억제
  - **투영**: 1x1 합성곱을 사용 채널 수 축소
  - **잔차 연결**(옵션): 효율적으로 기울기가 전파 되도록 함


# MBConv - Squeeze and Excitation

**동기**
- 깊이별 합성곱은 채널 별로 독립적으로 처리되며, 채널간 의존성을 고려하지 않음, 어떤 채널이 중요한지 모름

**아이디어**
- SE Block 은 어떤 채널이 중요한지 학습 가능하도록함.

**SE Block**
1. **Squeeze**
  - GAP 를 사용하여 HxWxC 크기의 특징맵을 1x1xC 의 벡터로 만듭니다.
  - 공간적인 위치 정보는 무시하고, 이 채널이 전체적으로 얼마나 활성화 됐는지 봄
2. **Excitation**
  - 입력: GAP 결과 값을 입력 받아 채널 중요도를 계산합니다.
  - FC - ReLU - FC: 채널간 중요도 즉 무엇을 강조해야하는지 학습
  - Sigmoid: 네트워크가 채널에 얼마나 집중할지 0~1 사이의 값으로 반환
3. **Scale**
  - Excitation 에서 얻은 가중치를 입력 특징맵에 곱하여 재조정된 특징맵을 만듬.

**Squeeze**
- 각 채널은 해당 채널의 활성화 값을 나타내는 스칼라 값으로 압축
$$
z_{c} = F_{sq}(u_{c}) = \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} u_{c}(i,j) \Rightarrow z \in \mathbb{R}^{1 \times 1 \times C}
$$

**Excitation**
- 계산을 줄이기 위해 병목 구조를 가진 두개의 FC 층 사용
$$s = F_{ex}(z, W) = \sigma(g(z, W)) = \sigma(W_{2} \cdot \delta(W_{1} \cdot z))$$
- $z$: 입력 벡터
- $\delta$: ReLU (FC1 이후)
- $\sigma$: Sigmoid (FC2 이후)
- FC1 ($W_{1}$**):** $W_{1} \in \mathbb{R}^{\frac{C}{r} \times C}$
  - 채널 차원을 $C \rightarrow C/r$로 축소 (Reduction).
  - $r$: 감소 비율(reduction ratio), 하이퍼파라미터 (예: 16, 4).
- FC2 ($W_{2}$**):** $W_{2} \in \mathbb{R}^{C \times \frac{C}{r}}$
  - 채널 차원을 다시 $C/r \rightarrow C$로 확장 (Expansion).

**Scale**
- 정보가 유익한 층은 강조하고 덜 유용한 층은 억제함
$$
\tilde{X}_{c} = F_{scale}(u_{c}, s_{c}) = s_{c} \cdot u_{c}
$$

# Swish 활성화 함수


- 공식
$$
 \text{swish}(x) = x \cdot \sigma(\beta x)
$$
- 특징
  - 비단조적
  - 부드러움
  - 뉴런이 죽는 문제를 방지
    - (ReLU 는 음수일때 기울기 0이어서 발생)


- ReLU vs Swish
  - ReLU: $\max(0, x)$ / 부드럽지 않음 / 단조적임 / 대부분의 CNN에서 사용됨.
  - Swish: $x \cdot \sigma(\beta x)$ / 부드러움 / 비단조적임 / EfficientNet에서 사용됨.

- 활성화맵 비교
  - ReLU는 날카로운 경계를 보인다. Swish는 부드러운 경계를 보인다.
  - Swish 함수가 옵티마이저가 부드러운 손실 표면을 쉽게 따라가게하고. 로컬 미니멈을 피하는데 도움


# MobileNet vs EfficientNet

| **특징 (Feature)**            | **MobileNet**                        | **EfficientNet**                                  |
| --------------------------- | ------------------------------------ | ------------------------------------------------- |
| **설계 전략**                   | 수동 + 휴리스틱 (Manual + Heuristic)       | 신경망 아키텍처 탐색(NAS) + 복합 스케일링 (Compound Scaling)     |
| **구성 요소 (Building Blocks)** | Depthwise Separable Conv             | MBConv + SE + Swish                               |
| **정확도 (Accuracy)**          | 중간 수준 (~70%)                         | 높음 (최대 84.4%)                                     |
| **파라미터 / FLOPs**            | 매우 적음 (Extremely Low)                | 효율적임 + 높은 정확도 (Efficient w/ High Accuracy)        |
| **사용 사례 (Use Cases)**       | 실시간 모바일 비전 (Real-time Mobile Vision) | 범용 고정확도 비전 (General-purpose High-Accuracy Vision) |
