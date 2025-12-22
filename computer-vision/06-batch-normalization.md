
# 배치 정규화
- BN 은 각 레이어의 입력을 정규화하여 훈련 내내 입력의 분포값을 안정적으로 유지되게 함

# 배치 정규화 동기
- 그라디언트 소실 폭주 문제
	- 문제: 느린 수렴, 높은 오류, 깊은 모델 훈련 불가
	- 원인: 큰 입력 변화가 작은 출력 변화로 이어짐 (Relu, tanh 포화함수이기 때문) -> 그라디언트 소실됨
	- 활성화 함수 변경, 신중한 가중치 초기화, 작은 학습률, **BN**
	- BN 이 근복적이 해결: 입력 분포가 일관되게 유지, 가중치 초기화 민감도 줄임, 훈련이 빠르고 안정적

- 내부 공변량 이동
	- 파라미터가 계속 학습됨에 따라 레이어에 대한 입력 분포가 계속 변함 
	- 내부 공변량: 이전 레어들로 인하여 입력 분포가 계속 변함
	- 이러한 분포의 이동은 훈련을 느리게하고 수렴을 어렵게함
	- BN 사용시: 입력 분포가 일관되어져 학습을 빠르고 안정적으로 함

# 배치 정규화 목표
- 그레디언트 소실/폭주 완화: 값들이 극단적으로 커지는 것을 방지
- 내부 공변량 이동 방지: 훈련중 입력 분포 계속 변하면 학습이 어렵고 수렴이 늦음
- 로컬 미니멈 문제 해결: 데이터 분포를 고르게 함으로써 손실 분포를 평탄하게 하여 로컬 미니멈을 피함

# 배치 적규화 공식

#### 공식

- **1. 평균과 분산 계산:**
	- $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$  
	- $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$  
- **2. 입력 정규화:**
	- $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$  
- **3. 스케일 및 시프트 적용:**
	- $y_i = \gamma \hat{x}_i + \beta$  

- $\epsilon$: 수치적 안정을 위한 작은 상수
- $\gamma$: 학습 가능한 스케일(scale) 파라미터
- $\beta$: 학습 가능한 시프트(shift) 파라미터


# 배치 정규화의 역전파
###### 1) $\displaystyle \frac{\partial \mathcal{L}}{\partial \sigma_B^2}$ 유도

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
= (x_i-\mu_B)\cdot(\sigma_B^2+\epsilon)^{-1/2}
$$
따라서
$$
\frac{\partial \hat{x}_i}{\partial \sigma_B^2}
= (x_i-\mu_B)\cdot\left(-\frac{1}{2}\right)(\sigma_B^2+\epsilon)^{-3/2}
= -\frac{1}{2} (x_i-\mu_B)(\sigma_B^2+\epsilon)^{-3/2}
$$
체인룰로 합치면
$$
\boxed{
\frac{\partial \mathcal{L}}{\partial \sigma_B^2}
= \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i}
\frac{\partial \hat{x}_i}{\partial \sigma_B^2}
= -\frac{1}{2}\sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} (x_i-\mu_B)(\sigma_B^2+\epsilon)^{-3/2}
}
$$


###### 2) $\displaystyle \frac{\partial \mathcal{L}}{\partial \mu_B}$ 유도

$\hat{x}_i$가 $\mu_B$에 두 경로로 의존한다는 점을 주의:

1. $\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$ 를 통해

2. $\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2$ 를 통해

먼저  1번. $\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$
$$
\frac{\partial \hat{x}_i}{\partial \mu_B}
= \underbrace{\frac{\partial (x_i-\mu_B)}{\partial \mu_B}}_{=-1}\cdot \frac{1}{\sqrt{\sigma_B^2+\epsilon}}
= -\frac{1}{\sqrt{\sigma_B^2+\epsilon}}
$$
다음 
$$
\frac{\partial}{\partial \sigma_B^2}\left(\frac{1}{\sqrt{\sigma_B^2+\epsilon}}\right)
= -\frac{1}{2}(\sigma_B^2+\epsilon)^{-3/2},
$$
$$
\frac{\partial \sigma_B^2}{\partial \mu_B}
= \frac{1}{m}\sum_{j} 2(x_j-\mu_B)\cdot\frac{\partial (x_j-\mu_B)}{\partial \mu_B}
= \frac{1}{m}\sum_{j} 2(x_j-\mu_B)\cdot(-1)
= -\frac{2}{m}\sum_{j}(x_j-\mu_B)
$$

하지만 $\sum_j (x_j-\mu_B)=0$ 이므로 $\frac{\partial \sigma_B^2}{\partial \mu_B}=0$. 따라서 두 번째 경로가 소멸하고,
$$
\frac{\partial \hat{x}_i}{\partial \mu_B}
= -\frac{1}{\sqrt{\sigma_B^2+\epsilon}}
$$

그러면
$$
\boxed{
\frac{\partial \mathcal{L}}{\partial \mu_B}
= \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i}
\frac{\partial \hat{x}_i}{\partial \mu_B}
= -\frac{1}{\sqrt{\sigma_B^2+\epsilon}}\sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i}
}
$$



###### 3) $\displaystyle \frac{\partial \mathcal{L}}{\partial x_i}$ 유도

$x_i$는 $\mu_B$와 $\sigma_B^2$를 거쳐 $\hat{x}_i$에 영향을 주므로, 세 경로로 역전파된다:

1. $\hat{x}_i$의 분자 $(x_i-\mu_B)$를 통한 직접 경로
2. $\hat{x}_i$의 분모 $\sigma_B^2 = \frac{1}{m}\sum_j (x_j-\mu_B)^2$를 통한 경로
3. $\mu_B = \frac{1}{m}\sum_j x_j$를 통한 경로

**경로 1**: $\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$에서 분자를 통해
$$
\frac{\partial \hat{x}_i}{\partial x_i}\bigg|_{\text{직접}} = \frac{1}{\sqrt{\sigma_B^2+\epsilon}}
$$

**경로 2**: $\sigma_B^2$를 통해
$$
\frac{\partial \hat{x}_i}{\partial \sigma_B^2} = -\frac{1}{2}(x_i-\mu_B)(\sigma_B^2+\epsilon)^{-3/2}
$$
$$
\frac{\partial \sigma_B^2}{\partial x_i} = \frac{2}{m}(x_i-\mu_B)
$$

**경로 3**: $\mu_B$를 통해
$$
\frac{\partial \hat{x}_i}{\partial \mu_B} = -\frac{1}{\sqrt{\sigma_B^2+\epsilon}}
$$
$$
\frac{\partial \mu_B}{\partial x_i} = \frac{1}{m}
$$

전체 그래디언트를 합치면:
$$
\frac{\partial \mathcal{L}}{\partial x_i}
= \frac{\partial \mathcal{L}}{\partial \hat{x}_i}\cdot\frac{1}{\sqrt{\sigma_B^2+\epsilon}}
+ \frac{\partial \mathcal{L}}{\partial \sigma_B^2}\cdot\frac{2}{m}(x_i-\mu_B)
+ \frac{\partial \mathcal{L}}{\partial \mu_B}\cdot\frac{1}{m}
$$

여기서 이미 구한 값들을 대입하면:
$$
\boxed{
\frac{\partial \mathcal{L}}{\partial x_i}
= \frac{\partial \mathcal{L}}{\partial \hat{x}_i}\cdot\frac{1}{\sqrt{\sigma_B^2+\epsilon}}
+ \frac{\partial \mathcal{L}}{\partial \sigma_B^2}\cdot\frac{2}{m}(x_i-\mu_B)
+ \frac{\partial \mathcal{L}}{\partial \mu_B}\cdot\frac{1}{m}
}
$$

