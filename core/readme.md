# 코어 라이브러리 패키지 (`core`)

본 패키지는 외부 수치해석 라이브러리(`deep_numerical`)에 대한 의존성 없이 프로젝트가 자체적으로 완전하게 동작하도록 구축된 수치해석 및 딥러닝 코어 라이브러리입니다.

---

## 1. 패키지 아키텍처

```
core/
├── __init__.py                 # EINSUM_STRING, repeat, ones, zeros 등 기본 헬퍼
├── neural/                     # 신경망 아키텍처 및 연산자 모듈
│   ├── collision_op/           # FNO, PFNO, RFNO, PRFNO 신경망 충돌 연산자
│   ├── layer/                  # MLP, SpectralConv, RadialSpectralConv, RadialFourierLayer
│   ├── operator/               # Fourier Neural Operator (FNO, RadialFNO)
│   ├── parameterized_op/       # 파라미터 조건부 연산자 래퍼
│   ├── utils/                  # Activations, 가중치 초기화 및 파라미터 수 계산기
│   └── __init__.py             # BaseModule 정의
├── numerical/                  # 키네틱 방정식 수치해석 모듈
│   ├── distribution/           # 맥스웰 평형 분포 및 유체역학적 모멘트 연산
│   ├── integrate/              # 가우스-르장드르 수치적분
│   ├── math/                   # area_of_unit_sphere, sinc 등 수학 유틸리티
│   ├── solver/                 # Fast Spectral Method (FSM) 볼츠만 솔버 및 RK4 적분기
│   └── __init__.py
├── fft/                        # 푸리에 주파수 그리드 생성 및 FFT 텐서 변환 연산
└── utils/                      # velocity_grid, relative_error, dtype 변환, 시간 로깅
```

---

## 2. 주요 모듈 및 임포트 예시

### (1) 신경망 아키텍처
```python
from core.neural.operator import PRFNO, RadialFNO
from core.neural.layer import RadialFourierLayer, MLP
from core.neural.utils import count_parameters
```

### (2) 고속 스펙트럴 볼츠만 솔버
```python
from core.numerical.solver import FastSM_Boltzmann_VHS, one_step_RK4_classic
from core.numerical.math import area_of_unit_sphere
from core.numerical.distribution import maxwellian_homogeneous, compute_moments_homogeneous
```

### (3) 격자 및 오차 측정 유틸리티
```python
from core.utils import velocity_grid, relative_error, get_time_str
```
