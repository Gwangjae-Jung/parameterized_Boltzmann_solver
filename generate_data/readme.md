# 데이터 생성 모듈 (`generate_data`)

본 디렉터리는 Variable Hard Sphere (VHS) 충돌 모델을 따르는 동차 볼츠만 방정식의 정밀 수치해를 생성하여 신경망 학습 및 테스트용 데이터셋으로 저장하는 모듈입니다.

수치해 계산에는 검증된 고속 스펙트럴 기법(**Fast Spectral Method, FSM**)이 사용되며, 시간 적분은 고전적 4차 룽게-쿠타 기법(**RK4**)을 통해 수행됩니다.

---

## 1. 파일 구성

- **`Boltzmann__bimaxwellian.py`**:
  - 두 개의 맥스웰 분포가 혼합된 Bi-Maxwellian 초기 조건($f_0$)을 무작위 섭동(속도, 온도)과 함께 샘플링합니다.
  - 설정된 각 커널 지수($\alpha \in [0, 1]$)에 대해 볼츠만 방정식 시계열 해($f(t, v)$)를 연산하여 `.pth` 파일로 저장합니다.
- **`Boltzmann__bimaxwellian__load_initial.py`**:
  - 기생성된 초기 조건 파일이 존재할 때 이를 로드하여 동일한 초기 조건 기반으로 다양한 충돌 커널 해를 추가 생성할 때 사용합니다.
- **`config_generation.yaml`**:
  - 기본 데이터셋 생성 설정 파일 (차원, 속도 격자 해상도, 시간 스텝, 샘플 개수, 저장 경로 등).
- **`config_generation__load.yaml`**:
  - 기저 초기 조건을 로드하여 생성할 때 사용하는 설정 파일.
- **`utils/`**:
  - YAML 설정 파서, 속도/온도 파라미터 무작위 샘플링 및 밀도 정규화 유틸리티.

---

## 2. 데이터 생성 프로세스

1. **초기 조건 생성**:
   - 유체역학적 제약 조건(단위 질량 밀도 등)을 만족하도록 무작위 Bi-Maxwellian 분포를 구성하고 정규화합니다.
2. **충돌 연산자 초기화**:
   - `core.numerical.solver.FastSM_Boltzmann_VHS`를 호출하여 주어진 해상도 및 커널 파라미터($\alpha$)에 대한 푸리에 모드를 설정합니다.
3. **시간 적분**:
   - $t=0$부터 $t_{\max}$까지 $\Delta t$ 간격으로 RK4 기법을 사용하여 분포 함수 $f(t, v)$의 변화 궤적을 산출합니다.
4. **저장**:
   - `datasets/` 경로에 텐서 파일(`alpha{VHS_ALPHA}__part{part}.pth`) 형태로 직렬화하여 저장합니다.

---

## 3. 실행 방법

```bash
# 기본 설정을 사용한 데이터 생성
python Boltzmann__bimaxwellian.py --config config_generation.yaml

# 다른 설정을 지정할 경우
python Boltzmann__bimaxwellian.py --config your_custom_config.yaml
```

---

## 4. 설정 항목 설명 (`config_generation.yaml`)

| 설정 키 | 설명 | 기본값 예시 |
| :--- | :--- | :--- |
| `computation.dtype` | 연산 정밀도 (`double` 또는 `float`) | `double` |
| `computation.device` | 수치해 계산에 사용할 GPU/CPU 장치 | `cuda:0` |
| `dataset.dim` | 속도 공간 차원 ($d=2$ 또는 $d=3$) | `3` |
| `dataset.res_v` | 속도 격자 해상도 (예: $32, 64, 128$) | `64` |
| `dataset.supp_v` | 속도 공간의 절단 반경 ($[-V_{\max}, V_{\max}]^d$) | `3.0` |
| `dataset.delta_t` | 시간 적분 간격 ($\Delta t$) | `0.1` |
| `dataset.max_t` | 최대 적분 시간 ($t_{\max}$) | `5.0` |
| `dataset.exp_speed` | 시뮬레이션할 VHS 충돌 지수 ($\alpha$) 리스트 | `[0.0, 0.05, ..., 1.0]` |
| `dataset.storage` | 생성된 데이터를 저장할 프로젝트 상대 경로 | `datasets/test` |
