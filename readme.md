# Parameterized Boltzmann Solver (Autoregressive Method)

본 프로젝트는 동차(Homogeneous) 볼츠만 방정식을 **Operator Learning (신경망 연산자 학습)** 기법을 활용하여 해결하는 딥러닝 연구 프로젝트입니다.  
다양한 충돌 커널(Collision Kernel) 파라미터(VHS 모델의 $\alpha$)가 주어지더라도 일관된 추론을 할 수 있는 단일 모델을 학습시키며, 초기 조건 $f(0, \cdot)$로부터 시간 단계마다 순차적으로 다음 시간의 해를 예측하는 **Autoregressive 방식($f(k \Delta t, \cdot) \mapsto f((k+1) \Delta t, \cdot)$)**으로 동작합니다.

---

## 1. 프로젝트 주요 특징

- **Autoregressive 시간 전진**: 이전 시간 단계의 분포 함수 $f_t$와 커널 파라미터 $\alpha$를 입력받아 다음 시간 단계 $f_{t+\Delta t}$를 직접 예측합니다.
- **물리적 대칭성 및 보존 법칙 반영**:
  - 회전 대칭성을 보존하는 방사 대칭 푸리에 신경 연산자(**RFNO, PRFNO**) 적용.
  - 질량, 운동량, 에너지 보존 법칙을 유도하는 보존 손실 함수(Conservation Loss) 탑재.
- **다양한 모델 아키텍처 비교**:
  - `FNO`: 기본 Fourier Neural Operator
  - `RFNO`: Radial Fourier Neural Operator (회전 불변성 반영)
  - `PFNO`: Parameterized FNO (커널 파라미터 조건부 입력)
  - `PRFNO`: Parameterized Radial FNO (커널 파라미터 + 방사 대칭 결합)
- **자체 완비형 코어 패키지 (`core`)**: 외부 라이브러리(`deep_numerical`) 의존성 없이 독립적으로 실행 가능.

---

## 2. 디렉터리 구조

```
parameterized_Boltzmann_solver/
├── ablation__config.py         # 모델 학습 및 어블레이션 실험 공통 설정 (차원, 주파수, 에포크 등)
├── main__fno_param.py          # FNO 계열 모델 자동화 학습 실행 진입점
├── main__pfno_param.py         # PFNO 모델 자동화 학습 실행 진입점
├── main__prfno_param.py        # PRFNO 모델 자동화 학습 실행 진입점
├── main__rfno_param.py         # RFNO 모델 자동화 학습 실행 진입점
├── runfile.sh                  # 백그라운드 멀티 GPU 학습 쉘 스크립트
├── core/                       # 신경망 레이어 및 볼츠만 고속 스펙트럴 수치 솔버 패키지
├── generate_data/              # 고속 스펙트럴 솔버(Fast Spectral Method)를 이용한 학습/테스트 데이터 생성
├── fno__VHS__param_only/       # 각 모델별 실제 학습 스크립트 및 체크포인트/로그 저장소
├── inference_2D/               # 2D 볼츠만 방정식 추론, 상대 오차 분석 및 GPU 벤치마크 노트북
├── inference_3D/               # 3D 볼츠만 방정식 추론 및 대규모 텐서 평가 노트북
└── train_utils/                # 데이터 로더, 손실 함수, 인자 파서 등 학습 공통 유틸리티
```

---

## 3. 시작하기 (Quick Start)

### (1) 데이터셋 생성
학습 및 평가에 필요한 볼츠만 방정식 수치해 궤적을 생성합니다.
```bash
cd generate_data
python Boltzmann__bimaxwellian.py --config config_generation.yaml
```
생성된 궤적 데이터는 `datasets/` 하위 경로에 저장됩니다.

### (2) 모델 학습 실행
루트 디렉터리에서 원하는 모델의 메인 스크립트를 실행합니다:
```bash
# 단일 GPU 실행 (예: CUDA 장치 0번)
python main__prfno_param.py --cuda_index 0

# 또는 백그라운드 일괄 실행
bash runfile.sh
```
학습 설정(에포크, 배치 크기, 모드 수 등)은 `ablation__config.py`에서 손쉽게 수정할 수 있습니다.

### (3) 결과 분석 및 추론
`inference_2D/` 또는 `inference_3D/` 디렉터리의 주피터 노트북을 실행하여 학습된 모델의 시간 추론 결과, 오차율, 속도 및 GPU 메모리 벤치마크를 확인합니다:
- `inference_2D/inference.ipynb`: 2D 시계열 롤아웃 오차 분석 및 시각화
- `inference_3D/inference.ipynb`: 3D 시계열 롤아웃 오차 분석 및 시각화

---

## 4. 참고 사항
- 모든 파일 경로는 프로젝트 루트(`_ROOT`)를 기준으로 자동 해석되므로, 어느 위치에서 스크립트를 호출하더라도 안전하게 실행됩니다.
- GPU 메모리가 부족할 경우 `ablation__config.py`의 `BATCH_SIZE`를 조절하시기 바랍니다.
