# 2D 추론 및 벤치마크 모듈 (`inference_2D`)

본 디렉터리는 학습 완료된 2차원 볼츠만 신경 연산자 모델들의 추론 성능, 시계열 롤아웃(Rollout) 오차, 연산 속도 및 GPU 메모리 사용량을 종합적으로 평가하고 시각화하는 환경입니다.

---

## 1. 주요 노트북 및 스크립트 구성

- **`inference.ipynb`**:
  - **핵심 평가 노트북**: 초기 분포 $f_0$로부터 최종 시간 $t=5.0$까지 autoregressive하게 연속 추론을 수행합니다.
  - 참값(FSM 수치해) 대비 각 시간 단계별 상대 $L_2$ 오차를 계산하고, 매크로스코픽 물리량(질량, 운동량, 에너지)의 보존 특성을 검증합니다.
- **`inference__speed.ipynb`**:
  - 각 모델 아키텍처(FNO, RFNO, PFNO, PRFNO) 및 전통 수치해석 솔버(FSM)의 1스텝당 추론 지연 시간(Latency) 및 스루풋(Throughput)을 정량 비교합니다.
- **`inference__peak_memory__*.ipynb`**:
  - PyTorch의 CUDA 메모리 추적 API를 사용하여 모델별 포워드 연산 시의 GPU 피크 메모리(Peak VRAM)를 측정합니다.
- **`loader.py`**:
  - `fno__VHS__param_only/` 경로의 학습 체크포인트(`state_dict.pth`, `normalizer.pth`, `train_history.pth`)를 편리하게 로드하는 유틸리티입니다.
- **`dataset.py` & `plot.py`**:
  - 테스트 데이터셋 로더 및 오차 분포, 1D 단면 그래프, 컨투어 시각화 함수 모음입니다.
- **`ood_test_dataset/`**:
  - 학습 분포 밖(Out-of-Distribution)의 극단적 초기 조건 또는 커널 파라미터에 대한 일반화 능력을 테스트하는 데이터셋 디렉터리입니다.

---

## 2. 사용 방법

1. Jupyter Notebook 환경에서 `inference.ipynb`를 실행합니다.
2. 상단 설정 셀에서 평가하고자 하는 모델과 파라미터를 지정합니다:
   ```python
   list_max_freq = [6, 8, 10]
   list_seed     = [0, 1, 2]
   max_freq      = list_max_freq[2] # 예: max_freq=10
   seed          = list_seed[0]     # 예: seed=0
   res_v         = 128              # 평가 격자 해상도
   ```
3. 순차적으로 셀을 실행하여 롤아웃 오차 곡선 및 시각화 플롯을 확인합니다.
