# 모델 학습 유틸리티 (`train_utils`)

본 디렉터리는 신경 연산자 모델들의 학습 파이프라인에서 공통으로 사용되는 데이터 로딩, 손실 함수, 학습률 스케줄러 및 하이퍼파라미터 파서 모듈들을 제공합니다.

---

## 1. 모듈별 기능 설명

- **`dataloaders.py`**:
  - `datasets/` 디렉터리에 저장된 시계열 궤적(`.pth`) 파일들을 불러와 훈련 및 검증용 `DataLoader`를 구성합니다.
  - 시간 인덱스 $t \mapsto t+\Delta t$ 쌍을 자동 생성하며, 궤적별 서브샘플링 및 가우시안 정규화(`GaussianNormalizer`) 적용을 지원합니다.
- **`loss_functions.py`**:
  - **상대 $L_2$ 오차 손실**: 예측값과 참값 간의 정규화된 차이를 계산합니다.
  - **물리적 보존 손실 (Conservation Loss)**: 동차 볼츠만 방정식의 필수 물리 성질인 질량(Mass), 운동량(Momentum), 운동 에너지(Energy)가 시간 전진 후에도 보존되도록 페널티를 부여하는 손실 함수입니다.
- **`parameter_transformers.py`**:
  - 스칼라 충돌 커널 파라미터 $\alpha$를 신경망 입력 형태(공간 확장 텐서 또는 임베딩 벡터)로 변환하는 도구 모음입니다.
- **`lr_schedulers.py`**:
  - Cosine Annealing, StepLR 등 학습 안정성을 위한 학습률 스케줄링 헬퍼입니다.
- **`parser.py` (`CustomParser`)**:
  - 명령행 인자(`argparse`)를 통해 차원, 주파수 모드 수, 레이어 수, 배치 크기, 학습률, 데이터 경로 등을 유연하게 주입받는 인자 파서입니다.
- **`helper.py`**:
  - 학습 소요 시간 포맷팅(`sec_to_hms`), 로그 출력 헬퍼 등 편의 함수를 포함합니다.
