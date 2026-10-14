# 모델 학습 모듈 (`fno__VHS__param_only`)

본 디렉터리는 다양한 VHS 충돌 커널 지수($\alpha$)를 조건으로 입력받아 동차 볼츠만 방정식의 시간 전진 해를 추론하는 4가지 신경 연산자 모델의 학습 스크립트와 가중치 체크포인트를 포함하고 있습니다.

---

## 1. 지원 모델 아키텍처

1. **`train__fno_param.py` (`FNO-param`)**:
   - 표준 Fourier Neural Operator 기반 모델입니다.
   - 입력 채널에 분포 $f(t, v)$와 함께 커널 파라미터 $\alpha$를 상수 필드로 확장하여 2개 채널(`in_channels=2`)로 전달합니다.
2. **`train__rfno_param.py` (`RFNO-param`)**:
   - 회전 변환에 대한 등변성(Equivariance)을 보장하는 방사 대칭 푸리에 레이어(`RadialFourierLayer`)를 탑재한 모델입니다.
3. **`train__pfno_param.py` (`PFNO-param`)**:
   - 파라미터 $\alpha$를 별도의 조건부 임베딩 레이어로 처리하여 푸리에 연산자의 웨이트 텐서와 결합하는 파라미터화 FNO 모델입니다 (`in_channels=1`).
4. **`train__prfno_param.py` (`PRFNO-param`)**:
   - **(본 연구의 핵심 모델)** 방사 대칭성(Radial Symmetry)과 파라미터 조건부 임베딩을 동시에 적용한 모델로, 가장 높은 파라미터 효율성과 낮은 롤아웃 누적 오차를 제공합니다.

---

## 2. 파일 및 디렉터리 구성

- **`train__*_param.py`**:
  - 각 모델별 메인 학습 스크립트입니다.
  - 학습 손실 함수(상대 $L_2$ 오차)와 물리적 보존 손실(질량, 모멘텀, 에너지)을 결합하여 학습합니다.
- **`{MODEL_NAME}/`** (예: `PRFNO2D-param/`, `PRFNO3D-param/` 등):
  - 학습 완료 후 최고 성능 모델 가중치(`state_dict.pth`), 데이터 정규화기(`normalizer.pth`), 학습 곡선 기록(`train_history.pth`) 및 로그 파일이 저장되는 디렉터리입니다.
- **`{MODEL_NAME}__ablation.txt`**:
  - 주파수 모드 수(`max_freq`), 난수 시드(`seed`)별 실험 진행 로그입니다.

---

## 3. 학습 실행 방법

일반적으로 프로젝트 루트의 `main__*_param.py` 또는 `runfile.sh`를 통해 실행하지만, 개별 스크립트를 직접 호출할 수도 있습니다:

```bash
# PRFNO 2D 모델 단독 학습 예시
python train__prfno_param.py \
    --model_type prfno \
    --dim_domain 2 \
    --max_freq 8 \
    --n_layers 4 \
    --learning_rate 1e-4 \
    --batch_size 1024 \
    --num_epochs 1000 \
    --path_train ../datasets/train/Boltzmann_2D__coeff1.59e-01__res_t051_v032 \
    --path_val ../datasets/validation/Boltzmann_2D__coeff1.59e-01__res_t051_v032 \
    --cuda_index 0
```
