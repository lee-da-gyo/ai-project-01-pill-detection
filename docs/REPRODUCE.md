# ⚙️ 실행 방법과 전체 구조

> [README](../README.md) 8 · 9장의 상세판이다. 전체 파일 트리, 모든 실험 명령, Kaggle Notebook 실행 방법, 재현 조건을 담았다.

## 프로젝트 구조

```
ai-project-01-pill-detection/
├── .github/workflows/ci.yml    # CI: ruff · pytest · 커밋된 OOF 분석 결과 재현 확인
├── pyproject.toml              # pip install -e . 로 pilldet 패키지 설치
├── src/pilldet/                # 📦 공통 코드 패키지 (노트북·스크립트가 함께 사용)
│   ├── config.py               #   경로 자동 탐색 (로컬 · Kaggle), seed, fold 수
│   ├── data.py                 #   라벨 병합, 이미지 로딩, 파일명 파싱
│   ├── boxes.py                #   좌표 변환, IoU, YOLO ↔ 픽셀 변환
│   ├── segment.py              #   배경 차분 기반 알약 분할
│   ├── clean.py                #   라벨 감사 · 규칙 기반 수정 · 사람 확인 결정 적용 · 무결성 검사
│   ├── split.py                #   조합 단위 Stratified Group K-Fold
│   ├── export.py               #   YOLO / COCO 내보내기 · 왕복 검증
│   ├── preprocess.py           #   CLAHE 전처리 (학습 · 검증 · 테스트 공통)
│   ├── augment.py              #   Copy-Paste 합성 (희귀 약 보강)
│   ├── train.py                #   설정 파일 기반 YOLO 학습
│   ├── predict.py              #   예측 · 도메인 후처리
│   ├── evaluate.py             #   COCO mAP · 클래스별 AP
│   ├── submission.py           #   Kaggle 제출 파일 생성 · 형식 검사
│   ├── classifier.py           #   VGG16 · ResNet50 · EfficientNet-B0 약 분류기
│   ├── two_stage.py            #   검출 박스 + 분류기 결합 (2단계 모델)
│   ├── ensemble.py             #   Weighted Boxes Fusion 직접 구현
│   └── viz.py                  #   한글 폰트, 박스 시각화
├── scripts/
│   ├── prepare_data.py         # ⚙️ 데이터 준비 전체를 한 번에 실행
│   ├── audit_labels.py         # ⚙️ 라벨 감사 후보: 같은 좌표 중복(데이터) · 고신뢰 불일치(모델 보조)
│   ├── make_copy_paste.py      # ⚙️ fold별 Copy-Paste 합성 데이터 생성
│   ├── make_preprocessed.py    # ⚙️ 전처리 데이터셋 생성 · Train/Test 분포 차이 측정
│   ├── run_experiment.py       # ⚙️ 학습 → 평가 → 제출 파일 → 실험 기록 (--fold, --skip-train)
│   ├── summarize_cv.py         # ⚙️ 5-Fold 점수 · OOF 집계
│   ├── analyze_oof.py          # ⚙️ open-set 상한 분해 · WBF 설정 선택 · 근사 메타 검증
│   ├── epoch_sensitivity.py    # ⚙️ 전체 학습 에포크 민감도
│   ├── compare_oof.py          # ⚙️ 두 실험 OOF 짝지은 비교 · 조합 단위 부트스트랩
│   ├── predict_oof.py          # ⚙️ 학습한 5-Fold 모델로 추론 설정(TTA)만 바꿔 OOF 다시 만들기
│   ├── error_analysis.py       # ⚙️ OOF 오류 분석: 클래스별 AP · IoU별 mAP · 실패 유형 · 처음 보는 약과 신뢰도
│   ├── prepare_aihub.py        # ⚙️ 허용 AI Hub 데이터 정제 · 변환 (금지 파일은 열지 않음)
│   ├── train_aihub.py          # ⚙️ AI Hub + 대회 데이터 학습 (최고 점수 모델)
│   ├── train_classifier.py     # ⚙️ 분류기 3종 5-Fold 비교
│   ├── eval_two_stage.py       # ⚙️ 2단계 모델 평가
│   ├── make_submission.py      # ⚙️ 최종 모델 WBF → 제출 파일
│   ├── build_report.py         # ⚙️ 보고서 HTML → PDF
│   └── build_worklog.py        # ⚙️ 업무일지 → PDF
├── app/
│   └── demo.py                 # 🖥️ 데모 앱 (Streamlit): 사진 → 약 이름 · 위치 · 신뢰도 · 확인 필요 표시
├── tests/
│   ├── test_core.py            # ✅ 핵심 함수 단위 테스트 (함수 45개 · 케이스 47개)
│   └── test_scripts.py         # ✅ 분석 스크립트 테스트: 부트스트랩 재표본 · WBF 짝 비교 · 메타 검증 (3개)
├── notebooks/                  # 📓 분석 과정 · 실행 결과 · 해석
│   ├── 01_eda.ipynb            #   탐색적 데이터 분석 → 의사결정
│   ├── 02_preprocess.ipynb     #   라벨 정제 · YOLO/COCO 변환 · 5-Fold (핵심 함수 코드 포함)
│   └── 03_baseline.ipynb       #   베이스라인 학습 결과 · 오류 분석 · 첫 제출
├── configs/                    # 실험 설정 (실험 하나 = 파일 하나) · 제출 형식 · 라벨 확인 결정(label_review.csv)
├── reports/
│   ├── figures/                # 분석 그래프 (보고서용)
│   ├── logs/                   # 스크립트 · 테스트 실행 로그
│   ├── label_fixes.csv         # 라벨 수정 내역
│   ├── class_map.csv           # 약 코드 ↔ YOLO 클래스 번호
│   ├── folds.csv               # 이미지별 fold
│   └── annotations_clean.json  # 정제된 COCO 라벨
└── docs/
    ├── PLAN.md                 # 계획서 · 요구사항 추적 · 실험 기록
    ├── EXPERIMENTS.md          # 실험 기록 전체 (표 · 그림 · 통계 검정)
    ├── REPRODUCE.md            # 이 문서: 전체 구조 · 실행 방법 · 재현 조건
    ├── demo/                   # 시연 영상(mp4)과 GIF
    ├── LABEL_REVIEW.md         # 라벨 수정 검수 기록
    ├── report/                 # 보고서 원본(HTML·CSS)과 PDF
    └── worklog/                # 일별 업무일지
```

**설계 원칙**: 로직은 `src/pilldet`에 한 번만 구현하고, 노트북은 그것을 불러 **과정과 결과를 설명**하며, 스크립트는 같은 로직을 **자동으로 실행**한다. 그래서 노트북과 스크립트의 결과가 항상 같다.

<br>

## 실행 방법

<details open>
<summary><b>로컬 (전체 파이프라인 · 실험과 제출에 사용)</b></summary>

```bash
pip install -r requirements.txt
python scripts/prepare_data.py                                       # 라벨 정제 · YOLO/COCO 변환 · 5-Fold (약 15초)
python scripts/audit_labels.py                                       # 라벨 감사 후보 (5-Fold 학습 뒤 OOF 필요)
python -m pytest tests -v                                            # 단위 테스트

# 실험 1회 = 설정 파일 1개 (RTX 3060 기준 YOLO11s 약 6분)
python scripts/run_experiment.py --config configs/e0_baseline.yaml   # fold 0
python scripts/run_experiment.py --config configs/e0_baseline.yaml --fold 1   # fold 1~4도 같은 방식

# 5-Fold 분석
python scripts/summarize_cv.py e0_baseline e2_yolo11m                # fold별·OOF 점수
python scripts/analyze_oof.py e0_baseline e2_yolo11m                 # open-set 상한 분해 · WBF 설정 선택 · 근사 메타 검증
python scripts/epoch_sensitivity.py e0_baseline e2_yolo11m           # 전체 학습 에포크 민감도

# 전처리 비교 (CLAHE): 데이터셋 생성 → 5-Fold (fold 1~4는 --fold) → 기준과 짝지은 비교
python scripts/make_preprocessed.py --method clahe
python scripts/run_experiment.py --config configs/e4_clahe.yaml
python scripts/compare_oof.py e0_baseline e4_clahe
python scripts/error_analysis.py                                     # 현재 최종 후보(11s+11m WBF) OOF 오류 분석

# ★ Kaggle 최고 점수 모델 재현 (public 0.63017, YOLO11s + 허용 AI Hub 조합 데이터)
#   라벨 TL_1,3~8_조합.zip과 이미지 TS_1,3~8_조합.zip을 받아 둔다 (금지 파일 TL_2·TS_2는 쓰지 않으며, 있어도 건너뛴다)
python scripts/prepare_aihub.py --labels <라벨 zip 폴더> --images <이미지 zip 폴더>
python scripts/train_aihub.py --config configs/a1_aihub_yolo11s.yaml         # 검증 학습 → best epoch 28
python scripts/train_aihub.py --config configs/a2_aihub_full_yolo11s.yaml    # 전체 데이터 28 에포크
python scripts/make_submission.py --runs a2_aihub_full_yolo11s --name a2_aihub_full_11s --class-map class_map_aihub.csv

# 추가 데이터를 쓰기 전 최종 모델 (232장 전체 · 검증 없음 · last.pt) → 제출 파일
python scripts/run_experiment.py --config configs/final_yolo11s.yaml
python scripts/run_experiment.py --config configs/final_yolo11m.yaml
python scripts/make_submission.py --runs final_yolo11s final_yolo11m --weights 2 1 --iou-thr 0.6 --name final_11s_11m_wbf_iou06

# 학습률(R11) · TTA(R14) 비교
python scripts/run_experiment.py --config configs/e5_lr_high.yaml   # fold 0, e5_lr_low도 같은 방식
python scripts/predict_oof.py e0_baseline --tta                      # 학습 없이 TTA로 OOF만 다시 → compare_oof.py로 비교

# 보고서 PDF · 데모 앱
python scripts/build_report.py                                       # docs/report/report.pdf (Edge 인쇄 엔진)
python scripts/build_worklog.py                                      # docs/worklog/worklog.pdf
streamlit run app/demo.py
```

데이터 폴더를 레포 옆(`../sprint_ai_project1_data`)에 두면 자동으로 찾는다. 다른 위치라면 `--data-dir` 옵션이나 환경 변수 `PILL_DATA_DIR`로 지정한다.
`--skip-train`으로 기존 가중치를 다시 평가할 때는 학습 당시 설정 기록과 지금 설정이 다르면 멈춘다.
사용자 폴더에 쓸 수 없는 환경(CI, 샌드박스)에서는 Ultralytics 설정 폴더를 지정한다: `YOLO_CONFIG_DIR=/tmp/ultralytics python -m pytest tests`
실행 로그: [`reports/logs/`](../reports/logs) · 단위 테스트: [`reports/logs/pytest.log`](../reports/logs/pytest.log)

> **재현 조건** – `requirements.txt`는 최종 모델을 만든 환경(Python 3.14, Windows 11, RTX 3060)의 버전이다. 실험 도중 pandas · Pillow 버전이 바뀐 구간이 있었고, 그 범위와 검증하지 않은 부분은 [실험 기록의 재현 조건](EXPERIMENTS.md#재현-조건-실행-환경-기록)에 적었다. 단위 테스트와 코드 검사(`ruff check src scripts tests`)는 [GitHub Actions](../.github/workflows/ci.yml)에서 push마다 실행된다.

</details>

<details>
<summary><b>Kaggle Notebook (EDA·전처리 노트북 실행)</b></summary>

1. Kaggle에서 노트북을 Import 한다 (File → Import Notebook)
2. 오른쪽 패널 **Add Input**에서 대회 데이터를 연결한다. 데이터 경로는 `/kaggle/input`에서 자동으로 찾는다
3. Settings → **Internet On** (공통 코드 내려받기, 한글 폰트 설치용) → **Run All**. 노트북이 이 레포를 자동으로 clone 해서 `src/pilldet`을 불러온다
4. `01_eda` → `02_preprocess` 순서로 실행한다. 결과물은 `/kaggle/working/`에 저장된다

</details>
