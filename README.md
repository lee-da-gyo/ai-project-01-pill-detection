# 💊 경구약제 이미지 객체 검출 (Pill Detection)

> 코드잇 스프린트 AI 15기 초급 프로젝트 (개인)
> 헬스케어 스타트업 **헬스잇(Health Eat)** 시나리오: 사용자가 찍은 알약 사진에서 **최대 4개 알약의 종류(클래스)와 위치(바운딩 박스)** 를 검출한다.

## 📌 프로젝트 개요
| 항목 | 내용 |
|---|---|
| 기간 | 2026-10-27 ~ 2026-11-05 |
| 과제 | 객체 검출 (Object Detection), 56개 클래스 |
| 데이터 | 학습 이미지 232장 (라벨 763개), 테스트 이미지 842장, 976×1280 |
| 평가 | Kaggle Private Competition |
| 환경 | Kaggle Notebook (GPU), Python, PyTorch |

## 📂 폴더 구조
```
pill-detection/
├── notebooks/        # 단계별 노트북 (EDA → 전처리 → 학습 → 추론)
│   ├── 01_eda.ipynb          # 탐색적 데이터 분석 → 모델링 의사결정
│   └── 02_preprocess.ipynb   # 라벨 정제 · YOLO 변환 · 조합 단위 K-Fold
├── src/              # 재사용 코드 (데이터 변환, 학습, 추론)
├── configs/          # 학습 설정 (데이터셋 yaml, 하이퍼파라미터)
├── reports/
│   └── figures/      # 보고서용 그래프
└── docs/
    ├── PLAN.md       # 일정표, 실험 기록
    └── worklog/      # 일별 업무일지
```
원본 데이터는 용량 문제로 레포에 넣지 않는다 (`.gitignore`).

## 🚀 실행 방법 (Kaggle Notebook)
1. Kaggle에서 `notebooks/01_eda.ipynb`를 Import 한다 (File → Import Notebook).
2. 오른쪽 패널 **Add Input**에서 대회 데이터를 연결한다. 데이터 경로는 노트북이 `/kaggle/input` 아래에서 자동으로 찾는다.
3. Settings에서 **Internet On** (한글 폰트 설치용) → Run All
4. 그래프와 `class_map.csv`가 `/kaggle/working/` 에 저장된다.

로컬에서 실행할 때는 환경 변수 `PILL_DATA_DIR`에 데이터 폴더 경로를 지정한다.

## 🧹 데이터 정제 요약
원본 라벨을 고전 영상처리(배경 차분)로 분할한 알약 영역과 대조해 **라벨 오류 10건을 규칙 기반으로 찾아 고쳤다.** 모든 수정 내역은 [`reports/label_fixes.csv`](reports/label_fixes.csv)에 있다.

| 오류 유형 | 건수 | 처리 |
|---|---|---|
| 이미지 범위 밖 박스 (x=6567) | 1 | 자릿수 입력 오류로 판단 → x=567 복원 |
| 빈 배경에 그려진 박스 | 1 | 라벨 없는 같은 크기의 알약 위치로 이동 |
| 누락 박스 (사진에는 있는 알약) | 8 | 파일명의 약 코드 + 라벨 없는 알약 → 박스 추가 |

- 검증 분할: **조합(combo) 단위 Stratified Group 5-Fold**. 같은 조합을 각도만 바꿔 찍은 사진이 train/val에 동시에 들어가지 않는다.
- 산출물: YOLO 데이터셋과 fold별 yaml, COCO 정제본(`annotations_clean.json`), `class_map.csv`, `folds.csv`

## 📊 결과
| 실험 | 모델 | val mAP | Kaggle |
|---|---|---|---|
| (진행 중) | | | |

## 📑 제출물
- 보고서 PDF: (작성 예정)
- 업무일지: [docs/worklog](docs/worklog)

## ⚠️ 데이터 사용 규칙
AI Hub `경구약제조합 5000종`의 `TL_2_조합.zip`, `TS_2_조합.zip`은 대회 데이터의 원본이므로 사용하지 않는다.
