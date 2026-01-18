# 🧠 Craving Biomarker Analysis (Official Test)

> **Official Test Repo of [STEAM - Craving Biomarker Analysis]** > STEAM 과제의 '갈망(Craving) 바이오마커 분석' 프로젝트를 위한 공식 테스트 및 검증 레포지토리입니다.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()

---

## 📖 프로젝트 개요 (Project Overview)

이 프로젝트는 **갈망(Craving)** 상태와 연관된 **생체신호(Biomarker)** 를 식별하고 분석하기 위한 연구의 일환입니다. 본 레포지토리는 본 연구의 분석 파이프라인, 전처리 알고리즘, 모델링 코드를 테스트하고 검증하기 위해 운영됩니다.

### 🎯 연구 목표
1. **데이터 전처리 검증:** 다양한 생체 신호 데이터(예: EEG, ECG, GSR 등)의 노이즈 제거 및 정규화 프로세스 테스트.
2. **피처 추출(Feature Extraction):** 갈망 상태를 대변할 수 있는 유의미한 바이오마커 후보군 추출.
3. **분석 모델 테스트:** 머신러닝/통계적 기법을 활용한 분류 및 예측 모델 성능 평가.

---

## 🛠 기술 스택 (Tech Stack)

이 프로젝트는 다음과 같은 기술과 라이브러리를 기반으로 합니다.

* **Language:** Python 3.x
* **Data Analysis:** Pandas, NumPy, SciPy
* **Visualization:** Matplotlib, Seaborn
* **Signal Processing:** [MNE-Python / NeuroKit2 등 사용하는 라이브러리 입력]
* **Machine Learning:** Scikit-learn, [PyTorch/TensorFlow 등]

---

## 📂 폴더 구조 (Directory Structure)

```bash
hyuki0003-Craving_Biomarker_Analysis_official_test/
├── 📂 code/
│   ├── calculate_decline_final.ipynb    # 갈망 중재 효과 검증
│   ├── features_correlation_final.ipynb # 갈망-SA 상관성 분석           
│   ├── util/                
│         └── util.py                    # Preprocessed data
├── .gitignore                           # Git 제외 파일 목록
├── requirements.txt                     # 의존성 패키지 목록
└── README.md                            # 프로젝트 설명 파일
