"""
건강검진 위험도 예측 모델 학습 스크립트
노트북(view_data.ipynb)의 학습 로직을 스크립트로 추출
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib

# ── 1. 데이터 로드 ──────────────────────────────────────────────────────────────
print("[1/5] CSV 데이터 로딩 중...")
p = "C:/Users/User/OneDrive/문서/Gibbeum Project/건강검진 플랫폼 검진결과분석 모델링/국민건강보험공단_건강검진정보_2024.CSV"
df = pd.read_csv(p, encoding="cp949", low_memory=False)

# 불필요한 열 제거
columns_to_drop = ['치아우식증유무', '결손치 유무', '치아마모증유무', '제3대구치(사랑니) 이상', '치석']
df = df.drop(columns=columns_to_drop, errors='ignore')
print(f"  로드 완료: {len(df):,}행, {df.shape[1]}열")

# ── 2. Winsorizing (극단값 클리핑) ──────────────────────────────────────────────
print("[2/5] 극단값 처리 (Winsorizing 1~99%) 중...")
numeric_columns = [
    '신장(5cm단위)', '체중(5kg단위)', '허리둘레', '수축기혈압', '이완기혈압',
    '식전혈당(공복혈당)', '총콜레스테롤', '트리글리세라이드', 'HDL콜레스테롤',
    'LDL콜레스테롤', '혈색소', '요단백', '혈청크레아티닌',
    '혈청지오티(AST)', '혈청지피티(ALT)', '감마지티피'
]
for col in numeric_columns:
    if col in df.columns and df[col].dtype in ['int64', 'float64']:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower, upper=upper)

# ── 3. BMI 계산 ──────────────────────────────────────────────────────────────
print("[3/5] BMI 파생변수 생성 중...")
df['BMI'] = df['체중(5kg단위)'] / (df['신장(5cm단위)'] / 100) ** 2

# ── 4. 위험도 레이블 생성 ─────────────────────────────────────────────────────
print("[4/5] 위험도 레이블 생성 중...")

def classify_risk(row):
    score = 0
    if row['수축기혈압'] >= 140 or row['이완기혈압'] >= 90:
        score += 2
    elif row['식전혈당(공복혈당)'] >= 126:
        score += 2
    elif row['식전혈당(공복혈당)'] >= 100:
        score += 1
    if row['총콜레스테롤'] >= 240:
        score += 1
    elif row['총콜레스테롤'] >= 200:
        score += 0.5
    if row['BMI'] >= 30:
        score += 2
    elif row['BMI'] >= 25:
        score += 1
    if row['성별코드'] == 1:
        if row['허리둘레'] >= 90: score += 1
    else:
        if row['허리둘레'] >= 85: score += 1
    if row['성별코드'] == 1:
        if row['혈색소'] < 13: score += 1
    else:
        if row['혈색소'] < 12: score += 1
    if row['성별코드'] == 1:
        if row['감마지티피'] > 77: score += 1
    else:
        if row['감마지티피'] > 45: score += 1
    if row['성별코드'] == 1:
        if row['혈청크레아티닌'] > 1.2: score += 1
    else:
        if row['혈청크레아티닌'] > 1.0: score += 1
    if score >= 4: return 2
    elif score >= 2: return 1
    else: return 0

df['위험도'] = df.apply(classify_risk, axis=1)
label_map = {0: '정상', 1: '주의', 2: '고위험'}
counts = df['위험도'].value_counts().sort_index()
for k, v in counts.items():
    print(f"  {label_map[k]}: {v:,}명 ({v/len(df)*100:.1f}%)")

# ── 5. 모델 학습 & 저장 ─────────────────────────────────────────────────────
print("[5/5] XGBoost 모델 학습 중...")
features = [
    '성별코드', '연령대코드(5세단위)', 'BMI', '허리둘레',
    '수축기혈압', '이완기혈압', '식전혈당(공복혈당)',
    '총콜레스테롤', 'HDL콜레스테롤', 'LDL콜레스테롤', '트리글리세라이드',
    '혈색소', '혈청크레아티닌', '혈청지오티(AST)', '혈청지피티(ALT)', '감마지티피',
    '흡연상태', '음주여부'
]

df_model = df[features + ['위험도']].dropna()
print(f"  학습 데이터: {len(df_model):,}행")

X = df_model[features]
y = df_model['위험도']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n=== 모델 성능 ===")
print(classification_report(y_test, y_pred, target_names=['정상', '주의', '고위험']))

model_path = "C:/Users/User/OneDrive/문서/Gibbeum Project/건강검진 플랫폼 검진결과분석 모델링/health_model.pkl"
joblib.dump(model, model_path)
print(f"\n모델 저장 완료: {model_path}")
