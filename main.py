from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="건강검진 위험도 예측 API")

# 서버 시작 시 모델 로드
model = joblib.load("health_model.pkl")

# 입력 스키마 정의
class HealthInput(BaseModel):
    성별: str           # '남' 또는 '여'
    연령대: int         # 5세 단위 코드 (예: 10=50~54세)
    신장: float         # cm
    체중: float         # kg
    허리둘레: float     # cm
    수축기혈압: float
    이완기혈압: float
    공복혈당: float
    총콜레스테롤: float
    HDL콜레스테롤: float
    LDL콜레스테롤: float
    트리글리세라이드: float
    혈색소: float
    혈청크레아티닌: float
    AST: float
    ALT: float
    감마지티피: float
    흡연상태: int       # 1=비흡연, 2=과거흡연, 3=현재흡연
    음주여부: int       # 0=안함, 1=함


@app.get("/")
def root():
    return {"message": "건강검진 위험도 예측 API 정상 동작 중"}


@app.post("/predict")
def predict(data: HealthInput):
    성별코드 = 1 if data.성별 == "남" else 2
    bmi = data.체중 / (data.신장 / 100) ** 2

    input_df = pd.DataFrame([{
        "성별코드": 성별코드,
        "연령대코드(5세단위)": data.연령대,
        "BMI": bmi,
        "허리둘레": data.허리둘레,
        "수축기혈압": data.수축기혈압,
        "이완기혈압": data.이완기혈압,
        "식전혈당(공복혈당)": data.공복혈당,
        "총콜레스테롤": data.총콜레스테롤,
        "HDL콜레스테롤": data.HDL콜레스테롤,
        "LDL콜레스테롤": data.LDL콜레스테롤,
        "트리글리세라이드": data.트리글리세라이드,
        "혈색소": data.혈색소,
        "혈청크레아티닌": data.혈청크레아티닌,
        "혈청지오티(AST)": data.AST,
        "혈청지피티(ALT)": data.ALT,
        "감마지티피": data.감마지티피,
        "흡연상태": data.흡연상태,
        "음주여부": data.음주여부,
    }])

    pred = int(model.predict(input_df)[0])
    proba = model.predict_proba(input_df)[0].tolist()

    label_map = {0: "정상", 1: "주의", 2: "고위험"}

    return {
        "판정": label_map[pred],
        "BMI": round(bmi, 1),
        "확률": {
            "정상": round(proba[0] * 100, 1),
            "주의": round(proba[1] * 100, 1),
            "고위험": round(proba[2] * 100, 1),
        }
    }
