import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows용)
plt.rc('font', family='Malgun Gothic')  # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

p = "C:/Users/User/OneDrive/문서/Gibbeum Project/건강검진 플랫폼 검진결과분석 모델링/국민건강보험공단_건강검진정보_2024.CSV"
df = pd.read_csv(p, encoding="cp949", low_memory=False)

missing = df.isna().sum().sort_values(ascending=False)
missing_ratio = (missing / len(df)).sort_values(ascending=False)
print(missing.head(30))
print(missing_ratio.head(30))

# 결측치 비율 바 차트
plt.figure(figsize=(12, 8))
missing_ratio.head(20).plot(kind='bar', color='skyblue')
plt.title('Top 20 결측치 비율')
plt.ylabel('결측 비율')
plt.xlabel('열 이름')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 결측치 히트맵 (샘플 1000행)
plt.figure(figsize=(12, 8))
sns.heatmap(df.head(1000).isna(), cbar=False, cmap='viridis')
plt.title('결측치 히트맵 (샘플 1000행)')
plt.show()
