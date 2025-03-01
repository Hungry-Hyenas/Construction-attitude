# 필요한 라이브러리 다시 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
# import ace_tools_open as tsools

# 데이터 로드
train = pd.read_csv("./open/train.csv", encoding="utf-8-sig")

# Train 데이터를 8:2로 분할
train_data_8, val_data_2 = train_test_split(train, test_size=0.2, random_state=42)

# 파일 저장
train_data_8.to_csv("train_8.csv", index=False, encoding="utf-8-sig")
val_data_2.to_csv("validation.csv", index=False, encoding="utf-8-sig")

# 결과 확인
# tools.display_dataframe_to_user(name="Train 8 Data", dataframe=train_data_8)
# tools.display_dataframe_to_user(name="Validation Data", dataframe=val_data_2)
