import pandas as pd

# 파일 로드
submission = pd.read_csv('Q1.csv')
test = pd.read_csv('open/test.csv')

# submission의 두 번째 열 선택
target_column = submission.iloc[:, 1]

# test의 마지막 열에 추가
test['new_column'] = target_column.values

# 결과 저장
test.to_csv('result.csv', index=False)

print("result.csv 파일이 생성되었습니다.")
