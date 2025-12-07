import os
import glob
import pandas as pd

DATA_DIR = "./ASL_translation/data"   # CSV 모여 있는 폴더

csv_paths = glob.glob(os.path.join(DATA_DIR, "*_sequences.csv"))

for path in csv_paths:
    # 파일명에서 라벨 추출
    filename = os.path.basename(path)
    label = filename.replace("_sequences.csv", "")  # 예: "로 가다"

    print(f"[INFO] 처리 중: {filename}  →  label='{label}'")

    # CSV 로드
    df = pd.read_csv(path)

    # 마지막 열 이름이 label이라고 가정
    df["label"] = label

    # 그대로 덮어쓰기
    df.to_csv(path, index=False, encoding="utf-8-sig")

print("✅ 모든 CSV의 label 컬럼이 한국어로 업데이트 완료!")