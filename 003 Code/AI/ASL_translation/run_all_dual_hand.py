# 데이터 전처리, 모델 학습, 인식용 카메라 통합 실행 코드

import subprocess
import sys

# 경로 설정
preprocess_script = "preprocess_multiple_csv_lstm.py"
train_script = "train_lstm_model.py"
realtime_script = "realtime_predict_optimized.py"

def run_script(script_path):
    print(f"\n🟢 실행 중: {script_path}")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"❌ 오류 발생: {script_path}")
        exit(1)

if __name__ == "__main__":
    print("=== [1/3] 양손 CSV 전처리 시작 ===")
    run_script(preprocess_script)

    print("=== [2/3] LSTM 모델 학습 시작 ===")
    run_script(train_script)

    print("=== [3/3] 실시간 예측 시작 ===")
    run_script(realtime_script)
