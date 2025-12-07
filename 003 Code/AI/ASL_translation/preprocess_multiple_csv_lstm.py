# 수집된 데이터 전처리 코드

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib

def preprocess_dual_hand_csv_lstm(csv_folder, save_dir):
    """
    여러 CSV 파일들을 읽어 LSTM 모델 학습을 위한 데이터셋(.npy)과 LabelEncoder(출석부, .pkl)를 생성
    """
    all_X, all_y = [], []
    print("▶ 데이터 전처리 작업을 시작합니다...")

    for file in sorted(os.listdir(csv_folder)): # 파일 이름 순으로 정렬하여 일관성 유지
        if not file.endswith(".csv"):
            continue

        label = file.replace("_sequences.csv", "").replace(".csv", "")
        file_path = os.path.join(csv_folder, file)
        df = None

        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            try:
                print(f"⚠️ INFO: '{file}' 파일의 인코딩이 UTF-8이 아닙니다. CP949로 다시 시도합니다...")
                df = pd.read_csv(file_path, encoding='cp949')
            except Exception as e:
                print(f"❌ ERROR: '{file}' 파일을 읽는 데 실패했습니다. 건너뜁니다. 원인: {e}")
                continue
        except Exception as e:
            print(f"❌ ERROR: '{file}' 파일을 읽는 중 예기치 않은 오류가 발생했습니다. 건너뜁니다. 원인: {e}")
            continue

        # 데이터(X) 추출
        if 'label' in df.columns:
            X = df.drop('label', axis=1).values.astype(np.float32)
        else:
            # 헤더가 없는 경우 모든 열을 데이터로 간주
            X = df.values.astype(np.float32)

        # 데이터 유효성 검사
        if X.shape[0] == 0:
            print(f"   - WARNING: '{file}' 파일에 데이터가 없습니다. 건너뜁니다.")
            continue
        
        # 각 시퀀스(행)가 3780개의 특징을 가지는지 확인
        if X.shape[1] != 3780: 
            print(f"   - WARNING: '{file}' 파일의 열 개수가 3780이 아닙니다 (현재: {X.shape[1]}). 건너뜁니다.")
            continue

        # 데이터를 (시퀀스 수, 프레임 수, 특징 수) 형태로 재구성
        num_sequences = X.shape[0]
        X = X.reshape(num_sequences, 30, 126) # (len(df), 30, 126)과 동일
        y = [label] * num_sequences
        
        all_X.append(X)
        all_y.extend(y)
        print(f"   - '{file}' 처리 완료 ({num_sequences}개 시퀀스 추가)")

    if not all_X:
        print("❌ 처리된 데이터가 없습니다. 'data' 폴더에 유효한 CSV 파일이 있는지 확인해주세요.")
        return

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.array(all_y)

    
    # 1. LabelEncoder를 생성하고 모든 단어로 학습
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all)

    # 2. 학습이 완료된 Encoder를 파일로 저장
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = os.path.join(save_dir, "label_encoder_lstm_dual.pkl")
    joblib.dump(label_encoder, encoder_path)
    
    print("\n✅ Label Encoder(출석부)가 성공적으로 저장되었습니다.")
    print(f"   - 저장 경로: {encoder_path}")
    print(f"   - 총 단어 개수: {len(label_encoder.classes_)}")
    print(f"   - 단어 목록: {label_encoder.classes_}")

    # 3. 레이블을 원-핫 인코딩으로 변환
    y_onehot = to_categorical(y_encoded)

    # 4. 최종 데이터셋을 .npy 파일로 저장
    np.save(os.path.join(save_dir, "X_seq_lstm_dual.npy"), X_all)
    np.save(os.path.join(save_dir, "y_seq_lstm_dual.npy"), y_onehot)

    print(f"\n✅ 데이터 전처리 최종 완료: 총 {X_all.shape[0]}개의 샘플이 생성되었습니다.")
    print(f"   - X 데이터 shape: {X_all.shape}")
    print(f"   - y 데이터 shape: {y_onehot.shape}")


if __name__ == "__main__":
    preprocess_dual_hand_csv_lstm(
        csv_folder="data",
        save_dir="processed_lstm"
    )

