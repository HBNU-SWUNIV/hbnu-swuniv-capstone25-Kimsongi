# 필요한 라이브러리들을 불러옵니다.
import numpy as np
import os
import datetime
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

# --- ⚙️ 설정: 이 스위치로 모드를 변경하세요 ---
# True: model_save_path에 있는 모델을 불러와서 이어서 학습하거나, 구조를 변경하여 전이 학습합니다.
# False: 기존처럼 새로운 모델을 처음부터 학습합니다.
CONTINUE_TRAINING = False
# -----------------------------------------

# 학습 과정을 그래프로 그려주는 함수
def plot_training_history(history, model_save_path):
    """훈련 결과를 받아 정확도와 손실 그래프를 그리고 저장합니다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 정확도 그래프
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # 손실 그래프
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # 그래프를 이미지 파일로 저장
    save_path = os.path.splitext(model_save_path)[0] + '_history.png'
    plt.savefig(save_path)
    print(f"📊 학습 과정 그래프가 '{save_path}'에 저장되었습니다.")
    # plt.show() # 노트북 환경 확인용

def train_lstm_model_dual(X_path, y_path, model_save_path):
    """
    전처리된 양손 데이터를 사용하여 LSTM 모델을 훈련하고, 성능을 분석 및 시각화합니다.
    - CONTINUE_TRAINING=True일 때, 클래스 개수가 같으면 이어서 학습하고, 다르면 전이 학습을 통해 지식을 이식합니다.
    """
    X = np.load(X_path)
    y = np.load(y_path)
    new_num_classes = y.shape[1]

    print(f"🔹 X shape: {X.shape}")
    print(f"🔹 y shape: {y.shape} (새로운 클래스 개수: {new_num_classes})")

    y_classes = np.argmax(y, axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(X, y_classes))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    if CONTINUE_TRAINING and os.path.exists(model_save_path):
        print(f"📖 기존 모델 '{model_save_path}'을(를) 불러옵니다.")
        old_model = load_model(model_save_path)
        old_num_classes = old_model.layers[-1].units
        print(f"   - 기존 모델의 클래스 개수: {old_num_classes}")

        if old_num_classes == new_num_classes:
            print("   - 클래스 개수가 동일하여, 모델을 그대로 이어서 학습합니다.")
            model = old_model
        else:
            print(f"   - ⚠️ 클래스 개수가 {old_num_classes} -> {new_num_classes} (으)로 변경되었습니다.")
            print("   - 전이 학습을 시작합니다: 기존 모델의 지식을 새로운 구조의 모델로 이식합니다.")
            
            model = Sequential([
                LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(128, return_sequences=True, activation='relu'),
                Dropout(0.2),
                LSTM(64, return_sequences=False, activation='relu'),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(new_num_classes, activation='softmax')
            ])
            
            for i, layer in enumerate(model.layers[:-1]):
                if len(layer.get_weights()) > 0 and len(old_model.layers[i].get_weights()) > 0:
                     layer.set_weights(old_model.layers[i].get_weights())

            print("   - 지식 이식이 완료되었습니다.")
    else:
        print("✨ 새로운 모델을 처음부터 학습합니다.")
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(128, return_sequences=True, activation='relu'),
            Dropout(0.2),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(new_num_classes, activation='softmax')
        ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    print("\n▶ 모델 학습을 시작합니다.")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )

    print("\n▶ 최종 모델 성능을 평가합니다.")
    best_model = load_model(model_save_path)
    loss, accuracy = best_model.evaluate(X_val, y_val, verbose=0)
    print(f"   - 최종 검증 손실 (Final Validation Loss): {loss:.4f}")
    print(f"   - 최종 검증 정확도 (Final Validation Accuracy): {accuracy:.4f}")

    # 학습 과정 시각화 함수 호출
    plot_training_history(history, model_save_path)

    print(f"\n✅ 양손 LSTM 모델 학습 및 모든 과정 완료: {model_save_path}")

if __name__ == "__main__":
    train_lstm_model_dual(
        X_path="processed_lstm/X_seq_lstm_dual.npy",
        y_path="processed_lstm/y_seq_lstm_dual.npy",
        model_save_path="models/gesture_lstm_model_dual_v4.h5"
    )

