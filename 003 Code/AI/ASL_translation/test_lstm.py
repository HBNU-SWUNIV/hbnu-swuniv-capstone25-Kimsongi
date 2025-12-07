# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import joblib
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ==========================
# 설정
# ==========================
MODEL_PATH = "/Users/kyungrim/Library/CloudStorage/GoogleDrive-20221999@edu.hanbat.ac.kr/내 드라이브/2025캡스톤프로젝트/ASL_lstm/models/gesture_lstm_model_dual_small_v1.h5"
ENCODER_PATH = "/Users/kyungrim/Library/CloudStorage/GoogleDrive-20221999@edu.hanbat.ac.kr/내 드라이브/2025캡스톤프로젝트/ASL_lstm/processed_lstm/label_encoder_lstm_dual.pkl"
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

FRAMES_PER_SEQUENCE = 30        # 학습 때도 30프레임 사용
CONFIDENCE_THRESHOLD = 0.75     # 화면에 초록색으로 표시할 기준
PREDICTION_INTERVAL = 3         # 몇 프레임마다 한 번씩만 예측할지
IDLE_CLEAR_FRAMES = 5           # 손이 안 보이는 프레임이 이만큼 쌓이면 버퍼/예측 리셋

# ==========================
# 한글 출력 함수
# ==========================
_font_cache = {}

def draw_korean_text(img, text, position, font_size=32, color=(255, 255, 255), max_width=None):
    if font_size not in _font_cache:
        _font_cache[font_size] = ImageFont.truetype(FONT_PATH, font_size)

    font = _font_cache[font_size]

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    x, y = position
    line_height = font_size + 5
    if max_width is None:
        max_width = img.shape[1] - x - 20

    words = text.split()
    current_line = []
    current_y = y

    for w in words:
        test_line = ' '.join(current_line + [w])
        bbox = font.getbbox(test_line)
        text_width = bbox[2] - bbox[0]

        if text_width < max_width:
            current_line.append(w)
        else:
            draw.text((x, current_y), ' '.join(current_line), font=font, fill=color)
            current_y += line_height
            current_line = [w]

    if current_line:
        draw.text((x, current_y), ' '.join(current_line), font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ===============================================
# 🔥 훈련 데이터와 동일한 방식의 landmark → feature 함수
# ===============================================
def extract_one_frame(results):
    """
    Mediapipe 결과에서 한 프레임(30 x 126 중 126 부분)을 만드는 함수.
    - 왼손/오른손 각각 21개 랜드마크 x (x,y,z) = 63
    - 왼손 63 + 오른손 63 = 126
    - 손목 기준 상대좌표로 정규화 (train 코드와 동일)
    """
    hand_data = {"Left": None, "Right": None}
    hand_detected = {"Left": False, "Right": False}

    if results.multi_handedness and results.multi_hand_landmarks:
        for lm_list, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handed.classification[0].label  # "Left" or "Right"

            coords = []
            for lm in lm_list.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            hand_data[label] = coords
            hand_detected[label] = True

    # --- 왼손 정규화 ---
    left_norm = [0.0] * 63
    if hand_detected["Left"]:
        left_np = np.array(hand_data["Left"]).reshape(21, 3)
        wrist = left_np[0]
        rel = left_np - wrist
        left_norm = rel.flatten().tolist()

    # --- 오른손 정규화 ---
    right_norm = [0.0] * 63
    if hand_detected["Right"]:
        right_np = np.array(hand_data["Right"]).reshape(21, 3)
        wrist = right_np[0]
        rel = right_np - wrist
        right_norm = rel.flatten().tolist()

    return left_norm + right_norm

# ==========================
# 모델 로드
# ==========================
print("📦 모델 로딩 중...")
model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
print("✅ 모델 및 LabelEncoder 로드 완료!")

# ==========================
# Mediapipe 초기화
# ==========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ==========================
# 30프레임 버퍼
# ==========================
frame_buffer = deque(maxlen=FRAMES_PER_SEQUENCE)

# ==========================
# 카메라 시작
# ==========================
cap = cv2.VideoCapture(1)   # 필요하면 0으로 바꿔서 다른 카메라 테스트
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("▶ 실시간 LSTM 테스트 시작 ('q' 종료)")

prediction_result = ("", 0.0)
is_predicting = False
frame_count = 0
no_hand_frames = 0   # 손이 안 잡힌 프레임 카운터

# ==========================
# 메인 루프
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # === 손 인식 여부에 따라 처리 ===
    if results.multi_hand_landmarks:
        # 화면에 랜드마크 그려주기 (디버깅용)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 손이 보이면 버퍼에 프레임 추가
        one_frame = extract_one_frame(results)
        frame_buffer.append(one_frame)

        # 손 보이기 시작했으니 idle 카운터 리셋
        no_hand_frames = 0
    else:
        # 손이 안 보이는 프레임 누적
        no_hand_frames += 1

        # 일정 프레임 이상 손이 안 보이면 완전 idle 처리
        if no_hand_frames >= IDLE_CLEAR_FRAMES:
            frame_buffer.clear()
            prediction_result = ("", 0.0)  # 화면에서 예측 텍스트 제거
            # 여기서 바로 no_hand_frames 계속 올려도 되지만,
            # 이미 idle 상태라 사실 큰 의미는 없음

    frame_count += 1

    # === 예측 (손이 보이고, 버퍼가 꽉 찼고, 일정 주기마다만) ===
    if (
        len(frame_buffer) == FRAMES_PER_SEQUENCE
        and not is_predicting
        and no_hand_frames == 0                    # 바로 직전에도 손이 보였을 때만
        and frame_count % PREDICTION_INTERVAL == 0 # 예측 주기 제어
    ):
        is_predicting = True

        seq = np.array(frame_buffer).reshape(1, FRAMES_PER_SEQUENCE, 126).astype("float32")
        pred = model.predict(seq, verbose=0)
        conf = float(np.max(pred))
        idx = int(np.argmax(pred))
        label = label_encoder.inverse_transform([idx])[0]

        prediction_result = (label, conf)
        is_predicting = False

    # === 화면 표시 ===
    label, conf = prediction_result
    if label:
        text = f"Predict: {label} ({conf:.2f})"
    else:
        text = "Predict: (none)"

    color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (255, 0, 0)
    frame = draw_korean_text(frame, text, (10, 30), font_size=40, color=color)

    cv2.imshow("LSTM Sign Test", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()