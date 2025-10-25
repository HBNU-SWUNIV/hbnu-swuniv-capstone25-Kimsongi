import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import joblib
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import threading
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer

# --- 💡 설정값 (가장 중요한 부분!) ---
# [수정] 데이터 추가 후 새로 훈련한 최신 모델 경로로 변경합니다.
MODEL_PATH = "models/gesture_lstm_model_dual_v4.h5" 
# [수정] preprocess 스크립트에서 저장한 파일명과 동일하게 맞춥니다.
ENCODER_PATH = "processed_lstm/label_encoder_lstm_dual.pkl"
T5_MODEL_PATH = "./my_finetuned_t5_model"
FRAMES_PER_SEQUENCE = 30 
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
CONFIDENCE_THRESHOLD = 0.75
PREDICTION_INTERVAL = 3

# --- 💡 전역 변수 ---
prediction_result = ("", 0.0)
sentence_words = []
generated_sentence = ""
is_predicting = False

# --- 모델 및 리소스 로드 ---
print("▶ 모델과 리소스 로드 중...")
try:
    model = load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)
    print("✅ 모든 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    print("   - preprocess와 train 스크립트가 먼저 실행되었는지 확인해주세요.")
    print(f"   - 필요한 파일: {MODEL_PATH}, {ENCODER_PATH}")
    exit()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

frame_buffer = deque(maxlen=FRAMES_PER_SEQUENCE)

# --- 한글 텍스트 출력 함수 ---
def draw_korean_text(img, text, position, font_size=32, color=(255, 255, 255), max_width=None):
    font = ImageFont.truetype(FONT_PATH, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    x, y = position
    line_height = font_size + 5
    if max_width is None:
        max_width = img.shape[1] * 0.9 - x

    words = text.split(' ')
    current_line = []
    current_y = y

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        text_width = bbox[2] - bbox[0]
        if text_width < max_width:
            current_line.append(word)
        else:
            draw.text((x, current_y), ' '.join(current_line), font=font, fill=color)
            current_y += line_height
            current_line = [word]
    if current_line:
        draw.text((x, current_y), ' '.join(current_line), font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- T5 문장 생성 스레드 함수 ---
def generate_sentence_with_t5(words):
    global generated_sentence
    if not words: return
    
    prompt = f"문장 생성: {', '.join(words)}"
    print(f"📝 T5 입력: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = t5_model.generate(
            inputs.input_ids, max_length=64, num_beams=5, early_stopping=True,
            repetition_penalty=2.0, no_repeat_ngram_size=2
        )
    result_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_sentence = result_sentence
    print(f"✅ T5 생성 문장: {generated_sentence}")

# --- LSTM 제스처 예측 스레드 함수 ---
def predict_gesture(sequence_data):
    global prediction_result, is_predicting, sentence_words, generated_sentence
    
    prediction = model.predict(sequence_data, verbose=0)
    confidence = np.max(prediction)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    
    prediction_result = (predicted_label, confidence)
    
    if confidence >= CONFIDENCE_THRESHOLD and (not sentence_words or sentence_words[-1] != predicted_label):
        if predicted_label == "OK":
            if sentence_words:
                threading.Thread(target=generate_sentence_with_t5, args=(list(sentence_words),), daemon=True).start()
                sentence_words.clear()
        else:
            generated_sentence = ""
            sentence_words.append(predicted_label)
            print(f"➕ 단어 추가: {predicted_label} (현재 리스트: {sentence_words})")
            
    is_predicting = False

# --- 메인 루프 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다."); exit()

print("▶ 실시간 수어 번역을 시작합니다. ('q': 종료, '스페이스바': 초기화)")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_data = {"Left": [], "Right": []}
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            hand_data[hand_label] = [lm for lm in hand_landmarks.landmark]

        normalized_left = [0.0] * 63
        if hand_data["Left"]:
            left_wrist = hand_data["Left"][0]
            for i, lm in enumerate(hand_data["Left"]):
                normalized_left[i*3:i*3+3] = [lm.x - left_wrist.x, lm.y - left_wrist.y, lm.z - left_wrist.z]

        normalized_right = [0.0] * 63
        if hand_data["Right"]:
            right_wrist = hand_data["Right"][0]
            for i, lm in enumerate(hand_data["Right"]):
                normalized_right[i*3:i*3+3] = [lm.x - right_wrist.x, lm.y - right_wrist.y, lm.z - right_wrist.z]
        
        one_frame = normalized_left + normalized_right
        frame_buffer.append(one_frame)
    else:
        frame_buffer.clear()

    frame_count += 1
    if len(frame_buffer) == FRAMES_PER_SEQUENCE and not is_predicting and frame_count % PREDICTION_INTERVAL == 0:
        is_predicting = True
        sequence_data = np.array(frame_buffer).reshape(1, FRAMES_PER_SEQUENCE, 126)
        threading.Thread(target=predict_gesture, args=(sequence_data,), daemon=True).start()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    label, conf = prediction_result
    feedback_text = f"Guess: {label} ({conf:.2f})"
    color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (255, 0, 0)
    frame = draw_korean_text(frame, feedback_text, (10, 30), font_size=32, color=color)

    words_text = " ".join(sentence_words)
    frame = draw_korean_text(frame, f"입력: {words_text}", (10, 80), font_size=40, color=(255, 235, 59), max_width=frame.shape[1] - 20)
    
    if generated_sentence:
        frame = draw_korean_text(frame, f"결과: {generated_sentence}", (10, 130), font_size=40, color=(129, 212, 250), max_width=frame.shape[1] - 20)

    cv2.imshow("Sign Language Translator", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        sentence_words.clear()
        generated_sentence = ""
        prediction_result = ("", 0.0)
        print("🔄 문장 및 단어 목록이 초기화되었습니다.")

cap.release()
cv2.destroyAllWindows()

