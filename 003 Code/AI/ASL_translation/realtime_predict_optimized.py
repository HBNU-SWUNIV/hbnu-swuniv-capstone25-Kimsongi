import asyncio
import cv2
import mediapipe as mp
import multiprocessing as mproc
import numpy as np
from keras.models import load_model
import joblib
from collections import deque
from livekit import rtc
from PIL import ImageFont, ImageDraw, Image
import queue
import sys
import threading
from livekit_auth import create_token
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

# 설정값
SERVER_URL = "ws://127.0.0.1:7880"  # sfu 서버 ip 주소 대입
ACCESS_TOKEN = create_token("asl_worker", "dev-room")
MODEL_PATH = "/Users/kyungrim/Library/CloudStorage/GoogleDrive-20221999@edu.hanbat.ac.kr/내 드라이브/2025캡스톤프로젝트/ASL_lstm/models/gesture_lstm_model_dual_v2.h5"  # lstm 모델 파일 경로 대입
ENCODER_PATH = "/Users/kyungrim/Library/CloudStorage/GoogleDrive-20221999@edu.hanbat.ac.kr/내 드라이브/2025캡스톤프로젝트/ASL_lstm/label_encoder_lstm_dual.pkl"  # preprocess/*.pkl 파일 경로 대입
T5_MODEL_PATH = "/Users/kyungrim/Library/CloudStorage/GoogleDrive-20221999@edu.hanbat.ac.kr/내 드라이브/2025캡스톤프로젝트/KSL_t5/my_finetuned_t5_model"
FRAMES_PER_SEQUENCE = 30
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
CONFIDENCE_THRESHOLD = 0.85
PREDICTION_INTERVAL = 3

prediction_result = ("", 0.0)
sentence_words = []
sentence_lock = threading.Lock()
generated_sentence = ""
is_predicting = False

# 모델 및 리소스 로드
if mproc.current_process().name == "MainProcess":
    print("▶ 모델과 리소스 로드 중...")
    try:
        model = load_model(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("✅ LSTM 모델 로드 완료!")
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        sys.exit(1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    frame_buffer = deque(maxlen=FRAMES_PER_SEQUENCE)

# 한글 텍스트 출력 함수
_font_cache = {}
def draw_korean_text(img, text, position, font_size=32, color=(255, 255, 255), max_width=None):
    if font_size not in _font_cache:
        try:
            _font_cache[font_size] = ImageFont.truetype(FONT_PATH, font_size)
            print(f"[DEBUG] Loaded font at size {font_size} from {FONT_PATH}")
        except OSError as e:
            print(f"[DEBUG] Font load failed ({FONT_PATH}): {e}, using default font")
            _font_cache[font_size] = ImageFont.load_default()
    font = _font_cache[font_size]

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

# T5 문장 생성 프로세스 함수
def t5_worker(input_queue, output_queue, t5_model_path):
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer
    torch.set_num_threads(1)

    try:
        tokenizer = T5Tokenizer.from_pretrained(t5_model_path, local_files_only=True)
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path, local_files_only=True)
        print("✅ T5 모델 로드 완료!")
    except Exception as e:
        print(f"❌ T5 모델 로드 중 오류 발생: {e}")
        sys.exit(1)

    while True:
        words = input_queue.get()
        print(f"[DEBUG] T5 입력 단어 리스트: {words}")
        prompt = f"문장 생성: {', '.join(words)}"
        print(f"📝 T5 입력: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs.input_ids,
                max_length=64,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=2.0,
                no_repeat_ngram_size=2,
            )
        result_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_queue.put(result_sentence)
        print(f"✅ T5 생성 문장: {result_sentence}")

# LSTM 제스처 예측 스레드 함수
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
                t5_input_queue.put(list(sentence_words))
                sentence_words.clear()
        else:
            generated_sentence = ""
            sentence_words.append(predicted_label)
            print(f"➕ 단어 추가: {predicted_label} (현재 리스트: {sentence_words})")
            
    is_predicting = False

class VideoLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quit_requested = False
        self.reset_requested = False

    def keyPressEvent(self, event):
        # q 누르면 종료 요청
        if event.key() == Qt.Key_Q:
            print("[DEBUG] 'q' 입력(Qt) → 메인 루프 종료 요청")
            self.quit_requested = True
        # 스페이스바 누르면 초기화 요청
        elif event.key() == Qt.Key_Space:
            print("[DEBUG] 'space' 입력(Qt) → 초기화 요청")
            self.reset_requested = True
        else:
            super().keyPressEvent(event)

# --- LiveKit 루프 ---
room = None
frame_queue = queue.Queue(maxsize=1)
livekit_task = None
connection_failed = threading.Event()
sentence_queue = asyncio.Queue()
livekit_loop = None

async def receive_frames(stream, source_identity):
    """원격 비디오 프레임 수신"""
    print(f"[DEBUG] receive_frames 시작 (source_identity={source_identity})")
    frame_idx = 0
    async for event in stream:
        converted_frame = event.frame.convert(rtc.VideoBufferType.RGB24)
        image = np.frombuffer(converted_frame.data, dtype=np.uint8)
        image = image.reshape((converted_frame.height, converted_frame.width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
                print("[DEBUG] frame_queue 가득참 → 가장 오래된 프레임 제거")
            except queue.Empty:
                pass
        frame_queue.put_nowait(image)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[DEBUG] {source_identity} 로부터 {frame_idx}개 프레임 수신 중 "
                  f"({converted_frame.width}x{converted_frame.height})")

async def receive_from_livekit():
    global room
    print("[DEBUG] LiveKit Room 객체 초기화")
    room = rtc.Room()

    @room.on("participant_connected")
    def on_participant_connected(participant):
        print(f"[DEBUG] participant_connected: identity={participant.identity}, sid={participant.sid}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        global livekit_task
        print(f"[DEBUG] participant_disconnected: identity={participant.identity}, sid={participant.sid}")
        if participant.identity == "asl" and livekit_task is not None:
            print("[DEBUG] asl 참가자 연결 종료 → livekit_task cancel")
            livekit_task.cancel()
            livekit_task = None

    @room.on("track_published")
    def on_track_published(publication, participant):
        print(f"[DEBUG] track_published: participant={participant.identity}, "
              f"kind={publication.kind}, source={publication.source}")
        if participant.identity == "asl" and publication.kind == rtc.TrackKind.KIND_VIDEO:
            publication.set_subscribed(True)
            print("[DEBUG] asl 참가자의 비디오 트랙 publish 감지 → auto_subscribe 설정 전 개발자 로직")

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        global livekit_task
        print(f"[DEBUG] track_subscribed: participant={participant.identity}, "
              f"track.kind={track.kind}, publication.source={publication.source}")
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print(f"[DEBUG] {participant.identity} 비디오 트랙 구독 시작")
            video_stream = rtc.VideoStream(track)
            livekit_task = asyncio.create_task(receive_frames(video_stream, participant.identity))

    @room.on("disconnected")
    def on_disconnected(reason):
        print(f"[DEBUG] LiveKit disconnected: reason={reason}")
        connection_failed.set()

    try:
        print(f"[DEBUG] LiveKit 서버에 연결 시도: {SERVER_URL}")
        await room.connect(SERVER_URL, ACCESS_TOKEN, rtc.RoomOptions(auto_subscribe=False))
        print("[DEBUG] LiveKit 서버 연결 성공")
        print("[DEBUG] 현재 remote_participants 목록:")
        for p in room.remote_participants.values():
            print(f"    - identity={p.identity}, sid={p.sid}")
    except rtc.ConnectError as e:
        print(f"[DEBUG] LiveKit 서버 연결 실패: {e}")
        connection_failed.set()
        return

    # asl 참가자의 기존 publish된 트랙이 있으면 구독
    if "asl" in room.remote_participants:
        print("[DEBUG] remote_participants에 'asl' 발견 → 비디오 트랙 구독 시도")
        for publication in room.remote_participants["asl"].track_publications.values():
            print(f"[DEBUG] asl track_publication: kind={publication.kind}, source={publication.source}")
            if publication.kind == rtc.TrackKind.KIND_VIDEO:
                publication.set_subscribed(True)
                print("[DEBUG] asl 비디오 트랙 set_subscribed(True)")
                break
    else:
        print("[DEBUG] 현재 'asl' 참가자가 없음 (핸드폰/AVD에서 아직 방에 안 들어온 상태일 수 있음)")

    asyncio.create_task(send_message_loop(room))
    print("[DEBUG] receive_from_livekit 완료, 이벤트 루프 대기 상태로 진입")
    await asyncio.Future()  # 무한 대기

async def send_message_loop(room):
    print("[DEBUG] send_message_loop 시작")
    while True:
        sentence = await sentence_queue.get()
        data = sentence.encode()
        print(f"[DEBUG] LiveKit 데이터 채널로 문장 전송: '{sentence}' -> destination ['ksl']")
        await room.local_participant.publish_data(data, destination_identities=["ksl"])

def run_livekit_background():
    global livekit_loop
    print("[DEBUG] LiveKit 백그라운드 스레드 시작")
    livekit_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(livekit_loop)
    try:
        livekit_loop.run_until_complete(receive_from_livekit())
    finally:
        livekit_loop.close()
        print("[DEBUG] LiveKit 백그라운드 이벤트 루프 종료")

# --- 메인 루프 ---
if __name__ == "__main__":
    app = QApplication([])
    qt_label = VideoLabel()
    qt_label.setWindowTitle("ASL Translator")
    qt_label.resize(960, 720)
    qt_label.setFocusPolicy(Qt.StrongFocus)  # 키보드 포커스 받을 수 있게
    qt_label.show()
    qt_label.setFocus()  # 실행 직후 바로 포커스 주기

    mproc.freeze_support()
    mproc.set_start_method("spawn", force=True)

    print("[DEBUG] T5 워커 프로세스 생성")
    t5_input_queue = mproc.Queue()
    t5_output_queue = mproc.Queue()
    mproc.Process(target=t5_worker, args=(t5_input_queue, t5_output_queue, T5_MODEL_PATH), daemon=True).start()

    mode = int(input("모드 선택 (1: 로컬 카메라, 2: 서버 모니터링) - "))
    print(f"[DEBUG] 선택한 모드: {mode}")
    if not 1 <= mode <= 2:
        print("잘못된 입력입니다.")
        sys.exit(1)

    if mode == 1:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
            sys.exit(1)
        print("[DEBUG] 로컬 카메라 모드: 웹캠 열기 성공")
    elif mode == 2:
        print("[DEBUG] 서버 모니터링 모드: LiveKit 백그라운드 스레드 시작")
        threading.Thread(target=run_livekit_background, daemon=True).start()

    print("▶ 실시간 수어 번역을 시작합니다. ('q': 종료, '스페이스바': 초기화)")
    frame_count = 0
    last_frame = np.zeros((640, 480, 3), dtype=np.uint8)

    while True:
        if mode == 1:
            if not cap.isOpened():
                print("[DEBUG] 카메라가 더 이상 열려 있지 않음, 루프 종료")
                break
            ret, frame = cap.read()
            if not ret:
                print("[DEBUG] 카메라 프레임 읽기 실패, 루프 종료")
                break
        elif mode == 2:
            if connection_failed.is_set():
                print("SFU 서버에 연결할 수 없습니다.")
                print("[DEBUG] connection_failed 플래그가 set됨, 루프 종료")
                break

            try:
                frame = frame_queue.get(timeout=0.1)
                last_frame = frame
            except queue.Empty:
                frame = last_frame

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
                    normalized_left[i * 3:i * 3 + 3] = [
                        lm.x - left_wrist.x,
                        lm.y - left_wrist.y,
                        lm.z - left_wrist.z,
                    ]

            normalized_right = [0.0] * 63
            if hand_data["Right"]:
                right_wrist = hand_data["Right"][0]
                for i, lm in enumerate(hand_data["Right"]):
                    normalized_right[i * 3:i * 3 + 3] = [
                        lm.x - right_wrist.x,
                        lm.y - right_wrist.y,
                        lm.z - right_wrist.z,
                    ]

            one_frame = normalized_left + normalized_right
            frame_buffer.append(one_frame)
        else:
            frame_buffer.clear()

        frame_count += 1
        if len(frame_buffer) == FRAMES_PER_SEQUENCE and not is_predicting and frame_count % PREDICTION_INTERVAL == 0:
            is_predicting = True
            sequence_data = np.array(frame_buffer).reshape(1, FRAMES_PER_SEQUENCE, 126)
            threading.Thread(target=predict_gesture, args=(sequence_data,), daemon=True).start()

        try:
            current_sentence = t5_output_queue.get_nowait()
            print(f"[DEBUG] T5 워커로부터 새 문장 수신: {current_sentence}")
            if mode == 2 and livekit_loop is not None:
                print(f"[DEBUG] LiveKit 전송 큐에 문장 추가: {current_sentence}")
                asyncio.run_coroutine_threadsafe(sentence_queue.put(current_sentence), livekit_loop)
            generated_sentence = current_sentence
        except queue.Empty:
            current_sentence = generated_sentence

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        label, conf = prediction_result
        feedback_text = f"Guess: {label} ({conf:.2f})"
        color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (255, 0, 0)
        frame = draw_korean_text(frame, feedback_text, (10, 30), font_size=32, color=color)

        with sentence_lock:
            words_text = " ".join(sentence_words)
        frame = draw_korean_text(
            frame,
            f"입력: {words_text}",
            (10, 80),
            font_size=40,
            color=(255, 235, 59),
            max_width=frame.shape[1] - 20,
        )

        if current_sentence:
            frame = draw_korean_text(
                frame,
                f"결과: {current_sentence}",
                (10, 130),
                font_size=40,
                color=(129, 212, 250),
                max_width=frame.shape[1] - 20,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = w * ch

        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled = qimg.scaled(960, 720, Qt.KeepAspectRatio)
        qt_label.setPixmap(QPixmap.fromImage(scaled))
        app.processEvents()

        # 🔹 여기서 Qt로 받은 키 입력 플래그 확인
        if qt_label.quit_requested:
            print("[DEBUG] Qt 'q' 입력 → 메인 루프 종료")
            break

        if qt_label.reset_requested:
            sentence_words.clear()
            generated_sentence = ""
            prediction_result = ("", 0.0)
            print("🔄 문장 및 단어 목록이 초기화되었습니다.")
            qt_label.reset_requested = False  # 한 번 처리했으니 다시 False로
        

    if mode == 1:
        cap.release()
        print("[DEBUG] 로컬 카메라 리소스 해제 완료")
    elif mode == 2 and room is not None and livekit_loop is not None:
        print("[DEBUG] LiveKit room.disconnect 요청")
        asyncio.run_coroutine_threadsafe(room.disconnect(), livekit_loop).result()
    cv2.destroyAllWindows()
    print("[DEBUG] OpenCV 윈도우 닫기 완료")