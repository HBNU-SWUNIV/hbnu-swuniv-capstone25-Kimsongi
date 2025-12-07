# 손 인식 학습을 위한 데이터 수집 코드

import cv2
import mediapipe as mp
import csv
import os
import numpy as np

def collect_hand_landmark_samples(
    label,
    save_path,
    frames_per_sample=30,
    samples_per_video=25
):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    mp_drawing = mp.solutions.drawing_utils

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    frame_buffer = []
    sample_count = 0
    file_exists = os.path.isfile(save_path)

    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # CSV 헤더가 없으면 생성 (헤더 내용은 동일)
        if not file_exists:
            header = []
            for i in range(frames_per_sample):
                for j in range(126):
                    header.append(f"f{i}_v{j}")
            header.append("label")
            writer.writerow(header)

        print("▶ 두 손 실시간 수집 시작 (30프레임 x 5샘플)")

        while cap.isOpened() and sample_count < samples_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # 두 손 데이터 초기화
            hand_data = {"Left": [0.0] * 63, "Right": [0.0] * 63}
            hand_detected = {"Left": False, "Right": False}

            if results.multi_hand_landmarks and results.multi_handedness:
                # 1. 모든 랜드마크의 절대 좌표 수집
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'
                    
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    
                    hand_data[hand_label] = coords
                    hand_detected[hand_label] = True

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 2. 손목 기준 상대 좌표로 정규화
                normalized_left_coords = [0.0] * 63
                normalized_right_coords = [0.0] * 63

                # 왼손이 감지되었을 경우
                if hand_detected["Left"]:
                    left_hand_np = np.array(hand_data["Left"]).reshape(21, 3)
                    left_wrist = left_hand_np[0] # 0번 랜드마크가 손목
                    relative_left = left_hand_np - left_wrist
                    normalized_left_coords = relative_left.flatten().tolist()

                # 오른손이 감지되었을 경우
                if hand_detected["Right"]:
                    right_hand_np = np.array(hand_data["Right"]).reshape(21, 3)
                    right_wrist = right_hand_np[0] # 0번 랜드마크가 손목
                    relative_right = right_hand_np - right_wrist
                    normalized_right_coords = relative_right.flatten().tolist()
                
                # 3. 정규화된 좌표를 하나의 프레임으로
                one_frame = normalized_left_coords + normalized_right_coords


                frame_buffer.append(one_frame)

                if len(frame_buffer) == frames_per_sample:
                    sample = [val for frame in frame_buffer for val in frame]
                    sample.append(label)
                    writer.writerow(sample)
                    sample_count += 1
                    frame_buffer = []
                    print(f"✔ 저장됨: 샘플 {sample_count}/{samples_per_video}")

                    print("⏸ 다음 샘플을 수집하려면 아무 키나 누르세요...")
                    while True:
                        cv2.imshow("Dual Hand Sample Collector", frame)
                        if cv2.waitKey(0) != -1:
                            break
            else:
                frame_buffer = []

            cv2.imshow("Dual Hand Sample Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 수집 중단됨")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ 전체 {sample_count}개 샘플 저장 완료 → {save_path}")


if __name__ == "__main__":
    label = "고양이"
    collect_hand_landmark_samples(
        label=f"{label}",
        save_path=f"ASL_translation/data/{label}_sequences.csv"
    )