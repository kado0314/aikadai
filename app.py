
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import random
import os
from PIL import Image
import cv2
import numpy as np
import math

# --- 関数群 ---

def calc_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    norm_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    norm_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot / (norm_ba * norm_bc + 1e-6)
    angle = math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))
    return angle

def mediapipe_reset():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_hands, mp_drawing, hands

def camera_input():
    return st.camera_input("📸 指の本数をカウントする写真を撮ってください")

def mediapipe_prosess(uploaded_file, mp_hands, mp_drawing, hands):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        total_finger_count = 0
        annotated_img = img.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                landmarks = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([id, cx, cy])

                if landmarks:
                    fingers = 0
                    mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP][1:3]
                    ip  = landmarks[mp_hands.HandLandmark.THUMB_IP][1:3]
                    tip = landmarks[mp_hands.HandLandmark.THUMB_TIP][1:3]
                    thumb_angle = calc_angle(mcp, ip, tip)
                    if thumb_angle > 160:
                        fingers += 1

                    if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][2] < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP][2]:
                        fingers += 1
                    if landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][2] < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP][2]:
                        fingers += 1
                    if landmarks[mp_hands.HandLandmark.RING_FINGER_TIP][2] < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP][2]:
                        fingers += 1
                    if landmarks[mp_hands.HandLandmark.PINKY_TIP][2] < landmarks[mp_hands.HandLandmark.PINKY_PIP][2]:
                        fingers += 1

                    total_finger_count += fingers
        return total_finger_count
    else:
        return 0

def camera():
    mp_hands, mp_drawing, hands = mediapipe_reset()
    uploaded_file = camera_input()
    if uploaded_file is not None:
        total_finger_count = mediapipe_prosess(uploaded_file, mp_hands, mp_drawing, hands)
        if total_finger_count > 0:
            st.success(f"画像内で**{total_finger_count}本**の指が伸びていると検出しました！")
            if st.button("✅ この結果で次へ進む"):
                return total_finger_count
        else:
            st.warning("伸びている指は検出されませんでした。もう一度試してください。")
    return None

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=False)
def load_diffusion_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "Lykon/dreamshaper-6",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True
    )
    return pipe.to(device)

@st.cache_resource(show_spinner=False)
def load_yolov5_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects_yolov5(model, image_path):
    results = model(image_path)
    df = results.pandas().xyxy[0]
    return len(df)

def reset_game_state():
    keys_to_reset = [
        "v001_fruit", "v001_num", "v001_image_path", "v001_detected_count",
        "v001_generated", "v001_result_shown", "v001_camera1_count", "v001_camera2_count",
        "v001_guess", "v001_photo1_taken", "v001_photo2_taken", "v001_answer_checked",
        "last_result_message", "last_result_type", "last_actual_num_message"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def camera_input_widget(label, key):
    return st.camera_input(label, key=key)

def mediapipe_process(uploaded_file, mp_hands, mp_drawing, hands):
    return mediapipe_prosess(uploaded_file, mp_hands, mp_drawing, hands)

# --- 手話モード用関数 ---
def hand_sign_mode():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    st.title("✋ 指の本数をカウントして挨拶クイズ")

    def count_fingers(hand_landmarks):
        finger_count = 0
        landmarks = hand_landmarks.landmark

        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y < landmarks[pip].y:
                finger_count += 1

        if landmarks[4].x < landmarks[3].x:
            finger_count += 1

        return finger_count

    def classify_greeting(count):
        if count == 1:
            return "こんにちは"
        elif count == 5:
            return "ありがとう"
        elif count == 0:
            return "バイバイ"
        else:
            return "認識できませんでした"

    uploaded = st.camera_input("📸 指の写真を撮ってください")

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            count = count_fingers(hand_landmarks)
            greeting = classify_greeting(count)

            st.image(img, caption=f"指の本数: {count}本", channels="BGR")
            st.session_state.quiz_greeting = greeting
            st.session_state.quiz_answered = False
            st.info("この手話は何の挨拶でしょうか？ 下の入力欄に答えてください。")
        else:
            st.warning("手が検出されませんでした。もう一度撮ってください。")

    if "quiz_greeting" in st.session_state and not st.session_state.get("quiz_answered", False):
        user_input = st.text_input("挨拶を入力してください（例：こんにちは、ありがとう、バイバイ）")

        if st.button("回答する"):
            correct_answer = st.session_state.quiz_greeting
            if user_input == correct_answer:
                st.success(f"正解！この手話は「{correct_answer}」です。")
            else:
                st.error(f"不正解！正解は「{correct_answer}」です。")

            st.session_state.quiz_answered = True

# --- メイン画面 ---
screen = st.sidebar.selectbox("", ["数当て", "手話","数当てver0.0.1"])

if screen == "数当て":
    fruit_options = ["りんご", "みかん"]
    if "fruit" not in st.session_state:
        st.session_state.fruit = random.choice(fruit_options)
        st.session_state.image_path = None
        st.session_state.generated = False
        st.session_state.result_shown = False

    pipe = load_diffusion_model()
    yolo_model = load_yolov5_model()

    st.title("🍎 果物当てゲーム（物体検知で正解を自動判定）🍊")
    st.write("画像に映る果物の数を予想してね！（答えは1〜10の範囲です）")

    if not st.session_state.generated:
        fruit = st.session_state.fruit
        eng = "red apples" if fruit == "りんご" else "oranges"

        while True:
            num = random.randint(1, 10)
            st.session_state.num = num

            prompt = (
                f"A high-resolution, realistic photo showing exactly {num} whole, clearly visible, and similarly sized {eng}, "
                "all fully in frame and evenly distributed on a clean white tabletop. No overlapping. "
                "Each fruit should be around palm-sized and spaced apart. Shot with soft, natural lighting, ultra-detailed, 4k quality."
            )

            with st.spinner(f"生成中..."):
                image = pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]

            save_path = "generated_fruit.png"
            image.save(save_path)
            st.session_state.image_path = save_path

            detected = detect_objects_yolov5(yolo_model, save_path)

            if detected <= 10:
                st.session_state.detected_count = detected
                break
            else:
                st.warning(f"⚠️ {detected}個検出されました。やり直します...")

        st.session_state.generated = True

    if st.session_state.image_path and os.path.exists(st.session_state.image_path):
        st.image(st.session_state.image_path, caption="この中に何個ある？", use_container_width=True)
    else:
        st.warning("画像がまだ生成されていません。")

    st.info("ヒント：果物の数は 1〜10 の間です。")
    st.write("いくつあると思いますか？（指で入力）")
    guess = camera()
    if guess is not None:
        st.session_state.guess = guess
        st.write(f'解答: {guess} 個')

    if st.button("答え合わせ"):
        if "guess" not in st.session_state:
            st.warning("まず指を使って果物の数を入力してください。")
        else:
            answer = st.session_state.detected_count
            guess = st.session_state.guess
            fruit = st.session_state.fruit
            if guess == answer:
                st.success(f"🎉 正解！{fruit}は {answer} 個ありました！")
            else:
                st.error(f"😢 残念！正解は {answer} 個の {fruit} でした。")
            st.info(f'実際は{fruit}を{st.session_state.num}個で生成した画像です')
            st.session_state.result_shown = True

    if st.session_state.result_shown:
        if st.button("もう一度遊ぶ"):
            st.session_state.clear()

elif screen == "手話":
    #hand_sign_mode()
    st.info("ローカル環境のみ対応ｺﾞﾒﾝｰ")
elif screen == "数当てver0.0.1":
    fruit_options = ["りんご", "みかん"]
    if "v001_consecutive_streak" not in st.session_state:
        st.session_state.v001_consecutive_streak = 0

    if "v001_fruit" not in st.session_state:
        reset_game_state()

    pipe = load_diffusion_model()
    yolo_model = load_yolov5_model()

    st.title("果物当てゲーム ver0.0.1")
    st.write("画像に映る果物の数を予想してね！（答えは1〜10の範囲です）")
    st.write(f"連続正解記録: **{st.session_state.v001_consecutive_streak}** 回")

    if not st.session_state.get("v001_generated", False):
        st.session_state.v001_fruit = random.choice(fruit_options)
        fruit = st.session_state.v001_fruit
        eng = "red apples" if fruit == "りんご" else "oranges"

        while True:
            num = random.randint(1, 10)
            st.session_state.v001_num = num

            prompt = (
                f"A high-resolution, realistic photo showing exactly {num} whole, clearly visible, and similarly sized {eng}, "
                "all fully in frame and evenly distributed on a clean white tabletop. No overlapping. "
                "Each fruit should be around palm-sized and spaced apart. Shot with soft, natural lighting, ultra-detailed, 4k quality."
            )

            with st.spinner("生成中..."):
                image = pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]

            save_path = "generated_fruit.png"
            image.save(save_path)
            st.session_state.v001_image_path = save_path

            detected = detect_objects_yolov5(yolo_model, save_path)

            if detected <= 10:
                st.session_state.v001_detected_count = detected
                break
            else:
                st.warning(f"{detected}個検出されました。やり直します...")

        st.session_state.v001_generated = True

    if st.session_state.get("v001_image_path") and os.path.exists(st.session_state.v001_image_path):
        st.image(st.session_state.v001_image_path, caption="この中に何個ある？", use_container_width=True)

    st.info("ヒント：果物の数は 1〜10 の間です。")
    st.write("いくつあると思いますか？（指で入力 - 2枚の写真を撮って合計します）")

    mp_hands, mp_drawing, hands = mediapipe_reset()

    if not st.session_state.get("v001_photo1_taken", False):
        st.subheader("📸 1枚目の写真を撮ってください")
        uploaded_file1 = camera_input_widget("1枚目の写真", key="v001_camera1_input")
        if uploaded_file1 is not None:
            count1 = mediapipe_process(uploaded_file1, mp_hands, mp_drawing, hands)
            st.session_state.v001_camera1_count = count1
            st.success(f"1枚目で{count1}本の指が検出されました")
            if st.button("✅ 次へ", key="v001_confirm_photo1"):
                st.session_state.v001_photo1_taken = True
                st.experimental_rerun()
    else:
        st.info(f"1枚目の指の数: {st.session_state.v001_camera1_count}本 (確定済み)")

    if st.session_state.get("v001_photo1_taken") and not st.session_state.get("v001_photo2_taken"):
        st.subheader("📸 2枚目の写真を撮ってください")
        uploaded_file2 = camera_input_widget("2枚目の写真", key="v001_camera2_input")
        if uploaded_file2 is not None:
            count2 = mediapipe_process(uploaded_file2, mp_hands, mp_drawing, hands)
            st.session_state.v001_camera2_count = count2
            st.success(f"2枚目で{count2}本の指が検出されました")
            if st.button("✅ 結果を見る", key="v001_confirm_photo2"):
                st.session_state.v001_photo2_taken = True
                st.session_state.v001_guess = st.session_state.v001_camera1_count + st.session_state.v001_camera2_count
                st.success(f"合計の予想: {st.session_state.v001_guess} 個")
                st.experimental_rerun()
    elif st.session_state.get("v001_photo1_taken") and st.session_state.get("v001_photo2_taken"):
        st.info(f"2枚目の指の数: {st.session_state.v001_camera2_count}本 (確定済み)")
        st.success(f"合計の予想: {st.session_state.v001_guess} 個")

    if st.session_state.get("v001_photo1_taken") and st.session_state.get("v001_photo2_taken") and st.session_state.get("v001_guess") is not None:
        if not st.session_state.get("v001_answer_checked", False):
            if st.button("答え合わせ", key="v001_check_answer"):
                answer = st.session_state.v001_detected_count
                guess = st.session_state.v001_guess
                fruit = st.session_state.v001_fruit

                if guess == answer:
                    st.session_state.last_result_message = f"🎉 正解！{fruit}は {answer} 個ありました！"
                    st.session_state.v001_consecutive_streak += 1
                    st.success(st.session_state.last_result_message)
                else:
                    st.session_state.last_result_message = f"😢 残念！正解は {answer} 個の {fruit} でした。"
                    st.session_state.v001_consecutive_streak = 0
                    st.error(st.session_state.last_result_message)

                st.session_state.v001_answer_checked = True

        else:
            st.info(st.session_state.last_result_message)
            if st.button("もう一度遊ぶ", key="v001_restart"):
                reset_game_state()
                st.experimental_rerun()
