import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import random
import os
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- ヘルパー関数 ---

# 角度を計算する関数
def calc_angle(a, b, c):
    """3つの点から角度を計算する"""
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    norm_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    norm_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    # ゼロ除算を防ぐための小さな値を追加
    cos_angle = dot / (norm_ba * norm_bc + 1e-6)
    # acosの引数が-1.0から1.0の範囲に収まるようにクリップ
    angle = math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))
    return angle

# MediaPipe Handsモデルをリセットする関数
@st.cache_resource(show_spinner=False)
def mediapipe_reset():
    """MediaPipe Handsモデルと描画ユーティリティを初期化する"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    # 静止画モードでHandsモデルを初期化（トラッキングは行わない）
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_hands, mp_drawing, hands

# Streamlitのカメラ入力ウィジェット
def camera_input(label):
    """Streamlitのカメラ入力ウィジェットを返す"""
    return st.camera_input(label)

# MediaPipeで指の数を処理する関数
def mediapipe_prosess(uploaded_file, mp_hands, mp_drawing, hands):
    """
    アップロードされた画像からMediaPipeを使って指の数を検出し、描画する。
    親指と他の指の検出ロジックを含む。
    """
    if uploaded_file is not None:
        # アップロードされたファイルをバイト配列として読み込み、OpenCVでデコード
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # MediaPipeで手を処理
        results = hands.process(img_rgb)

        total_finger_count = 0
        annotated_img = img.copy() # 描画用に画像のコピーを作成

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 検出された手のランドマークを描画
                mp_drawing.draw_landmarks(
                    annotated_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # ランドマークの色
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) # 接続の色
                )

                landmarks = []
                h, w, c = img.shape
                for id, lm in enumerate(hand_landmarks.landmark):
                    # ランドマークの座標を画像ピクセルに変換
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([id, cx, cy])

                if landmarks:
                    fingers = 0
                    
                    # 親指の検出
                    # 親指の付け根(MCP)、中間関節(IP)、先端(TIP)の座標を取得
                    mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP][1:3]
                    ip  = landmarks[mp_hands.HandLandmark.THUMB_IP][1:3]
                    tip = landmarks[mp_hands.HandLandmark.THUMB_TIP][1:3]
                    
                    # 親指の関節角度を計算
                    thumb_angle = calc_angle(mcp, ip, tip)
                    
                    # 角度が一定以上であれば親指が伸びていると判断
                    if thumb_angle > 160: # この閾値は調整が必要かもしれません
                        fingers += 1

                    # 人差し指から小指までの検出 (指の先端がPIP関節より上にあるかで判断)
                    if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][2] < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP][2]:
                        fingers += 1
                    if landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][2] < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP][2]:
                        fingers += 1
                    if landmarks[mp_hands.HandLandmark.RING_FINGER_TIP][2] < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP][2]:
                        fingers += 1
                    if landmarks[mp_hands.HandLandmark.PINKY_TIP][2] < landmarks[mp_hands.HandLandmark.PINKY_PIP][2]:
                        fingers += 1

                    total_finger_count += fingers
        
        # 処理された画像を表示
        st.image(annotated_img, caption="検出結果", channels="BGR", use_container_width=True)
        return total_finger_count
    else:
        return 0

# カメラ入力と指カウントのロジックをまとめた関数
def camera():
    """カメラからの入力で指の数をカウントし、結果を表示する"""
    mp_hands, mp_drawing, hands = mediapipe_reset()
    uploaded_file = camera_input("📸 指の本数をカウントする写真を撮ってください")
    if uploaded_file is not None:
        total_finger_count = mediapipe_prosess(uploaded_file, mp_hands, mp_drawing, hands)
        if total_finger_count > 0:
            st.success(f"画像内で**{total_finger_count}本**の指が伸びていると検出しました！")
            if st.button("✅ この結果で次へ進む"):
                return total_finger_count
        else:
            st.warning("伸びている指は検出されませんでした。もう一度試してください。")
    return None

# ゲームの状態をリセットするヘルパー関数
def reset_game_state():
    """ゲームのセッション状態をリセットする"""
    keys_to_reset = [
        "v001_fruit", "v001_num", "v001_image_path", "v001_detected_count",
        "v001_generated", "v001_result_shown", "v001_camera1_count", "v001_camera2_count",
        "v001_guess", "v001_photo1_taken", "v001_photo2_taken", "v001_answer_checked",
        "last_result_message", "last_result_type", "last_actual_num_message",
        "fruit", "num", "image_path", "generated", "result_shown", "guess",
        "quiz_greeting", "quiz_answered" # 手話モードのステートもリセット
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # 連続正解記録はリセットしない（ゲーム全体のリセットではないため）
    # if "v001_consecutive_streak" in st.session_state:
    #     del st.session_state["v001_consecutive_streak"]


# カメラ入力ウィジェット（キー付き）
def camera_input_widget(label, key):
    """キー付きのStreamlitカメラ入力ウィジェットを返す"""
    return st.camera_input(label, key=key)

# --- モデル読み込み関数 ---

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=False)
def load_diffusion_model():
    """Stable Diffusionモデルをロードする"""
    pipe = StableDiffusionPipeline.from_pretrained(
        "Lykon/dreamshaper-6",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True
    )
    return pipe.to(device)

@st.cache_resource(show_spinner=False)
def load_yolov5_model():
    """YOLOv5モデルをロードする"""
    # torch.hub.loadはインターネットからモデルをダウンロードしてロードする
    # ultralytics/yolov5リポジトリから'yolov5s'モデルをロード
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# YOLOv5でオブジェクトを検出する関数
def detect_objects_yolov5(model, image_path):
    """YOLOv5モデルを使用して画像内のオブジェクトを検出し、数を返す"""
    results = model(image_path)
    # 検出結果をpandas DataFrameとして取得し、検出されたオブジェクトの数を返す
    df = results.pandas().xyxy[0]
    return len(df)

# --- Streamlitアプリのメインロジック ---

# サイドバーでスクリーンを選択
screen = st.sidebar.selectbox("モード選択", ["数当て", "手話", "数当てver0.0.1"])

# --- 数当てモード ---
if screen == "数当て":
    st.title("🍎 果物当てゲーム（物体検知で正解を自動判定）🍊")
    st.write("画像に映る果物の数を予想してね！（答えは1〜10の範囲です）")

    # 初期化
    fruit_options = ["りんご", "みかん"]
    if "fruit" not in st.session_state:
        st.session_state.fruit = random.choice(fruit_options)
        st.session_state.image_path = None
        st.session_state.generated = False
        st.session_state.result_shown = False
        st.session_state.guess = None # 予想数を初期化

    # モデル読み込み
    # モデルのロードは時間がかかるため、st.cache_resourceでキャッシュされる
    pipe = load_diffusion_model()
    yolo_model = load_yolov5_model()

    # 画像生成
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

            with st.spinner(f"画像を生成中... (CPU環境のため数分かかる場合があります)"):
                image = pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]

            save_path = "generated_fruit.png"
            image.save(save_path)
            st.session_state.image_path = save_path

            # 生成された画像で物体検出を行い、検出された数を取得
            detected = detect_objects_yolov5(yolo_model, save_path)

            # 検出数が10個以下であればOK、そうでなければ再生成
            if detected <= 10:
                st.session_state.detected_count = detected
                break
            else:
                st.warning(f"⚠️ {detected}個検出されました。目標の10個を超えたため、画像を再生成します...")

        st.session_state.generated = True

    # 画像表示
    if st.session_state.image_path and os.path.exists(st.session_state.image_path):
        st.image(st.session_state.image_path, caption="この中に何個ある？", use_container_width=True)
    else:
        st.warning("画像がまだ生成されていません。")

    # カメラで指の数を入力（予想）
    st.info("ヒント：果物の数は 1〜10 の間です。")
    st.write("いくつあると思いますか？（指で入力）")
    
    # camera()関数は、指の数を検出し、ボタンが押されたらその数を返す
    current_guess = camera()
    if current_guess is not None:
        st.session_state.guess = current_guess # 検出された指の数をセッションステートに保存
        st.write(f'あなたの予想: {st.session_state.guess} 個')

    # 答え合わせボタン
    if st.button("答え合わせ"):
        if st.session_state.guess is None:
            st.warning("まず指を使って果物の数を入力してください。")
        else:
            answer = st.session_state.detected_count # YOLOv5で検出された実際の数
            guess = st.session_state.guess # ユーザーが指で入力した予想数
            fruit = st.session_state.fruit
            
            if guess == answer:
                st.success(f"🎉 正解！{fruit}は {answer} 個ありました！")
            else:
                st.error(f"😢 残念！正解は {answer} 個の {fruit} でした。")
            
            # 生成時の目標数も表示（検出数と異なる場合があるため）
            st.info(f'実際は{fruit}を{st.session_state.num}個で生成した画像です')
            st.session_state.result_shown = True # 結果が表示されたことを記録

    # もう一度遊ぶボタン
    if st.session_state.result_shown:
        if st.button("もう一度遊ぶ"):
            st.session_state.clear() # セッションステートをクリアしてゲームをリセット
            st.rerun() # アプリを再実行

# --- 手話モード ---
elif screen == "手話":
    st.title("✋ 指の本数をカウントして挨拶クイズ")

    # Mediapipe初期化 (手話モード用に別途初期化)
    # static_image_mode=False でリアルタイム処理を想定
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 指の本数を数える関数 (手話モード用)
    def count_fingers(hand_landmarks):
        finger_count = 0
        landmarks = hand_landmarks.landmark

        # 人差し指から小指までの先端とPIP関節のY座標を比較
        tips = [8, 12, 16, 20] # 各指の先端のランドマークID
        pips = [6, 10, 14, 18] # 各指のPIP関節のランドマークID
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y < landmarks[pip].y: # 指が伸びているか
                finger_count += 1

        # 親指（右手基準で、先端が付け根より左にあるかで判断）
        # 左手の場合はX座標の比較が逆になる可能性があるため、より汎用的な角度計算の方が良い場合もある
        # 今回は簡易的にX座標で判断
        if landmarks[4].x < landmarks[3].x: # 親指の先端が付け根より左にある（親指が伸びている）
            finger_count += 1

        return finger_count

    # 指の本数で挨拶を分類する関数
    def classify_greeting(count):
        if count == 1:
            return "こんにちは"
        elif count == 5:
            return "ありがとう"
        elif count == 0:
            return "バイバイ"
        else:
            return "認識できませんでした"

    # ビデオ処理クラス (streamlit-webrtc用)
    class HandShot(VideoTransformerBase):
        def __init__(self):
            self.frame = None
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.mp_drawing = mp.solutions.drawing_utils

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # リアルタイムで指の数をカウントし、画面に表示することも可能
                    # count = count_fingers(hand_landmarks)
                    # greeting = classify_greeting(count)
                    # cv2.putText(img, f"Fingers: {count} ({greeting})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            self.frame = img # 最新のフレームを保存
            return img

    # webrtc_streamerコンポーネント
    ctx = webrtc_streamer(
        key="finger-count-webrtc", # ユニークなキー
        video_processor_factory=HandShot,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # 撮影結果とクイズのプレースホルダー
    captured_image_placeholder = st.empty()
    result_placeholder = st.empty()

    # 撮影ボタン
    if st.button("📸 撮影して指の本数を認識しクイズ開始"):
        if ctx.video_processor and ctx.video_processor.frame is not None:
            image = ctx.video_processor.frame.copy() # 最新のフレームを取得
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb) # 静止画として処理

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                count = count_fingers(hand_landmarks)
                greeting = classify_greeting(count)

                # 撮影結果画像を表示
                captured_image_placeholder.image(image, caption="撮影結果", channels="BGR", use_container_width=True)

                # クイズ問題をセッションに保存
                st.session_state.quiz_greeting = greeting
                st.session_state.quiz_answered = False

                result_placeholder.info("クイズ開始！この手話は何の挨拶でしょうか？ 下の入力欄に答えてください。")

            else:
                result_placeholder.warning("手が検出されませんでした。もう一度撮影してください。")
        else:
            st.error("画像が取得できませんでした。カメラが起動しているか確認してください。")

    # クイズ回答入力欄
    if "quiz_greeting" in st.session_state and not st.session_state.get("quiz_answered", False):
        user_input = st.text_input("挨拶を入力してください（例：こんにちは、ありがとう、バイバイ）")

        if st.button("回答する"):
            correct_answer = st.session_state.quiz_greeting
            if user_input == correct_answer:
                result_placeholder.success(f"正解！この手話は「{correct_answer}」です。")
            else:
                result_placeholder.error(f"不正解！正解は「{correct_answer}」です。")

            st.session_state.quiz_answered = True
            # 回答後、もう一度撮影ボタンでリトライできるようにする

# --- 数当てver0.0.1 モード ---
elif screen == "数当てver0.0.1":
    st.title("果物当てゲーム ver0.0.1")
    st.write("画像に映る果物の数を予想してね！（答えは1〜10の範囲です）")

    # 連続正解記録の初期化
    if "v001_consecutive_streak" not in st.session_state:
        st.session_state.v001_consecutive_streak = 0
    st.write(f"連続正解記録: **{st.session_state.v001_consecutive_streak}** 回")

    # ゲーム状態の初期化（初回または「もう一度遊ぶ」時）
    if "v001_generated" not in st.session_state or not st.session_state.v001_generated:
        # v001_consecutive_streak 以外をリセット
        temp_streak = st.session_state.v001_consecutive_streak
        reset_game_state()
        st.session_state.v001_consecutive_streak = temp_streak
        
        st.session_state.v001_fruit = random.choice(["りんご", "みかん"])
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

            with st.spinner("画像を生成中... (CPU環境のため数分かかる場合があります)"):
                image = pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]

            save_path = "generated_fruit.png"
            image.save(save_path)
            st.session_state.v001_image_path = save_path

            detected = detect_objects_yolov5(yolo_model, save_path)

            if detected <= 10:
                st.session_state.v001_detected_count = detected
                break
            else:
                st.warning(f"⚠️ {detected}個検出されました。目標の10個を超えたため、画像を再生成します...")

        st.session_state.v001_generated = True
        st.rerun() # 画像生成後に再実行して画像を表示

    # モデル読み込み (初回のみロード)
    pipe = load_diffusion_model()
    yolo_model = load_yolov5_model()

    if st.session_state.get("v001_image_path") and os.path.exists(st.session_state.v001_image_path):
        st.image(st.session_state.v001_image_path, caption="この中に何個ある？", use_container_width=True)

    st.info("ヒント：果物の数は 1〜10 の間です。")
    st.write("いくつあると思いますか？（指で入力 - 2枚の写真を撮って合計します）")

    # MediaPipeの初期化
    mp_hands, mp_drawing, hands = mediapipe_reset()

    # 1枚目の写真
    if not st.session_state.get("v001_photo1_taken", False):
        st.subheader("📸 1枚目の写真を撮ってください")
        uploaded_file1 = camera_input_widget("1枚目の写真", key="v001_camera1_input")
        if uploaded_file1 is not None:
            # mediapipe_process 関数で画像処理と表示を行う
            count1 = mediapipe_prosess(uploaded_file1, mp_hands, mp_drawing, hands)
            st.session_state.v001_camera1_count = count1
            st.success(f"1枚目で{count1}本の指が検出されました")
            if st.button("✅ 次へ", key="v001_confirm_photo1"):
                st.session_state.v001_photo1_taken = True
                st.rerun()
    else:
        st.info(f"1枚目の指の数: {st.session_state.v001_camera1_count}本 (確定済み)")

    # 2枚目の写真
    if st.session_state.get("v001_photo1_taken") and not st.session_state.get("v001_photo2_taken"):
        st.subheader("📸 2枚目の写真を撮ってください")
        uploaded_file2 = camera_input_widget("2枚目の写真", key="v001_camera2_input")
        if uploaded_file2 is not None:
            count2 = mediapipe_prosess(uploaded_file2, mp_hands, mp_drawing, hands)
            st.session_state.v001_camera2_count = count2
            st.success(f"2枚目で{count2}本の指が検出されました")
            if st.button("✅ 結果を見る", key="v001_confirm_photo2"):
                st.session_state.v001_photo2_taken = True
                st.session_state.v001_guess = st.session_state.v001_camera1_count + st.session_state.v001_camera2_count
                st.success(f"合計の予想: {st.session_state.v001_guess} 個")
                st.rerun()
    elif st.session_state.get("v001_photo1_taken") and st.session_state.get("v001_photo2_taken"):
        st.info(f"2枚目の指の数: {st.session_state.v001_camera2_count}本 (確定済み)")
        st.success(f"合計の予想: {st.session_state.v001_guess} 個")

    # 答え合わせ
    if st.session_state.get("v001_photo1_taken") and st.session_state.get("v001_photo2_taken") and st.session_state.get("v001_guess") is not None:
        if not st.session_state.get("v001_answer_checked", False):
            if st.button("答え合わせ", key="v001_check_answer"):
                answer = st.session_state.v001_detected_count
                guess = st.session_state.v001_guess
                fruit = st.session_state.v001_fruit

                if guess == answer:
                    st.session_state.last_result_message = f"🎉 正解！{fruit}は {answer} 個でした！"
                    st.session_state.last_result_type = "success"
                    st.session_state.v001_consecutive_streak += 1
                else:
                    st.session_state.last_result_message = f"😢 残念！正解は {answer} 個の {fruit} でした。"
                    st.session_state.last_result_type = "error"
                    st.session_state.v001_consecutive_streak = 0 # 不正解で連続記録をリセット

                st.session_state.last_actual_num_message = f'実際は{fruit}を{st.session_state.v001_num}個で生成しました'
                st.session_state.v001_result_shown = True
                st.session_state.v001_answer_checked = True
                st.rerun() # 結果表示のために再実行
        else:
            st.info("答え合わせは完了しています。")

    # 結果表示と「もう一度遊ぶ」ボタン
    if st.session_state.get("v001_result_shown"):
        if st.session_state.last_result_type == "success":
            st.success(st.session_state.last_result_message)
        else:
            st.error(st.session_state.last_result_message)
        st.info(st.session_state.last_actual_num_message)
        st.write(f"現在の連続正解記録: {st.session_state.v001_consecutive_streak} 回")

        if st.button("もう一度遊ぶ", key="v001_play_again"):
            # 連続記録以外をリセットして再実行
            temp_streak = st.session_state.v001_consecutive_streak
            reset_game_state()
            st.session_state.v001_consecutive_streak = temp_streak
            st.rerun()

