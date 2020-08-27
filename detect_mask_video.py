
# 必要なパッケージをインポートする
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	
	# フレームの寸法を取得して、ブロブを作成します
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# blobをネットワークに渡し、顔検出を取得します
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# 顔のリスト、対応する場所、および顔マスクネットワークからの予測のリストを初期化する
	faces = []
	locs = []
	preds = []

	# detectionsをループする
	for i in range(0, detections.shape[2]):
		# detectionsに関連する信頼度（確率とか）を抽出する
		confidence = detections[0, 0, i, 2]

		# 信頼度が最小信頼度よりも高いことを確認して、弱い検出を除外する
		if confidence > args["confidence"]:
			# オブジェクトの境界ボックスの（x、y）座標を計算する
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 境界ボックスがフレームの寸法内に収まるようにする
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# 顔のROIを抽出し、BGRからRGBチャネルの順序に変換し、224x224にサイズ変更して、前処理する
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# 面と境界ボックスをそれぞれのリストに追加する
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# 少なくとも1つの顔が検出された場合にのみ予測を行う
	if len(faces) > 0:

		# より高速な推論のために、上記の「for」ループで1つずつ予測するのではなく、*all*面で同時にバッチ予測を行います
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# 顔の位置とそれに対応する2つのタプルを返す

	return (locs, preds)

# 引数パーサーを構築し、引数を解析します
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# シリアル化された顔検出モデルをディスクから読み込む
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# ディスクからフェイスマスク検出器モデルを読み込む
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# ビデオストリームを初期化し、カメラセンサーがウォームアップできるようにします
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# ビデオストリームのフレームをループします
while True:
	# スレッド化されたビデオストリームからフレームを取得し、最大幅が400ピクセルになるようにサイズ変更します
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# フレーム内の顔を検出し、フェイスマスクを着用しているかどうかを判断する
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# 検出された顔の位置と対応する位置をループします
	for (box, pred) in zip(locs, preds):
		# 境界ボックスと予測を解凍する
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# 境界ボックスとテキストの描画に使用するクラスのラベルと色を決定します
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# 確率をラベルに含める
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# ラベルとバウンディングボックスの長方形を出力フレームに表示する
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# 出力フレームを表示する
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# `q`キーが押された場合は、ループから抜けます
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
