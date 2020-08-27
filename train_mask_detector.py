# 必要なパッケージをインポートする
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 引数パーサーを構築し、引数を解析します
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# 初期学習率、学習するエポック数、バッチサイズを初期化する
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# データセットディレクトリ内の画像のリストを取得し、データ（画像）とクラス画像のリストを初期化します
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# 画像パスをループする
for imagePath in imagePaths:
	# ファイル名からクラスラベルを抽出する
	label = imagePath.split(os.path.sep)[-2]

	# 入力画像（224x224）をロードして前処理する
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# データとラベルのリストをそれぞれ更新する
	data.append(image)
	labels.append(label)

# データとラベルをNumPy配列に変換する
data = np.array(data, dtype="float32")
labels = np.array(labels)

# ラベルでワンホットエンコーディングを実行する
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# データの75％をトレーニングに使用し、残りの25％をテストに使用して、データをトレーニングとテストの分割に分割する
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# データ拡張のためのトレーニング画像ジェネレータを構築する
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2ネットワークを読み込み、ヘッドFCレイヤーセットがオフになっていることを確認します。
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# ベースモデルの上に配置されるモデルのヘッドを構築します
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# ヘッドFCモデルをベースモデルの上に配置します（これは、トレーニングする実際のモデルになります）
model = Model(inputs=baseModel.input, outputs=headModel)

# 基本モデルのすべてのレイヤーをループしてフリーズし、最初のトレーニングプロセス中にそれらが更新されないようにします。
for layer in baseModel.layers:
	layer.trainable = False

# モデルをコンパイルする
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# ネットワークの責任者を訓練する
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# テストセットを予測する
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# テストセット内の各画像について、対応する最大予測確率を持つラベルのインデックスを見つける必要があります
predIdxs = np.argmax(predIdxs, axis=1)

# うまくフォーマットされた分類レポートを表示する
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# モデルをディスクにシリアル化する
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# トレーニングの損失と精度をプロットする
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
