import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from deepforest import CascadeForestRegressor
import os
import gdown

# Google DriveのファイルID（共有リンクから取得）
FILE_ID = os.getenv("DRIVE_FILE_ID")
OUTPUT_CSV = "housoukikan.csv"

# ファイルのダウンロード
gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", OUTPUT_CSV, quiet=False)

# データの読み込み
df = pd.read_csv(OUTPUT_CSV)

# 特徴量と目的変数に分割
X = df.drop(columns=['BD枚数'])  # 必要に応じて修正
y = df['BD枚数']

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル構築と学習
model = CascadeForestRegressor(random_state=42, n_jobs=-1)
model.fit(X_train.values, y_train.values)

# 予測と評価
y_pred = model.predict(X_test.values)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
