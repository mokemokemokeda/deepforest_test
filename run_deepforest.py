import pandas as pd
import numpy as np
import json
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from deepforest import CascadeForestRegressor
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import sys
import os

# deep-forest を clone していれば使えるようにパス追加
sys.path.append(os.path.join(os.getcwd(), "deep-forest"))

# 環境変数からサービスアカウントキーを取得
google_credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT")
if not google_credentials_json:
    raise ValueError("GOOGLE_SERVICE_ACCOUNT が設定されていません。")

json_data = json.loads(google_credentials_json)
credentials = service_account.Credentials.from_service_account_info(json_data)
drive_service = build("drive", "v3", credentials=credentials)

# ファイル名からID取得
def get_file_id(file_name):
    results = drive_service.files().list(
        q=f"name = '{file_name}' and trashed = false",
        fields="files(id, name)",
        spaces='drive'
    ).execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None

file_name = "housoukikan.csv"
file_id = get_file_id(file_name)

if not file_id:
    raise FileNotFoundError(f"{file_name} がGoogle Driveに存在しません。")

# ファイル取得
request = drive_service.files().get_media(fileId=file_id)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while not done:
    status, done = downloader.next_chunk()

fh.seek(0)
df = pd.read_csv(fh)
print("CSV読み込み成功:", df.columns.tolist())

#deepforestで処理
X = df.drop(columns=['BD枚数'])  # 必要に応じて修正
y = df['BD枚数']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CascadeForestRegressor(random_state=42, n_jobs=-1)
model.fit(X_train.values, y_train.values)

y_pred = model.predict(X_test.values)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
