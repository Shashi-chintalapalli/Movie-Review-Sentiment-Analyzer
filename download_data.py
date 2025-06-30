# download_data.py
import gdown
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Google Drive file ID from your link
file_id = "1ZlITLK55R20yY3IZy-39jxXsiEl16c7_"
url = f"https://drive.google.com/uc?id={file_id}"

# Output path
output = "data/data.csv"

print("⬇️ Downloading data.csv from Google Drive...")
gdown.download(url, output, quiet=False)
print("✅ Download completed: data/data.csv")
