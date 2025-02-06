import requests
import json

# URL của API (giả sử Flask app đang chạy trên localhost)
url = "http://127.0.0.1:5000/predict"

# Dữ liệu đầu vào
data = {
    "text": "there will always be tai. So it's not just beautiful"
}

# Gửi yêu cầu POST đến API
response = requests.post(url, json=data)

# Kiểm tra trạng thái phản hồi
if response.status_code == 200:
    # Hiển thị kết quả dự đoán cảm xúc
    result = response.json()
    print(result)
else:
    print(f"Error: {response.status_code}, {response.text}")
