import requests
import json

# URL của API
url = "http://localhost:5000/predict"

# Dữ liệu thử nghiệm
data = {
    "text": "I am feeling great today!"
}

# Gửi yêu cầu POST tới API
response = requests.post(url, json=data)

# Kiểm tra phản hồi từ server
if response.status_code == 200:
    print("API response:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Request failed with status code {response.status_code}")
