import requests

# Địa chỉ của API
url_form = "http://127.0.0.1:5000/form"  # Thay đổi nếu server Flask đang chạy trên địa chỉ khác
url_form1 = "http://127.0.0.1:5000/form1"  # API cho form1

# Dữ liệu mẫu cho /form (các câu trả lời từ q1 đến q21)
form_data = {
    'q1': 5,
    'q2': 3,
    'q3': 4,
    'q4': 2,
    'q5': 3,
    'q6': 4,
    'q7': 2,
    'q8': 5,
    'q9': 3,
    'q10': 4,
    'q11': 5,
    'q12': 3,
    'q13': 4,
    'q14': 5,
    'q15': 4,
    'q16': 3,
    'q17': 5,
    'q18': 4,
    'q19': 2,
    'q20': 5,
    'q21': 3
}

# Dữ liệu mẫu cho /form1 (các giá trị từ x1 đến x13)
form1_data = {
    'x1': 5,
    'x2': 3,
    'x3': 4,
    'x4': 2,
    'x5': 3,
    'x6': 4,
    'x7': 2,
    'x8': 5,
    'x9': 3,
    'x10': 4,
    'x11': 5,
    'x12': 3,
    'x13': 4
}

# Gửi dữ liệu tới API /form
def test_form_api():
    response = requests.post(url_form, data=form_data)
    
    if response.status_code == 200:
        print("Response from /form API:")
        print(response.json())  # In kết quả trả về từ API
    else:
        print(f"Failed to get response from /form: {response.status_code}")

# Gửi dữ liệu tới API /form1
def test_form1_api():
    response = requests.post(url_form1, data=form1_data)
    
    if response.status_code == 200:
        print("Response from /form1 API:")
        print(response.json())  # In kết quả trả về từ API
    else:
        print(f"Failed to get response from /form1: {response.status_code}")

if __name__ == "__main__":
    # Test API /form
    test_form_api()
    
    # Test API /form1
    test_form1_api()
