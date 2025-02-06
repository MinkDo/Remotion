from threading import Thread
from flask import Flask, render_template, request, jsonify
import pyttsx3
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import AutoModel
import torch
import model as m # Giả sử bạn đã có mô-đun model đã được import
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertConfig
from huggingface_hub import notebook_login


try:
    import speech_recognition as sr
except ImportError:
    sr = None  # Xử lý khi speech_recognition không khả dụng

# Khởi tạo Flask
app = Flask(__name__)

# Thiết lập text-to-speech
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 130)
engine.setProperty('volume', 0.8)  # Giá trị từ 0.0 đến 1.0

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

label_mapping = {
    'Anxiety': 0,
    'Normal': 1,
    'Depression': 2,
    'Suicidal': 3,
    'Stress': 4,
    'Bipolar': 5,
    'Personality disorder': 6
}


model_path = "model2.pth"
config = BertConfig.from_pretrained(
        'bert-mini',
        num_labels=7,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )
try:
    
    
    # Tải mô hình BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained('bert-mini', config=config)
    
    # Kiểm tra và tải checkpoint
    checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print("Mô hình đã được tải thành công!")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp mô hình tại {model_path}.")
except Exception as e:
    print(f"Lỗi khi tải tệp: {e}")
    
tokenizer = BertTokenizer.from_pretrained("bert-mini")

# Tạo lại pipeline cho sentiment analysis với mô hình đã fine-tuned
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Hàm dự đoán cảm xúc sử dụng mô hình đã fine-tuned
def predict_minBERT(texts):
    predictions = classifier(texts, truncation=True, max_length=256)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    return [reverse_label_mapping[int(pred['label'].split('_')[-1])] for pred in predictions]



@app.route("/")
def index():
    param = 'Hello Everyone. We help you to find if you are suffering from Anxiety/Stress/Depression and suggest measures to control it or manage it.'
    thr = Thread(target=speak, args=[param])
    thr.start()
    return render_template("index.html")

@app.route("/index")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu từ request
        input_data = request.get_json()

        # Lấy text từ request
        text = input_data.get("text", "")
        if not text:
            return jsonify({"error": "Text is required"}), 400

        # Dự đoán cảm xúc
        emotion = predict_minBERT(text)

        print(emotion)

        return jsonify({"predicted_emotion": emotion[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/form", methods=["POST"])
def model():
    if request.method == "POST":
        
       
        # Đọc dữ liệu JSON từ request
        data = request.json

        # Truy cập các giá trị từ JSON
        ans1 = data.get("q1")
        ans4 = data.get("q4")
        ans7 = data.get("q7")
        ans10 = data.get("q10")
        ans13 = data.get("q13")
        ans16 = data.get("q16")
        ans19 = data.get("q19")

        ans2 = data.get("q2")
        ans5 = data.get("q5")
        ans8 = data.get("q8")
        ans11 = data.get("q11")
        ans14 = data.get("q14")
        ans17 = data.get("q17")
        ans20 = data.get("q20")

        ans3 = data.get("q3")
        ans6 = data.get("q6")
        ans9 = data.get("q9")
        ans12 = data.get("q12")
        ans15 = data.get("q15")
        ans18 = data.get("q18")
        ans21 = data.get("q21")


        print(ans1)
        try:
            # Chuyển đổi dữ liệu từ form thành kiểu float
            ans1 = float(ans1)
            ans2 = float(ans2)
            ans3 = float(ans3)
            ans4 = float(ans4)
            ans5 = float(ans5)
            ans6 = float(ans6)
            ans7 = float(ans7)
            ans8 = float(ans8)
            ans9 = float(ans9)
            ans10 = float(ans10)
            ans11 = float(ans11)
            ans12 = float(ans12)
            ans13 = float(ans13)
            ans14 = float(ans14)
            ans15 = float(ans15)
            ans16 = float(ans16)
            ans17 = float(ans17)
            ans18 = float(ans18)
            ans19 = float(ans19)
            ans20 = float(ans20)
            ans21 = float(ans21)

            global anxietyscore, stressscore, depressionscore
            # Tính toán các điểm cho Anxiety, Depression và Stress
            anxietyscore = (ans1 + ans4 + ans7 + ans10 + ans13 + ans16 + ans19) * 2
            depressionscore = (ans2 + ans5 + ans8 + ans11 + ans14 + ans17 + ans20) * 2
            stressscore = (ans3 + ans6 + ans9 + ans12 + ans15 + ans18 + ans21) * 2

            # Trả về kết quả dưới dạng JSON
            return jsonify({
                "anxiety_score": anxietyscore,
                "depression_score": depressionscore,
                "stress_score": stressscore,
                "calculation_success": True
            })

        except ValueError:
            # Xử lý khi có lỗi với dữ liệu nhập
            return jsonify({
                "error": "Invalid input. Please provide valid numerical values.",
                "calculation_success": False
            })

    return jsonify({
        "error": "Invalid request method. Only POST is allowed.",
        "calculation_success": False
    })


@app.route("/form1", methods=["POST"])
def model1():
    if request.method == "POST":
        # Lấy dữ liệu từ form (inputs: x1 đến x13)
        in1 = request.form.get('x1')
        in2 = request.form.get('x2')
        in3 = request.form.get('x3')
        in4 = request.form.get('x4')
        in5 = request.form.get('x5')
        in6 = request.form.get('x6')
        in7 = request.form.get('x7')
        in8 = request.form.get('x8')
        in9 = request.form.get('x9')
        in10 = request.form.get('x10')
        in11 = request.form.get('x11')
        in12 = request.form.get('x12')
        in13 = request.form.get('x13')

        try:
            # Chuyển dữ liệu thành float
            in1 = float(in1)
            in2 = float(in2)
            in3 = float(in3)
            in4 = float(in4)
            in5 = float(in5)
            in6 = float(in6)
            in7 = float(in7)
            in8 = float(in8)
            in9 = float(in9)
            in10 = float(in10)
            in11 = float(in11)
            in12 = float(in12)
            in13 = float(in13)

            # Log sau khi chuyển đổi
            app.logger.info(f"Converted Inputs: {in1}, {in2}, {in3}, {in4}, {in5}, {in6}, {in7}, {in8}, {in9}, {in10}, {in11}, {in12}, {in13}")
            # Dự đoán từ mô hình
            answer = m.anxiety_pred(in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, anxietyscore)
            answer1 = m.stress_pred(in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, stressscore)
            answer2 = m.depression_pred(in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, depressionscore)
             # Nếu kết quả trả về là ndarray, chúng ta chuyển nó thành list
            if isinstance(answer, np.ndarray):
                answer = answer.tolist()  # Chuyển ndarray thành list
            if isinstance(answer1, np.ndarray):
                answer1 = answer1.tolist()
            if isinstance(answer2, np.ndarray):
                answer2 = answer2.tolist()

            # Log kết quả dự đoán
            app.logger.info(f"Predictions: Anxiety: {answer}, Stress: {answer1}, Depression: {answer2}")

            # Trả về kết quả dưới dạng JSON
            return jsonify({
                "anxiety_pred": answer,
                "stress_pred": answer1,
                "depression_pred": answer2,
                "calculation_success": True
            })

        except ValueError:
            # Xử lý khi có lỗi với dữ liệu nhập
            return jsonify({
                "error": "Invalid input. Please provide valid numerical values.",
                "calculation_success": False
            })

    return jsonify({
        "error": "Invalid request method. Only POST is allowed.",
        "calculation_success": False
    })

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
