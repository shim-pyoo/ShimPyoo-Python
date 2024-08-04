from flask import Flask, request, jsonify
import recordData  # 필요한 모듈
import pandas as pd
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# 데이터를 저장할 데이터프레임 생성
data = pd.DataFrame(columns=["date", "pef"])

# 녹음한 데이터를 분석하여 가장 큰 세 개의 PEF를 측정 순서대로 반환
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # 파일을 저장하고 분석 실행
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)
    
    # 오디오 분석 실행
    top_pefs = recordData.analyze_audio(file_path)
    
    # 하루의 가장 큰 PEF 값 추출
    max_pef = max(pef for pef, start, end in top_pefs)
    
    # 날짜와 함께 데이터 저장
    date = file.filename.split('.')[0]  # 예: 파일명이 '2024-08-04.wav'라면 날짜를 '2024-08-04'로 사용
    global data
    data = data.append({"date": date, "pef": max_pef}, ignore_index=True)
    
    # 하루의 세 개의 PEF 값 시각화
    visualize_daily_pefs(date, top_pefs)

    return jsonify({"date": date, "max_pef": max_pef}), 200

@app.route('/monthly_report', methods=['GET'])
def monthly_report():
    # 한 달간의 PEF 값 시각화
    visualize_monthly_pefs(data)
    return jsonify(data.to_dict(orient="records")), 200

def visualize_daily_pefs(date, pefs):
    # 하루의 세 개의 PEF 값 시각화
    plt.figure(figsize=(10, 5))
    pef_values = [pef for pef, start, end in pefs]
    plt.bar(range(1, len(pef_values) + 1), pef_values)
    plt.xlabel('Breath Number')
    plt.ylabel('PEF (L/min)')
    plt.title(f'Daily PEF Values for {date}')
    plt.savefig(f'./static/daily_pefs_{date}.png')
    plt.close()

def visualize_monthly_pefs(data):
    # 한 달간의 가장 큰 PEF 값 시각화
    plt.figure(figsize=(14, 7))
    plt.plot(pd.to_datetime(data['date']), data['pef'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Max PEF (L/min)')
    plt.title('Monthly Max PEF Values')
    plt.grid(True)
    plt.savefig('./static/monthly_pefs.png')
    plt.close()

@app.route('/')
def home():
    return 'This is Home!'

if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')
    if not os.path.exists('./static'):
        os.makedirs('./static')
    app.run(host='0.0.0.0', port=5001, debug=True)