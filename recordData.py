import audioread
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 오디오 파일을 읽고 목표 샘플 속도로 리샘플링
def read_audio(file_path, sr=16000):
    y = []
    with audioread.audio_open(file_path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        for frame in input_file:
            samples = np.frombuffer(frame, dtype=np.int16)
            samples = samples.reshape(-1, n_channels)
            samples = samples.mean(axis=1)
            y.extend(samples)
    y = np.array(y, dtype=np.float32)
    y = scipy.signal.resample(y, int(len(y) * sr / sr_native))
    return y, sr

# 저주파 필터 함수 정의
def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# 신호의 에너지를 계산하는 함수 정의
def calculate_energy(data):
    return np.square(data)

# 호흡 구간 감지 함수 정의
def detect_breathing_intervals(data, sr, threshold=15, min_interval=0.5):
    breath_intervals = []
    above_threshold = False
    min_samples = int(min_interval * sr)
    start = None
    for i in range(len(data)):
        if data[i] > threshold and not above_threshold:
            if start is None:
                start = i
            above_threshold = True
        elif data[i] <= threshold and above_threshold:
            if start is not None and (i - start > min_samples):
                breath_intervals.append((start, i))
                start = None
            above_threshold = False
    return breath_intervals

# PEF 추정 함수 정의
def estimate_pef_from_energy(data):
    peak_energy = np.max(data)
    pef = peak_energy * 10  # 예시 상수로 조정
    return pef

audio_data = 'audio.wav'
x, sr = read_audio(audio_data, sr=16000)

# 저주파 필터 적용
filtered_data = low_pass_filter(x, cutoff=5, fs=sr)

# 신호의 에너지 계산
energy_data = calculate_energy(filtered_data)

# 에너지 신호 시각화
plt.figure(figsize=(14, 5))
plt.plot(np.linspace(0, len(energy_data)/sr, len(energy_data)), energy_data)
plt.title('Energy of Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.show()

# 호흡 구간 감지
breathing_intervals = detect_breathing_intervals(energy_data, sr, threshold=15, min_interval=0.5)

# 각 호흡 구간의 최대 에너지 및 PEF 추정
pefs = []
for start, end in breathing_intervals:
    breath_segment = energy_data[start:end]
    pef = estimate_pef_from_energy(breath_segment)
    pefs.append(pef)

# 결과 출력
for i, pef in enumerate(pefs):
    print(f"Breath {i+1}: Estimated Peak Expiratory Flow (PEF) = {pef:.2f} L/min")

# 호흡 구간 시각화
plt.figure(figsize=(14, 5))
plt.plot(np.linspace(0, len(energy_data)/sr, len(energy_data)), energy_data, label='Energy Signal')
for idx, (start, end) in enumerate(breathing_intervals):
    plt.axvline(start/sr, color='red', linestyle='--', label='Breath Start' if idx == 0 else "")
    plt.axvline(end/sr, color='blue', linestyle='--', label='Breath End' if idx == 0 else "")
plt.title('Detected Breathing Intervals in Energy Signal')
plt.xlabel('Time (s)')
plt.ylabel('Energy')

# 중복 레이블 제거
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.show()