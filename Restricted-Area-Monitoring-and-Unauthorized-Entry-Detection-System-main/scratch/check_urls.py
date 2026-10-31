import urllib.request

urls = [
    'https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4',
    'https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4',
    'https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female-and-male.mp4',
    'https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4',
    'https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4',
    'https://github.com/ageitgey/face_recognition/raw/master/examples/short_video.mp4',
    'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg',
    'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg'
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        code = urllib.request.urlopen(req).getcode()
        print(f"[OK] {url} -> {code}")
    except Exception as e:
        print(f"[FAIL] {url} -> {e}")
