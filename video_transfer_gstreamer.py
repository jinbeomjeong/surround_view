import cv2

# GStreamer 송출 파이프라인 설정
gst_pipeline = ("appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=7000 speed-preset=superfast ! "
    "rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.93.1 port=9999")

# OpenCV에서 GStreamer를 사용해 VideoWriter 초기화
out = cv2.VideoWriter(gst_pipeline, cv2.VideoWriter_fourcc(*"X264"), 30.0, (1920, 1080), True)

# 카메라(또는 비디오 파일)로부터 영상 캡처
cap = cv2.VideoCapture('video_2.mp4')  # 0번 카메라를 사용
cv2.namedWindow("video", cv2.WINDOW_NORMAL)
# OpenCV  생성

while True:
    ret, frame = cap.read()
    if not ret:
        print("영상을 읽을 수 없습니다. 종료합니다.")
        break

    # OpenCV 창에 영상 표시 (디버깅용)
    cv2.imshow("video", frame)

    # GStreamer를 통해 영상 송출
    out.write(frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()