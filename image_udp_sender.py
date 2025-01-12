import cv2, time
import socket
import struct

# UDP 설정
UDP_IP = "localhost"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 동영상 파일 열기
video_path = "video_fhd.mp4"
cap = cv2.VideoCapture(video_path)

# 패킷 크기 설정
MAX_PACKET_SIZE = 65000  # 65KB 미만으로 설정
HEADER_SIZE = 8          # 패킷 헤더 (frame_id + packet_id + total_packets)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 압축 (JPEG)
    _, compressed_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    frame_data = compressed_frame.tobytes()
    frame_size = len(frame_data)

    # 패킷 분할
    total_packets = (frame_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE  # 총 패킷 수
    frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)  # 현재 프레임 번호

    for packet_id in range(total_packets):
        start = packet_id * MAX_PACKET_SIZE
        end = min(start + MAX_PACKET_SIZE, frame_size)
        packet_data = frame_data[start:end]

        # 헤더 생성 (frame_id, packet_id, total_packets)
        header = struct.pack("IHH", int(frame_id), packet_id, total_packets)
        sock.sendto(header + packet_data, (UDP_IP, UDP_PORT))

    #time.sleep(0.033)  # 10ms delay

cap.release()
sock.close()

