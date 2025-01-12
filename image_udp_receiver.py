import cv2
import socket
import struct
import numpy as np

# UDP 설정
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# 데이터 버퍼
buffer = {}
HEADER_SIZE = 8
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

while True:
    # 데이터 수신
    data, addr = sock.recvfrom(65507)

    # 헤더 파싱 (frame_id, packet_id, total_packets)
    header = data[:HEADER_SIZE]
    frame_id, packet_id, total_packets = struct.unpack("IHH", header)
    packet_data = data[HEADER_SIZE:]

    # 프레임 데이터 버퍼에 저장
    if frame_id not in buffer:
        buffer[frame_id] = [None] * total_packets

    buffer[frame_id][packet_id] = packet_data

    # 모든 패킷이 도착했는지 확인
    if None not in buffer[frame_id]:
        # 데이터 조합
        frame_data = b"".join(buffer[frame_id])
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None:
            print("Frame decoding failed.")
            continue

        # 화면에 프레임 출력
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 버퍼에서 해당 프레임 제거
        del buffer[frame_id]

cv2.destroyAllWindows()
sock.close()

