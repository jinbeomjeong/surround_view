import imagezmq, cv2, screeninfo
import numpy as np

from nvjpeg import NvJpeg

def get_screen_resolution():
    screen = screeninfo.get_monitors()[0]  # 첫 번째 모니터 기준
    return screen.width, screen.height


def resize_to_fit_display(image, display_width, display_height):
    h, w = image.shape[:2]
    scale = min(display_width / w, display_height / h)  # 비율 유지
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_image = cv2.resize(src=image, dsize=(new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image


image_hub = imagezmq.ImageHub()
nv_jpeg = NvJpeg()
cv2.namedWindow(winname="img", flags=cv2.WINDOW_NORMAL)
display_width, display_height = get_screen_resolution()


while True:
    sender_name, compressed_img = image_hub.recv_image()
    image_hub.send_reply(b'OK')

    raw_img = nv_jpeg.decode(compressed_img)  # for Nvidia JPEG decoding
    # raw_img = cv2.imdecode(np.frombuffer(compressed_img, dtype=np.uint8), cv2.IMREAD_COLOR)  # for jpeg-turbo of opencv decoding
    resized_img = resize_to_fit_display(raw_img, display_width, display_height)

    cv2.imshow(winname='img', mat=resized_img)

    if cv2.waitKey(1) == 'q':  # Exit on pressing 'ESC'
        break

image_hub.close()
cv2.destroyAllWindows()