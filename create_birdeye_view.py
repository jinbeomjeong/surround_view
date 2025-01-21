import cv2, imagezmq, threading
import numpy as np

from nvjpeg import NvJpeg
from utils.surround_view import blend_bird_eye_img_v2, convert_bird_eye_view

image_hub = imagezmq.ImageHub()
nv_jpeg = NvJpeg()
bird_eye_img_list = []

front_cam_arr = np.empty(shape=(1080, 1920, 3), dtype=np.uint8)
rear_cam_arr = np.empty(shape=(1080, 1920, 3), dtype=np.uint8)
left_cam_arr = np.empty(shape=(1080, 1920, 3), dtype=np.uint8)
right_cam_arr = np.empty(shape=(1080, 1920, 3), dtype=np.uint8)

result_img = np.empty(shape=(1740-170, 1740-350, 3), dtype=np.uint8)

cv2.namedWindow(winname='video', flags=cv2.WINDOW_NORMAL)


def create_bird_eye_img():
    global bird_eye_img_list, front_cam_arr, rear_cam_arr, left_cam_arr, right_cam_arr, result_img

    while True:
        bird_eye_img_list.clear()

        # front_cam_arr = image[0:1080, 0:1920]
        # rear_cam_arr = image[0:1080, 1920:3840]
        # left_cam_arr = image[1080:2160, 0:1920]
        # right_cam_arr = image[1080:2160, 1920:3840]

        for img in [front_cam_arr, left_cam_arr, right_cam_arr, rear_cam_arr]:
            bird_eye_img_list.append(convert_bird_eye_view(img=img))

        raw_bird_eye_img = blend_bird_eye_img_v2(*bird_eye_img_list)

        result_img[:] = raw_bird_eye_img[170:1740, 350:1740]


bird_eye_img_task = threading.Thread(target=create_bird_eye_img)
bird_eye_img_task.daemon = True
bird_eye_img_task.start()


while True:
    sender_name, img_byte = image_hub.recv_image()
    image_hub.send_reply(b'OK')
    raw_img = nv_jpeg.decode(img_byte)

    if sender_name=='front':
        front_cam_arr[:] = raw_img

    if sender_name=='left':
        left_cam_arr[:] = raw_img

    if sender_name=='right':
        right_cam_arr[:] = raw_img

    if sender_name=='rear':
        rear_cam_arr[:] = raw_img

    cv2.imshow('video', result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  break

cv2.destroyAllWindows()