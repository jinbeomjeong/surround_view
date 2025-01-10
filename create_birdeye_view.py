import cv2, imagezmq, threading
import numpy as np

from utils.surround_view import blend_bird_eye_img, convert_bird_eye_view_cuda

image_hub = imagezmq.ImageHub()
image_ready = False
bird_eye_img_list = []
gpu_img = cv2.cuda_GpuMat()

front_cam_arr = np.empty(shape=(1080, 1920, 3), dtype=np.uint8)
rear_cam_arr = np.empty(shape=(1080, 1920, 3), dtype=np.uint8)
left_cam_arr = np.empty(shape=(1080, 1920, 3), dtype=np.uint8)
right_cam_arr = np.empty(shape=(1080, 1920, 3), dtype=np.uint8)

image = np.empty(shape=(1080 * 2, 1920 * 2, 3), dtype=np.uint8)
result_img = np.empty(shape=(1740-170, 1740-350, 3), dtype=np.uint8)

cv2.namedWindow(winname='video', flags=cv2.WINDOW_NORMAL)


def create_bird_eye_img():
    global bird_eye_img_list, front_cam_arr, rear_cam_arr, left_cam_arr, right_cam_arr, result_img

    while True:
        bird_eye_img_list.clear()

        front_cam_arr = image[0:1080, 0:1920]
        rear_cam_arr = image[0:1080, 1920:3840]
        left_cam_arr = image[1080:2160, 0:1920]
        right_cam_arr = image[1080:2160, 1920:3840]

        for img in [front_cam_arr, left_cam_arr, right_cam_arr, rear_cam_arr]:
            bird_eye_img_list.append(convert_bird_eye_view_cuda(gpu_img=gpu_img, img=img))

        raw_bird_eye_img = blend_bird_eye_img(*bird_eye_img_list)

        result_img[:] = raw_bird_eye_img[170:1740, 350:1740]


bird_eye_img_task = threading.Thread(target=create_bird_eye_img)
bird_eye_img_task.daemon = True
bird_eye_img_task.start()


while True:
    sender_name, compressed_img = image_hub.recv_image()
    image_hub.send_reply(b'OK')

    image[:] = cv2.imdecode(np.frombuffer(compressed_img, dtype=np.uint8), cv2.IMREAD_COLOR)

    cv2.imshow('video', result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  break

cv2.destroyAllWindows()