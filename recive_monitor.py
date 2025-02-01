import imagezmq, cv2, threading, torch
import numpy as np
from utils.surround_view import get_screen_resolution, resize_to_fit_display
from torchvision.io import decode_jpeg, ImageReadMode


stop_event = threading.Event()
jpg_bytearr = None
jpg_data_ready = False


def receive_image():
    global jpg_bytearr, jpg_data_ready

    image_hub = imagezmq.ImageHub()

    while not stop_event.is_set():
        name, jpg_bytestring = image_hub.recv_jpg()
        jpg_bytearr = np.frombuffer(jpg_bytestring, dtype=np.uint8)
        image_hub.send_reply(b'OK')
        jpg_data_ready = True

    image_hub.close()


if __name__ == "__main__":
    display_width, display_height = get_screen_resolution()
    cv2.namedWindow(winname="img", flags=cv2.WINDOW_NORMAL)

    receive_img_thread = threading.Thread(target=receive_image)
    receive_img_thread.start()

    while True:
        raw_img = decode_jpeg(torch.from_numpy(jpg_bytearr), mode=ImageReadMode.RGB, device='CPU')  # for nvidia JPEG decoding of x86-64 platform
        # raw_img = nv_jpeg.decode(compressed_img)  # for nvidia JPEG decoding of jetson
        # raw_img = cv2.imdecode(np.frombuffer(compressed_img, dtype=np.uint8), cv2.IMREAD_COLOR)  # for jpeg-turbo of opencv decoding
        resized_img = resize_to_fit_display(raw_img, display_width, display_height)

        cv2.imshow(winname='img', mat=resized_img)

        if cv2.waitKey(1) == 'q':  # Exit on pressing 'ESC'
            break

    stop_event.set()
    receive_img_thread.join()
    cv2.destroyAllWindows()