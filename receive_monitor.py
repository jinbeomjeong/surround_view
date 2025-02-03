import imagezmq, cv2, threading, torch, itertools
import numpy as np
from utils.surround_view import get_screen_resolution, resize_to_fit_display
from torchvision.io import decode_jpeg, ImageReadMode


stop_event = threading.Event()
jpg_bytearr = None
jpg_data_ready = False
img_save_flag = False


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

    for i in itertools.count() :
        if jpg_data_ready:
            raw_img = decode_jpeg(torch.from_numpy(jpg_bytearr), mode=ImageReadMode.UNCHANGED, device='cuda')  # for nvidia JPEG decoding of x86-64 platform
            raw_img = raw_img.permute(1, 2, 0).cpu().numpy()  # convert to numpy array
            # raw_img = nv_jpeg.decode(compressed_img)  # for nvidia JPEG decoding of jetson
            # raw_img = cv2.imdecode(np.frombuffer(compressed_img, dtype=np.uint8), cv2.IMREAD_COLOR)  # for jpeg-turbo of opencv decoding
            resized_img = resize_to_fit_display(raw_img, display_width, display_height)

            cv2.imshow(winname='img', mat=resized_img)

            if img_save_flag:
                cv2.imwrite(f'save_img\\output_{i}.jpg', raw_img)

            if cv2.waitKey(1) == 'q':
                break

    stop_event.set()
    receive_img_thread.join()
    cv2.destroyAllWindows()