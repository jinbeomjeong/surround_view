import imagezmq, cv2, threading, torch, itertools
import numpy as np
from utils.surround_view import get_screen_resolution, resize_to_fit_display
from torchvision.io import decode_jpeg, ImageReadMode
from utils.defisheye import Defisheye
from utils.surround_view import blend_bird_eye_img_fisheye


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
    conv_img_list = []
    dst_img_list = []

    dump_img = np.zeros(shape=(540, 960, 3), dtype=np.uint8)
    defisheye = Defisheye(dump_img, dtype='equalarea', format='fullframe', fov=360, pfov=120, pad=270)

    front_img_cal = np.load("parameter\\fish_eye_img\\front_img_cal.npy")
    rear_img_cal = np.load("parameter\\fish_eye_img\\rear_img_cal.npy")
    left_img_cal = np.load("parameter\\fish_eye_img\\left_img_cal.npy")
    right_img_cal = np.load("parameter\\fish_eye_img\\right_img_cal.npy")

    img_pos = {'front': [[0, 540], [0, 960]],
               'rear': [[0, 540], [960, 1980]],
               'left': [[540, 1080], [0, 960]],
               'right': [[540, 1080], [960, 1980]]}

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
            # resized_img = resize_to_fit_display(raw_img, display_width, display_height)

            conv_img_list.clear()
            dst_img_list.clear()


            front_img = raw_img[img_pos['front'][0][0]:img_pos['front'][0][1],
                        img_pos['front'][1][0]:img_pos['front'][1][1]]
            rear_img = raw_img[img_pos['rear'][0][0]:img_pos['rear'][0][1], img_pos['rear'][1][0]:img_pos['rear'][1][1]]
            left_img = raw_img[img_pos['left'][0][0]:img_pos['left'][0][1], img_pos['left'][1][0]:img_pos['left'][1][1]]
            right_img = raw_img[img_pos['right'][0][0]:img_pos['right'][0][1],
                        img_pos['right'][1][0]:img_pos['right'][1][1]]
            img_list = [front_img, rear_img, left_img, right_img]

            for img in img_list:
                conv_img = defisheye.convert(img)[150:940]
                # conv_img = cv2.resize(src=conv_img, fx=0.5, fy=0.5, dsize=None, interpolation=cv2.INTER_LINEAR)
                conv_img_list.append(conv_img)

            h, w, c = conv_img_list[0].shape
            front_dst_img = cv2.warpPerspective(src=conv_img_list[0], M=front_img_cal, dsize=(w, int(h)),
                                                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))[:, 100:980]

            h, w, c = conv_img_list[1].shape
            rear_dst_img = cv2.warpPerspective(src=conv_img_list[1], M=rear_img_cal, dsize=(w, int(h)),
                                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))[:, 100:980]

            h, w, c = conv_img_list[2].shape
            left_dst_img = cv2.warpPerspective(src=conv_img_list[2], M=left_img_cal, dsize=(w, int(h)),
                                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))[:, 100:980]

            h, w, c = conv_img_list[3].shape
            right_dst_img = cv2.warpPerspective(src=conv_img_list[3], M=right_img_cal, dsize=(w, int(h)),
                                                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))[:, 100:980]

            # cv2.imshow(winname='img', mat=raw_img)
            cv2.imshow('img', blend_bird_eye_img_fisheye(front_dst_img, rear_dst_img, left_dst_img, right_dst_img))
            # cv2.imshow('img', raw_img)
            if img_save_flag:
                cv2.imwrite(f'save_img\\output_{i}.jpg', raw_img)

            if cv2.waitKey(1) == 'q':
                break

    stop_event.set()
    receive_img_thread.join()
    cv2.destroyAllWindows()