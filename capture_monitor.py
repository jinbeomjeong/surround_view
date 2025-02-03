import dxcam, cv2, threading, keyboard, imagezmq, torch, itertools
import numpy as np
from torchvision.io import encode_jpeg


print(dxcam.device_info())

monitor_list = []
screen_img_list = []
n_of_monitor = 4
comp_img_ready = False

stop_event = threading.Event()
sender = imagezmq.ImageSender(connect_to="tcp://192.168.0.1:5555")


def capture_monitor(monitor_index=0):
    global screen_img_list

    while monitor_list[monitor_index].is_capturing:
        screen_img_list[monitor_index][:] = monitor_list[monitor_index].get_latest_frame()


# def send_img():
#     while not stop_event.is_set():
#         if comp_img_ready:
#             sender.send_jpg(msg='img', jpg_buffer=compressed_img)
#
#     sender.close()


if __name__ == "__main__":
    for monitor_idx in range(n_of_monitor):
        monitor = dxcam.create(output_idx=monitor_idx, output_color='BGR')
        monitor_list.append(monitor)
        monitor.start(target_fps=30, video_mode=True, region=(0, 0, 540, 960))
        screen_img_list.append(np.empty(shape=monitor.get_latest_frame().shape, dtype=np.uint8))

        monitor_capture = threading.Thread(target=capture_monitor, args=(monitor_idx,))
        monitor_capture.daemon = True
        monitor_capture.start()

    for i in itertools.count():
        img1 = np.hstack(screen_img_list[0:2])
        img2 = np.hstack(screen_img_list[2:4])
        total_img = np.vstack((img1, img2))

        img_tensor = torch.from_numpy(total_img).to('cuda', dtype=torch.uint8).permute(2, 0, 1)  # convert to img tensor based cuda
        compressed_img = encode_jpeg(input=img_tensor , quality=100).cpu().numpy()  # for encode jpeg of cuda
        # compressed_img = cv2.imencode(ext='.jpg', img=total_img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])[1]  # for encode jpeg of cpu
        # comp_img_ready = True

        sender.send_image(msg='img', image=compressed_img)  # send jpg image to receiver

        if keyboard.is_pressed('q'): break

    for monitor in monitor_list:
        monitor.stop()

    print("Program terminated.")
