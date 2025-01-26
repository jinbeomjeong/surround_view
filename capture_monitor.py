import dxcam, cv2, threading, keyboard, imagezmq
import numpy as np


print(dxcam.device_info())
monitor_list = []
screen_img_list = []
n_of_monitor = 2

for i in range(2):
    monitor = dxcam.create(output_idx=i, output_color='BGR')
    monitor_list.append(monitor)
    monitor.start(target_fps=30, video_mode=True)
    screen_img_list.append(np.empty(shape=monitor.get_latest_frame().shape, dtype=np.uint8))


def capture_monitor(id=0):
    global screen_img_list
    while monitor_list[id].is_capturing:
        screen_img_list[id][:] = monitor_list[id].get_latest_frame()

for i in range(n_of_monitor):
    monitor_capture = threading.Thread(target=capture_monitor, args=(i, ))
    monitor_capture.daemon = True
    monitor_capture.start()

if __name__ == "__main__":
    sender = imagezmq.ImageSender(connect_to="tcp://192.168.137.7:5555")

    while True:
        total_img = np.hstack(screen_img_list)
        compressed_img = cv2.imencode(ext='.jpg', img=total_img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])[1]
        sender.send_image(msg='img', image=compressed_img)

        if keyboard.is_pressed('q'): break

    for monitor in monitor_list:
        monitor.stop()

    sender.close()

    print("Program terminated.")
