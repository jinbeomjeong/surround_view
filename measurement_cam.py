import carla, pygame, cv2, imagezmq, socket, zmq
import numpy as np

from utils.carla_utils import initialize_simulation, spawn_camera, carla_img_to_rgb_array, main_view_render, ego_vehicle_manual_control


display_width, display_height = 1920, 1080
screen = pygame.display.set_mode((display_width, display_height),  pygame.HWSURFACE | pygame.DOUBLEBUF)
img_sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')
host_name = socket.gethostname()

camera_set = {'cam_1': np.array([]), 'cam_2': np.array([]), 'cam_3': np.array([]), 'cam_4': np.array([])}

surr_cam_pitch_angle = -50
total_cam_img = np.zeros(shape=(1080 * 2, 1920 * 2, 3), dtype=np.uint8)


def main_view_render_callback(img_inst):
    rgb_array = carla_img_to_rgb_array(img_inst)
    main_view_render(cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB), screen)

def camera_callback(img_inst, cameras_name: str):
    global camera_set
    camera_set[cameras_name] = carla_img_to_rgb_array(img_inst)


def main():
    global total_cam_img

    pygame.init()
    pygame.font.init()

    # Pygame 디스플레이 설정
    pygame.display.set_caption("Carla Simulator")

    client, world, camera_bp, vehicle = initialize_simulation()

    main_view_camera = spawn_camera(world, camera_bp, vehicle, x_pos=-5, y_pos=0, z_pos=2.5, pitch=-10, yaw=0, fov=90)
    cam_1 = spawn_camera(world, camera_bp, vehicle, x_pos=2.4, y_pos=0, z_pos=1, pitch=surr_cam_pitch_angle, yaw=0, fov=128)  # front camera
    cam_2 = spawn_camera(world, camera_bp, vehicle, x_pos=0, y_pos=-1, z_pos=1, pitch=surr_cam_pitch_angle, yaw=-90, fov=128) # left camera
    cam_3 = spawn_camera(world, camera_bp, vehicle, x_pos=0, y_pos=1, z_pos=1, pitch=surr_cam_pitch_angle, yaw=90, fov=128)  # right camera
    cam_4 = spawn_camera(world, camera_bp, vehicle, x_pos=-2.4, y_pos=0, z_pos=1, pitch=surr_cam_pitch_angle, yaw=180, fov=128)  # rear camera

    clock = pygame.time.Clock()

    # 차량 제어 객체 생성
    control = carla.VehicleControl()

    # 카메라 데이터 콜백 함수 설정
    main_view_camera.listen(main_view_render_callback)
    cam_1.listen(lambda img: camera_callback(img_inst=img, cameras_name='cam_1'))
    cam_2.listen(lambda img: camera_callback(img_inst=img, cameras_name='cam_2'))
    cam_3.listen(lambda img: camera_callback(img_inst=img, cameras_name='cam_3'))
    cam_4.listen(lambda img: camera_callback(img_inst=img, cameras_name='cam_4'))

    try:
        while True:
            bird_eye_img_list = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # 키 입력 처리
            pressed_key = pygame.key.get_pressed()
            ego_vehicle_manual_control(key=pressed_key, control=control)

            # 차량 제어 적용
            vehicle.apply_control(control)

            front_cam_arr = camera_set['cam_1']
            left_cam_arr = camera_set['cam_2']
            right_cam_arr = camera_set['cam_3']
            rear_cam_arr = camera_set['cam_4']

            if np.all([front_cam_arr.shape[0] > 0, left_cam_arr.shape[0] > 0,
                       right_cam_arr.shape[0] > 0, rear_cam_arr.shape[0] > 0]):
                cam_front_rear = np.hstack([front_cam_arr, rear_cam_arr])
                cam_left_rear = np.hstack([left_cam_arr, right_cam_arr])
                total_cam_img[:] = np.vstack((cam_front_rear, cam_left_rear))

                img_sender.send_image(msg=host_name, image=total_cam_img)
                img_sender.zmq_socket.setsockopt(zmq.LINGER, 0)  # prevents ZMQ error on exit

            if pressed_key[pygame.K_c]:
                cv2.imwrite('save_img\\front_img.jpg', front_cam_arr)
                cv2.imwrite('save_img\\left_img.jpg', left_cam_arr)
                cv2.imwrite('save_img\\right_img.jpg', right_cam_arr)
                cv2.imwrite('save_img\\rear_img.jpg', rear_cam_arr)

            # 화면 업데이트
            clock.tick(30)

    finally:
        print("Cleaning up...")
        main_view_camera.stop()

        for cam in [cam_1, cam_2, cam_3, cam_4]:
            cam.stop()

        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()

