import carla, pygame, cv2
import numpy as np

from utils.carla_utils import initialize_simulation, spawn_camera, carla_img_to_rgb_array, main_view_render
from utils.surround_view import convert_bird_eye_view, blend_bird_eye_img

display_width, display_height = 1024, 768
screen = pygame.display.set_mode((display_width, display_height),  pygame.HWSURFACE | pygame.DOUBLEBUF)

camera_set = {'cam_1': np.array([]), 'cam_2': np.array([]), 'cam_3': np.array([]), 'cam_4': np.array([])}

surr_cam_pitch_angle = -50

def main_view_render_callback(img_inst):
    rgb_array = carla_img_to_rgb_array(img_inst)
    main_view_render(rgb_array, screen)

def camera_callback(img_inst, cameras_name: str):
    global camera_set
    array = np.frombuffer(img_inst.raw_data, dtype=np.uint8)
    array = array.reshape((img_inst.height, img_inst.width, 4))  # RGBA 포맷
    rgb_array = array[:, :, [0, 1, 2]]
    camera_set[cameras_name] = rgb_array


# 메인 함수
def main():
    pygame.init()
    pygame.font.init()

    # Pygame 디스플레이 설정
    pygame.display.set_caption("Carla Simulator")

    client, world, vehicle = initialize_simulation()
    main_view_camera = spawn_camera(world, vehicle, x_pos=-5, y_pos=0, z_pos=2.5, pitch=-10, yaw=0, fov=90)
    cam_1 = spawn_camera(world, vehicle, x_pos=2.4, y_pos=0, z_pos=1, pitch=surr_cam_pitch_angle, yaw=0, fov=128)  # front camera
    cam_2 = spawn_camera(world, vehicle, x_pos=0, y_pos=-1, z_pos=1, pitch=surr_cam_pitch_angle, yaw=-90, fov=128) # left camera
    cam_3 = spawn_camera(world, vehicle, x_pos=0, y_pos=1, z_pos=1, pitch=surr_cam_pitch_angle, yaw=90, fov=128)  # right camera
    cam_4 = spawn_camera(world, vehicle, x_pos=-2.4, y_pos=0, z_pos=1, pitch=surr_cam_pitch_angle, yaw=180, fov=128)  # rear camera

    clock = pygame.time.Clock()

    # 차량 제어 객체 생성
    control = carla.VehicleControl()

    # 카메라 데이터 콜백 함수 설정
    main_view_camera.listen(main_view_render_callback)
    cam_1.listen(lambda img: camera_callback(img_inst=img, cameras_name='cam_1'))
    cam_2.listen(lambda img: camera_callback(img_inst=img, cameras_name='cam_2'))
    cam_3.listen(lambda img: camera_callback(img_inst=img, cameras_name='cam_3'))
    cam_4.listen(lambda img: camera_callback(img_inst=img, cameras_name='cam_4'))

    #cv2.namedWindow(winname='video', flags=cv2.WINDOW_NORMAL)
    cv2.namedWindow(winname='bev', flags=cv2.WINDOW_NORMAL)

    try:
        while True:
            bird_eye_img_list = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # 키 입력 처리
            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_w]:
                control.throttle = 1.0
            else:
                control.throttle = 0.0

            if keys_pressed[pygame.K_s]:
                control.brake = 1.0
            else:
                control.brake = 0.0

            if keys_pressed[pygame.K_a]:
                control.steer = max(control.steer - 0.05, -1.0)
            elif keys_pressed[pygame.K_d]:
                control.steer = min(control.steer + 0.05, 1.0)
            else:
                control.steer = 0.0

            # 차량 제어 적용
            vehicle.apply_control(control)

            front_cam_arr = camera_set['cam_1']
            left_cam_arr = camera_set['cam_2']
            right_cam_arr = camera_set['cam_3']
            rear_cam_arr = camera_set['cam_4']

            if front_cam_arr.shape[0] > 0 and left_cam_arr.shape[0] > 0 and right_cam_arr.shape[0] > 0 and rear_cam_arr.shape[0] > 0:
                #cam_front_rear = np.hstack([front_cam_arr, rear_cam_arr])
                #cam_left_rear = np.hstack([left_cam_arr, right_cam_arr])
                #total_cam_arr = np.vstack((cam_front_rear, cam_left_rear))

                #cv2.imshow("video", total_cam_arr)

                #if cv2.waitKey(1) & 0xFF == ord('q'): break

                for img in [front_cam_arr, left_cam_arr, right_cam_arr, rear_cam_arr]:
                    bird_eye_img_list.append(convert_bird_eye_view(img))

                result_img = blend_bird_eye_img(*bird_eye_img_list)

                cv2.imshow("bev", result_img)

                if cv2.waitKey(1) & 0xFF == ord('q'): break

            if keys_pressed[pygame.K_c]:
                cv2.imwrite('save_img\\front_img.jpg', front_cam_arr)
                cv2.imwrite('save_img\\left_img.jpg', left_cam_arr)
                cv2.imwrite('save_img\\right_img.jpg', right_cam_arr)
                cv2.imwrite('save_img\\rear_img.jpg', rear_cam_arr)

            # 화면 업데이트
            clock.tick(30)

    finally:
        print("Cleaning up...")
        main_view_camera.stop()
        cam_1.stop()
        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
