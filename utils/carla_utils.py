import carla
import pygame
import numpy as np


# 초기화 함수: Carla 서버 연결 및 차량 생성
def initialize_simulation():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    camera_bp = bp_lib.find('sensor.camera.rgb')

    spawn_point = world.get_map().get_spawn_points()[0]

    # 차량 스폰
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    return client, world, camera_bp, vehicle


# 카메라 부착 함수: 차량에 3인칭 RGB 카메라 추가
def spawn_camera(world, camera_bp, attach_obj, x_pos=0, y_pos=0, z_pos=0, pitch=0, yaw=0, fov=90):
    # 카메라 설정 (해상도와 FOV 설정 가능)
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', str(fov))  # 넓은 시야각 설정
    #camera_bp.set_attribute('gamma', '2.2')  # 감마 값 조정

    # 카메라 위치 설정 (차량 뒤쪽 위로 약간 떨어지게 배치)
    spawn_point = carla.Transform(carla.Location(x=x_pos, y=y_pos, z=z_pos),carla.Rotation(pitch=pitch, yaw=yaw))      # 약간 아래를 바라보도록 회전

    # 차량에 카메라 부착
    camera = world.spawn_actor(camera_bp, spawn_point, attach_to=attach_obj)

    return camera


# 이미지 처리 함수: 카메라 데이터 -> Pygame 화면
def carla_img_to_rgb_array(image_inst):
    array = np.frombuffer(image_inst.raw_data, dtype=np.uint8)
    array = array.reshape((image_inst.height, image_inst.width, 4))  # RGBA 포맷
    rgb_array = array[:, :, 0:3]

    return rgb_array


def main_view_render(rgb_array, screen):
    # Pygame 표준 형식으로 변환
    surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))
    pygame.display.flip()


def ego_vehicle_manual_control(key: pygame.key.get_pressed, control: carla.VehicleControl):
    if key[pygame.K_w]:
        control.throttle = 1.0
    else:
        control.throttle = 0.0

    if key[pygame.K_s]:
        control.brake = 1.0
    else:
        control.brake = 0.0

    if key[pygame.K_a]:
        control.steer = max(control.steer - 0.05, -1.0)
    elif key[pygame.K_d]:
        control.steer = min(control.steer + 0.05, 1.0)
    else:
        control.steer = 0.0
