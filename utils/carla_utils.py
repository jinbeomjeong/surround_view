import carla
import pygame
import numpy as np


# 초기화 함수: Carla 서버 연결 및 차량 생성
def initialize_simulation():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 차량 스폰 지점 가져오기
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprint = blueprint_library.find('vehicle.tesla.model3')
    spawn_point = world.get_map().get_spawn_points()[0]

    # 차량 스폰
    vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
    return client, world, vehicle


# 카메라 부착 함수: 차량에 3인칭 RGB 카메라 추가
def spawn_camera(world, vehicle, x_pos=0, y_pos=0, z_pos=0, pitch=0, yaw=0, fov=90):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')

    # 카메라 설정 (해상도와 FOV 설정 가능)
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', str(fov))  # 넓은 시야각 설정
    #camera_bp.set_attribute('gamma', '2.2')  # 감마 값 조정

    # 카메라 위치 설정 (차량 뒤쪽 위로 약간 떨어지게 배치)
    spawn_point = carla.Transform(carla.Location(x=x_pos, y=y_pos, z=z_pos),carla.Rotation(pitch=pitch, yaw=yaw))      # 약간 아래를 바라보도록 회전

    # 차량에 카메라 부착
    camera = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)

    return camera


# 이미지 처리 함수: 카메라 데이터 -> Pygame 화면
def carla_img_to_rgb_array(image_inst):
    array = np.frombuffer(image_inst.raw_data, dtype=np.uint8)
    array = array.reshape((image_inst.height, image_inst.width, 4))  # RGBA 포맷
    rgb_array = array[:, :, [2, 1, 0]]

    return rgb_array


def main_view_render(rgb_array, screen):
    # Pygame 표준 형식으로 변환
    surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
