import cv2
import numpy as np


w, h = 1920, 1080
dst_roi = (0, 0, int(w * 1.1), int(h * 1.8))
blender = cv2.detail_MultiBandBlender()
blender.setNumBands(1)

gpu_img = cv2.cuda_GpuMat()
result_img = np.empty(shape=(w, w, 3), dtype=np.uint8)


def convert_bird_eye_view(img):
    h, w, c = img.shape

    src_h_ratio = 0.08
    src_w_ratio_offset = 0.104

    dst_h_ratio = 0.6
    dst_w_ratio = 0.0463

    src_pts = np.array([[w * (0.5 - src_w_ratio_offset), h * src_h_ratio],
                        [w * (0.5 + src_w_ratio_offset), h * src_h_ratio],
                        [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array([[w * 0.4, 0], [w * 0.6, 0], [w * (0.5 + dst_w_ratio), h * dst_h_ratio],
                        [w * (0.5 - dst_w_ratio), h * dst_h_ratio]], dtype=np.float32)

    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    dst_img = cv2.warpPerspective(src=img, M=m, dsize=(w, int(h*dst_h_ratio)), flags=cv2.INTER_LINEAR,
                                  borderValue=(0, 0, 0))

    return dst_img


def top_img_to_bird_eye_view_cuda(top_img: np.array):
    gpu_img.upload(top_img)

    h, w = top_img.shape[:2]

    # 변환에 필요한 비율 계산
    src_h_ratio = 0.08
    src_w_ratio_offset = 0.104

    dst_h_ratio = 0.6
    dst_w_ratio = 0.0463

    # 소스와 대상 좌표 설정
    src_pts = np.array([[w * (0.5 - src_w_ratio_offset), h * src_h_ratio],
                        [w * (0.5 + src_w_ratio_offset), h * src_h_ratio],
                        [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array([[w * 0.4, 0], [w * 0.6, 0], [w * (0.5 + dst_w_ratio), h * dst_h_ratio],
                        [w * (0.5 - dst_w_ratio), h * dst_h_ratio]], dtype=np.float32)

    # Perspective transformation matrix 계산
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # CUDA 기반 warpPerspective 사용
    dst_size = (w, int(h * dst_h_ratio))
    gpu_dst_img = cv2.cuda.warpPerspective(gpu_img, m, dst_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    gpu_dst_gray_img = cv2.cuda.cvtColor(gpu_dst_img, cv2.COLOR_BGR2GRAY)
    _, gpu_dst_bin_img = cv2.cuda.threshold(gpu_dst_gray_img, 1, 255, cv2.THRESH_BINARY)

    return gpu_dst_img.download(), gpu_dst_bin_img.download()


def left_img_to_bird_eye_view_cuda(left_img: np.array):
    gpu_img.upload(left_img)

    h, w = left_img.shape[:2]

    # 변환에 필요한 비율 계산
    src_h_ratio = 0.08
    src_w_ratio_offset = 0.104

    dst_h_ratio = 0.6
    dst_w_ratio = 0.0463

    # 소스와 대상 좌표 설정
    src_pts = np.array([[w * (0.5 - src_w_ratio_offset), h * src_h_ratio],
                        [w * (0.5 + src_w_ratio_offset), h * src_h_ratio],
                        [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array([[w * 0.4, 0], [w * 0.6, 0], [w * (0.5 + dst_w_ratio), h * dst_h_ratio],
                        [w * (0.5 - dst_w_ratio), h * dst_h_ratio]], dtype=np.float32)

    # Perspective transformation matrix 계산
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # CUDA 기반 warpPerspective 사용
    dst_size = (w, int(h * dst_h_ratio))
    gpu_dst_img = cv2.cuda.warpPerspective(gpu_img, m, dst_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1)

    rotation_matrix[0, 2] = 0
    rotation_matrix[1, 2] = w

    gpu_dst_img = cv2.cuda.warpAffine(src=gpu_dst_img, M=rotation_matrix, dsize=(h, w), flags=cv2.INTER_LINEAR,
                                      borderValue=(0, 0, 0))

    gpu_dst_gray_img = cv2.cuda.cvtColor(gpu_dst_img, cv2.COLOR_BGR2GRAY)
    _, gpu_dst_bin_img = cv2.cuda.threshold(gpu_dst_gray_img, 1, 255, cv2.THRESH_BINARY)

    return gpu_dst_img.download(), gpu_dst_bin_img.download()


def right_img_to_bird_eye_view_cuda(right_img: np.array):
    gpu_img.upload(right_img)

    h, w = right_img.shape[:2]

    # 변환에 필요한 비율 계산
    src_h_ratio = 0.08
    src_w_ratio_offset = 0.104

    dst_h_ratio = 0.6
    dst_w_ratio = 0.0463

    # 소스와 대상 좌표 설정
    src_pts = np.array([[w * (0.5 - src_w_ratio_offset), h * src_h_ratio],
                        [w * (0.5 + src_w_ratio_offset), h * src_h_ratio],
                        [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array([[w * 0.4, 0], [w * 0.6, 0], [w * (0.5 + dst_w_ratio), h * dst_h_ratio],
                        [w * (0.5 - dst_w_ratio), h * dst_h_ratio]], dtype=np.float32)

    # Perspective transformation matrix 계산
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # CUDA 기반 warpPerspective 사용
    dst_size = (w, int(h * dst_h_ratio))
    gpu_dst_img = cv2.cuda.warpPerspective(gpu_img, m, dst_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 270, 1)

    rotation_matrix[0, 2] = h
    rotation_matrix[1, 2] = 0

    gpu_dst_img = cv2.cuda.warpAffine(src=gpu_dst_img, M=rotation_matrix, dsize=(h, w), flags=cv2.INTER_LINEAR,
                                      borderValue=(0, 0, 0))

    gpu_dst_gray_img = cv2.cuda.cvtColor(gpu_dst_img, cv2.COLOR_BGR2GRAY)
    _, gpu_dst_bin_img = cv2.cuda.threshold(gpu_dst_gray_img, 1, 255, cv2.THRESH_BINARY)

    return gpu_dst_img.download(), gpu_dst_bin_img.download()


def bottom_img_to_bird_eye_view_cuda(bottom_img: np.array):
    gpu_img.upload(bottom_img)

    h, w = bottom_img.shape[:2]

    # 변환에 필요한 비율 계산
    src_h_ratio = 0.08
    src_w_ratio_offset = 0.104

    dst_h_ratio = 0.6
    dst_w_ratio = 0.0463

    # 소스와 대상 좌표 설정
    src_pts = np.array([[w * (0.5 - src_w_ratio_offset), h * src_h_ratio],
                        [w * (0.5 + src_w_ratio_offset), h * src_h_ratio],
                        [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array([[w * 0.4, 0], [w * 0.6, 0], [w * (0.5 + dst_w_ratio), h * dst_h_ratio],
                        [w * (0.5 - dst_w_ratio), h * dst_h_ratio]], dtype=np.float32)

    # Perspective transformation matrix 계산
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # CUDA 기반 warpPerspective 사용
    dst_size = (w, int(h * dst_h_ratio))
    gpu_dst_img = cv2.cuda.warpPerspective(gpu_img, m, dst_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1)

    gpu_dst_img = cv2.cuda.warpAffine(src=gpu_dst_img, M=rotation_matrix, dsize=(w, h), flags=cv2.INTER_LINEAR,
                                      borderValue=(0, 0, 0))

    gpu_dst_gray_img = cv2.cuda.cvtColor(gpu_dst_img, cv2.COLOR_BGR2GRAY)
    _, gpu_dst_bin_img = cv2.cuda.threshold(gpu_dst_gray_img, 1, 255, cv2.THRESH_BINARY)

    return gpu_dst_img.download(), gpu_dst_bin_img.download()


def blend_bird_eye_img_v1(front_img, left_img, right_img, rear_img):
    blender.prepare(dst_roi)

    left_img_rot = cv2.rotate(left_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    right_img_rot = cv2.rotate(right_img, cv2.ROTATE_90_CLOCKWISE)
    rear_img_rot = cv2.rotate(rear_img, cv2.ROTATE_180)

    gray = cv2.cvtColor(front_img, cv2.COLOR_BGR2GRAY)
    _, front_img_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(left_img_rot, cv2.COLOR_BGR2GRAY)
    _, left_img_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(right_img_rot, cv2.COLOR_BGR2GRAY)
    _, right_img_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(rear_img_rot, cv2.COLOR_BGR2GRAY)
    _, rear_img_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    blender.feed(front_img, front_img_mask, (90, 170))  # 첫 번째 이미지
    blender.feed(left_img_rot, left_img_mask, (350, 0))  # 두 번째 이미지
    blender.feed(right_img_rot, right_img_mask, (1100, 0))
    blender.feed(rear_img_rot, rear_img_mask, (90, 1100))

    result, result_mask = blender.blend(None, None)
    result[result < 0] = 0
    result[result >= 255] = 255

    return result.astype(np.uint8)


def blend_bird_eye_img_v2(front_img, left_img, right_img, rear_img):
    left_img_rot = cv2.rotate(src=left_img, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    right_img_rot = cv2.rotate(src=right_img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    rear_img_rot = cv2.rotate(src=rear_img, rotateCode=cv2.ROTATE_180)

    front_img_top_offset = 170
    left_img_left_offset = 260
    right_img_right_offset = 260

    border_color = (0, 0, 0)

    front_ext = cv2.copyMakeBorder(front_img, front_img_top_offset, 1920 - (front_img.shape[0] + front_img_top_offset),
                                   0, 0,
                                   cv2.BORDER_CONSTANT, value=border_color)

    rear_ext = cv2.copyMakeBorder(rear_img_rot, 1920 - (rear_img.shape[0] + front_img_top_offset), front_img_top_offset,
                                  0, 0,
                                  cv2.BORDER_CONSTANT, value=border_color)

    left_ext = cv2.copyMakeBorder(left_img_rot, 0, 0, left_img_left_offset,
                                  1920 - (left_img_rot.shape[1] + left_img_left_offset),
                                  cv2.BORDER_CONSTANT, value=border_color)

    right_ext = cv2.copyMakeBorder(right_img_rot, 0, 0, 1920 - (right_img_rot.shape[1] + right_img_right_offset),
                                   right_img_right_offset,
                                   cv2.BORDER_CONSTANT, value=border_color)

    gray = cv2.cvtColor(src=front_ext, code=cv2.COLOR_BGR2GRAY)
    _, front_img_mask = cv2.threshold(gray, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    gray = cv2.cvtColor(src=left_ext, code=cv2.COLOR_BGR2GRAY)
    _, left_img_mask = cv2.threshold(gray, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    gray = cv2.cvtColor(src=right_ext, code=cv2.COLOR_BGR2GRAY)
    _, right_img_mask = cv2.threshold(gray, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    gray = cv2.cvtColor(src=rear_ext, code=cv2.COLOR_BGR2GRAY)
    _, rear_img_mask = cv2.threshold(gray, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    a = np.bitwise_and(front_img_mask, left_img_mask)
    b = np.bitwise_and(front_img_mask, right_img_mask)
    c = np.bitwise_and(rear_img_mask, left_img_mask)
    d = np.bitwise_and(rear_img_mask, right_img_mask)

    e = a + b + c + d

    front_ext[e == 255] = (0, 0, 0)
    rear_ext[e == 255] = (0, 0, 0)

    result_img[:] = front_ext + rear_ext + left_ext + right_ext

    return result_img
