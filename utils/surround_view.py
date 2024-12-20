import cv2
import numpy as np


w, h = 1920, 1080
dst_roi = (0, 0, int(w * 1.2), int(h * 2))
blender = cv2.detail_MultiBandBlender()
blender.setNumBands(1)


def convert_bird_eye_view(img):
    h, w, c = img.shape

    dst_h_ratio = 0.3

    src_pts = np.array([[w * 0.40, h * 0.6], [w * 0.60, h * 0.6], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array([[w*0.4, 0], [w*0.6, 0], [w*0.6, h*dst_h_ratio], [w*0.4, h*dst_h_ratio]], dtype=np.float32)

    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    dst_img = cv2.warpPerspective(src=img, M=m, dsize=(w, int(h*dst_h_ratio)), flags=cv2.INTER_LINEAR)

    return dst_img


def blend_bird_eye_img(front_img, left_img, right_img, rear_img):
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

    blender.feed(front_img, front_img_mask, (90, 280))  # 첫 번째 이미지
    blender.feed(left_img_rot, left_img_mask, (390, 30))  # 두 번째 이미지
    blender.feed(right_img_rot, right_img_mask, (1390, 30))
    blender.feed(rear_img_rot, rear_img_mask, (90, 1360))

    result, result_mask = blender.blend(None, None)
    result[result < 0] = 0
    result[result >= 255] = 255

    return result.astype(np.uint8)
