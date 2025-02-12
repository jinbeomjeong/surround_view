import os, re, cv2
import numpy as np

from utils.defisheye import Defisheye
from utils.surround_view import blend_bird_eye_img_fisheye
from tqdm.auto import tqdm

file_list = os.listdir("e:\\result")
file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

front_img_cal = np.load("parameter\\fish_eye_img\\front_img_cal.npy")
rear_img_cal = np.load("parameter\\fish_eye_img\\rear_img_cal.npy")
left_img_cal = np.load("parameter\\fish_eye_img\\left_img_cal.npy")
right_img_cal = np.load("parameter\\fish_eye_img\\right_img_cal.npy")

img_pos = {'front': [[0, 540], [0, 960]],
           'rear':[[0, 540], [960, 1980]],
           'left': [[540, 1080], [0, 960]],
           'right':[[540, 1080], [960, 1980]]}

cv2.namedWindow("image1", cv2.WINDOW_NORMAL)
cv2.namedWindow("image2", cv2.WINDOW_NORMAL)
cv2.namedWindow("image3", cv2.WINDOW_NORMAL)

img = cv2.imread(f"e:\\result\\{file_list[0]}")
front_img = img[img_pos['front'][0][0]:img_pos['front'][0][1], img_pos['front'][1][0]:img_pos['front'][1][1]]

defisheye = Defisheye(front_img, dtype='equalarea', format='fullframe', fov=360, pfov=120, pad=270)
conv_img_list = []
dst_img_list = []

for file in tqdm(file_list):
    conv_img_list.clear()
    dst_img_list.clear()

    raw_img = cv2.imread(f"e:\\result\\{file}")

    front_img = raw_img[img_pos['front'][0][0]:img_pos['front'][0][1],img_pos['front'][1][0]:img_pos['front'][1][1]]
    rear_img = raw_img[img_pos['rear'][0][0]:img_pos['rear'][0][1], img_pos['rear'][1][0]:img_pos['rear'][1][1]]
    left_img = raw_img[img_pos['left'][0][0]:img_pos['left'][0][1], img_pos['left'][1][0]:img_pos['left'][1][1]]
    right_img = raw_img[img_pos['right'][0][0]:img_pos['right'][0][1], img_pos['right'][1][0]:img_pos['right'][1][1]]
    img_list = [front_img, rear_img, left_img, right_img]

    for img in img_list:
        conv_img = defisheye.convert(img)[150:940]
        #conv_img = cv2.resize(src=conv_img, fx=0.5, fy=0.5, dsize=None, interpolation=cv2.INTER_LINEAR)
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

    cv2.imshow('image1', raw_img)

    cv2.imshow('image2', blend_bird_eye_img_fisheye(front_dst_img, rear_dst_img, left_dst_img, right_dst_img))

    if cv2.waitKey(1) == 'q':
        break

cv2.destroyAllWindows()
