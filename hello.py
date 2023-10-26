import cv2
import numpy as np
from matplotlib import pyplot as plt

desktop_path = "/Users/sohatakeyama/Desktop/"
img_rgb = cv2.imread(desktop_path + 'apex_sample.jpeg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(desktop_path + 'predator.jpeg', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.2
loc = np.where(res >= threshold)

# 重複を除外して異なる位置にマッチングした結果の数を数える
unique_locations = []
for pt in zip(*loc[::-1]):
    is_unique = True
    for existing_pt in unique_locations:
        if abs(pt[0] - existing_pt[0]) < w and abs(pt[1] - existing_pt[1]) < h:
            is_unique = False
            break
    if is_unique:
        unique_locations.append(pt)

num_unique_matches = len(unique_locations)

print("異なる位置にマッチングされた結果の数:", num_unique_matches)

for pt in unique_locations: cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imwrite(desktop_path + 'res.png', img_rgb)
