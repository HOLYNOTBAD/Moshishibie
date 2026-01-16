import cv2
import os

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])
OPENCV_MINOR_VERSION = int(cv2.__version__.split('.')[1])

def is_inside(i, o):
    ix, iy, iw, ih = i
    ox, oy, ow, oh = o
    return ix > ox and ix + iw < ox + ow and \
        iy > oy and iy + ih < oy + oh

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Construct absolute path to image relative to this script
img_path = os.path.join(os.path.dirname(__file__), 'data', 'haying.jpg')
img = cv2.imread(img_path)

if OPENCV_MAJOR_VERSION >= 5 or \
        (OPENCV_MAJOR_VERSION == 4 and OPENCV_MINOR_VERSION >= 6):
    # OpenCV 4.6 or a later version is being used.
    found_rects, found_weights = hog.detectMultiScale(
        img, winStride=(4, 4), scale=1.02, groupThreshold=1.9)
else:
    # OpenCV 4.5 or an earlier version is being used.
    # The groupThreshold parameter used to be named finalThreshold.
    found_rects, found_weights = hog.detectMultiScale(
        img, winStride=(4, 4), scale=1.02, finalThreshold=1.9)

found_rects_filtered = []
found_weights_filtered = []
for ri, r in enumerate(found_rects):
    for qi, q in enumerate(found_rects):
        if ri != qi and is_inside(r, q):
            break
    else:
        found_rects_filtered.append(r)
        found_weights_filtered.append(found_weights[ri])

for ri, r in enumerate(found_rects_filtered):
    x, y, w, h = r
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = '%.2f' % found_weights_filtered[ri]
    cv2.putText(img, text, (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow('Women in Hayfield Detected', img)

# Create result directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), 'result')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'women_in_hayfield_detected.png')
cv2.imwrite(output_path, img)
cv2.waitKey(0)
