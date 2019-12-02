import numpy as np
import cv2
from numba import autojit
import copy
from scipy import misc
"""stream = cv2.VideoCapture(0)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
stream.set(cv2.CAP_PROP_FPS,60)"""


@autojit
def frame_resizer(frame, height):
    weight_size = np.divide(height, frame.shape[0])
    dim = (int(np.multiply(frame.shape[1], weight_size)), height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)
    return resized


"""while True:
    ret,image = stream.read()
    cv2.imshow("Original", image)
    # print("orgin", image.shape)
    resized = frame_resizer(image, 300)
    print(resized.shape)
    cv2.imshow("Resized (Height) ", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
stream.release()
cv2.waitKey(0)"""


def crop(self, image, bbox, face_crop_size, face_crop_margin):
    """
    Crop an image according to a bounding box
    :param bbox: the bounding box
    :return: the cropped image
    """
    _image = copy.deepcopy(image)
    bounding_box = np.zeros(4, dtype=np.int32)
    img_size = np.asarray(_image.shape)[0:2]
    bounding_box[0] = np.maximum(bbox[0] - face_crop_margin / 2, 0)
    bounding_box[1] = np.maximum(bbox[1] - face_crop_margin / 2, 0)
    bounding_box[2] = np.minimum(bbox[2] + face_crop_margin / 2, img_size[1])
    bounding_box[3] = np.minimum(bbox[3] + face_crop_margin / 2, img_size[0])
    cropped = _image[bounding_box[1]:bounding_box[3],
                     bounding_box[0]:bounding_box[2], :]
    cropped = misc.imresize(
        cropped, (face_crop_size, face_crop_size), interp='bilinear')
    return cropped