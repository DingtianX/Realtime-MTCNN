import numpy as np
from threading import Thread, Lock
import ctypes
import os
import time
import argparse
import imutils
import cv2
import numba
from align import detect_face,_detect_face
import tensorflow as tf
import sys
import resize
from scipy import misc

sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10"  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "0" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "20"  # export NUMEXPR_NUM_THREADS=6
#parser = argparse.ArgumentParser()
#parser.add_argument("--img", type = str, required=True)
#args = parser.parse_args()


# some constants kept as default from facenet
minsize = 50
threshold = [0.55, 0.60, 0.65]
#threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 0
face_crop_size=160
gpu_memory_fraction = 1.0

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)


def getFace(img, compress_frame=True):
    faces = []
    bounding_boxes = None
    #print (type(img))
    img_size = img.shape[0:2]
    if compress_frame is True:
        resize_img = resize.frame_resizer(img, 300)
        bounding_boxes, points = _detect_face.detect_face(
            resize_img, minsize, pnet, rnet, onet, threshold, factor)
        bounding_boxes = bbox_resizer(
            img, bounding_boxes, size_frame=resize_img.shape[0:2])
        #img = np.resize(img, (288, 352))
    #print (img.shape)
    # print(img_size)
    else:
        bounding_boxes, _ = detect_face.detect_face(
            img, minsize, pnet, rnet, onet, threshold, factor)
    #print (bounding_boxes.shape)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                #det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                _margin = int(np.divide(margin, 2))
                bb[0] = np.maximum(np.subtract(
                    face[0], _margin), -float(img_size[1]))
                bb[1] = np.maximum(np.subtract(
                    face[1], _margin), -float(img_size[0]))
                bb[2] = np.minimum(np.add(face[2], _margin),
                                   float(img_size[1]))
                bb[3] = np.minimum(np.add(face[3], _margin),
                                   float(img_size[0]))
                cropped = img[bb[1]: bb[3], bb[0]: bb[2], :]
                #_cropped = misc.imresize(
                #    cropped, (face_crop_size, face_crop_size), interp='bilinear')
                #_cropped=cv2.resize(cropped,(face_crop_size,face_crop_size),interpolation=cv2.INTER_LINEAR)
                #print (cropped.shape)
                #resized = cv2.resize(cropped, dsize=(input_image_size, input_image_size))
                #cv2.imshow("resize", _cropped)
                #cv2.imshow("crop",cropped)
                # print(resized.shape)
                faces.append({'face': cropped, 'rect': [
                             bb[0], bb[1], bb[2], bb[3]], 'acc': face[4]})
                #print (faces)
    return faces, points


#@numba.autojit
def bbox_resizer(frame, bboxs, size_frame):
    # imageToPredict = cv2.imread("img.jpg", 3)
    # Note: flipped comparing to your original code!
    # x_ = imageToPredict.shape[0]
    # y_ = imageToPredict.shape[1]
    new_bbox = []
    for bbox in bboxs:
        y_ = frame.shape[0]
        x_ = frame.shape[1]
       
        x_scale = np.divide(x_,size_frame[1])
        y_scale = np.divide(y_,size_frame[0])
        origLeft, origTop, origRight, origBottom = bbox[0:4]
        x = int(np.round(np.multiply(origLeft, x_scale)))
        y = int(np.round(np.multiply(origTop, y_scale)))
        xmax = int(np.round(np.multiply(origRight, x_scale)))
        ymax = int(np.round(np.multiply(origBottom, y_scale)))
        new_bbox.append([x, y, xmax, ymax, bbox[4]])
        
    return np.asarray(new_bbox)


class WebcamVideoStream:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                frame = np.array(frame)
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()


if __name__ == "__main__":
    vs = WebcamVideoStream(src=1, width=1920, height=1080, fps=30).start()
    while True:
        frame = vs.read()
        timestart = time.clock()
        faces,points = getFace(frame, compress_frame=True)
        for face in faces:
            cv2.rectangle(frame, (face['rect'][0], face['rect'][1]),
                          (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
            # landmarks 绘制人脸关键点
        # for p in points.T:
        #     for i in range(5):
        #         cv2.circle(frame, (p[i], p[i + 5]), 1, (225, 0, 0), 2)
        cv2.imshow("faces", frame)
        print(time.clock() - timestart)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vs.stop()
    vs.__exit__()
    cv2.destroyAllWindows()
