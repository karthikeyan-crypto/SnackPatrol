import os
import sys
import cv2
import time
import tarfile
import numpy as np
import tensorflow as tf

from datetime import datetime
from collections import Counter
from six.moves import urllib


# ============================================================
# USER CONFIGURATION
# ============================================================
# Camera index:
# 0 = default webcam
# 1 = second camera / DroidCam
CAM_INDEX = 0 
# native capture res (see note below)
CAPTURE_W = 640 
CAPTURE_H = 480

PROCESS_EVERY_N = 2 # only run detection every Nth frame

CPU_THREADS = 4 # number of threads to use for CPU processing

LOW_THRESH = 0.30 # below this, a raw detection is ignored entirely
CONFIRM_THRESH = 0.50 # at/above this, a detection is trusted immediately
SMALL_AREA_FRAC = 0.05 # boxes smaller than 5% of the frame = "possibly far away"

IOU_MATCH_THRESH = 0.30 # IoU threshold for matching detections
MIN_HITS_TO_CONFIRM = 3 # minimum number of consecutive detections to confirm an object
MAX_MISSES = 5 # maximum number of consecutive misses before an object is considered lost


# Path to your TensorFlow Object Detection API
# Change this path if your Object Detection API is installed elsewhere.
OD_API_PATH = r"C:\Users\intel\TensorFlow\models-1.12.0\research"

sys.path.insert(0, OD_API_PATH)

from object_detection.utils import label_map_util


# 1. MODEL SETUP

MODEL_NAME = "ssd_mobilenet_v1_coco_2017_11_17"

MODEL_FILE = MODEL_NAME + ".tar.gz"
DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "pretrained_models"
)


PATH_TO_CKPT = os.path.join(
    MODEL_DIR,
    MODEL_NAME,
    "frozen_inference_graph.pb"
)


# Path to COCO label map
PATH_TO_LABELS = os.path.join(
    OD_API_PATH,
    "object_detection",
    "data",
    "mscoco_label_map.pbtxt"
)

NUM_CLASSES = 90

print("Checking for model...")

os.makedirs(MODEL_DIR, exist_ok=True)

PATH_TO_MODEL_FILE = os.path.join(
    MODEL_DIR,
    MODEL_FILE
)

if not os.path.exists(PATH_TO_CKPT):
    print("Downloading model...")

    urllib.request.urlretrieve(
        DOWNLOAD_BASE + MODEL_FILE,
        PATH_TO_MODEL_FILE
    )

    with tarfile.open(PATH_TO_MODEL_FILE) as tar_file:
        tar_file.extractall(MODEL_DIR)

    print("Model downloaded and extracted.")
else:
    print("Model already exists.")

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)

print("Loading TensorFlow graph...")

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
        od_graph_def.ParseFromString(fid.read())
    tf.import_graph_def(od_graph_def, name="")

print("Graph loaded.")


# 2. FOOD JUDGE 


HEALTHY_IDS = {52, 53, 55, 56, 57}   # banana, apple, orange, broccoli, carrot
JUNK_IDS    = {58, 59, 60, 61}       # hot dog, pizza, donut, cake
DRINK_IDS   = {44, 46, 47}           # bottle, wine glass, cup

WATCHED_IDS = HEALTHY_IDS | JUNK_IDS | DRINK_IDS

def judge(class_id):
    if class_id in HEALTHY_IDS:
        return "HEALTHY", (0, 200, 0)
    if class_id in JUNK_IDS:
        return "JUNK FOOD", (0, 0, 255)
    if class_id in DRINK_IDS:
        return "DRINK", (255, 150, 0)
    return None, None




# NOTE on resolution: the SSD graph internally resizes whatever
# you feed it to a fixed 300x300 before the actual convolutions
# run. That means capturing at a HIGHER native resolution barely
# slows inference down (the heavy math still runs at 300x300),
# but it does preserve more real detail for small/far objects
# before that internal resize happens. That's why this version
# defaults to 640x480 instead of 320x240 - it's close to free
# accuracy. If your webcam or CPU can't keep up, drop it to
# 480x360 first before touching PROCESS_EVERY_N.


# 4. CPU-ONLY SESSION CONFIG
# use when you have no GPU or don't want to use it. If you have a GPU and want to use it, remove this section entirely.

cpu_config = tf.ConfigProto(
    device_count={"GPU": 0},
    intra_op_parallelism_threads=CPU_THREADS,
    inter_op_parallelism_threads=CPU_THREADS,
)
sess = tf.Session(graph=detection_graph, config=cpu_config)

image_tensor      = detection_graph.get_tensor_by_name("image_tensor:0")
detection_boxes   = detection_graph.get_tensor_by_name("detection_boxes:0")
detection_scores  = detection_graph.get_tensor_by_name("detection_scores:0")
detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
num_detections    = detection_graph.get_tensor_by_name("num_detections:0")

def run_inference(image_np):
    """One forward pass. Returns squeezed boxes/scores/classes."""
    expanded = np.expand_dims(image_np, axis=0)
    boxes, scores, classes, num = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: expanded},
    )
    return (
        np.squeeze(boxes),
        np.squeeze(scores),
        np.squeeze(classes).astype(np.int32),
    )


# 5. ZOOM-AND-REVERIFY  (fixes: poor accuracy on small/far objects)

# For a detection that's small and only medium-confidence, crop
# just that region out of the full-res frame, blow it up so the
# object fills the frame like a digital zoom, and run a SECOND,
# focused inference pass on just that patch. This gives the
# model a much bigger, cleaner view of a distant object instead
# of relying on a tiny cluster of pixels in the full scene.

def zoom_and_reverify(frame_rgb, box, pad=0.20):
    h, w, _ = frame_rgb.shape
    ymin, xmin, ymax, xmax = box
    box_h, box_w = ymax - ymin, xmax - xmin

    ymin_p = max(0.0, ymin - box_h * pad)
    xmin_p = max(0.0, xmin - box_w * pad)
    ymax_p = min(1.0, ymax + box_h * pad)
    xmax_p = min(1.0, xmax + box_w * pad)

    y1, y2 = int(ymin_p * h), int(ymax_p * h)
    x1, x2 = int(xmin_p * w), int(xmax_p * w)

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    zoomed = cv2.resize(crop, (300, 300))
    boxes, scores, classes = run_inference(zoomed)

    best_i = int(np.argmax(scores))
    if scores[best_i] < CONFIRM_THRESH or classes[best_i] not in WATCHED_IDS:
        return None

    # Keep the ORIGINAL box position (it's where the object actually
    # is in the full frame) but trust the zoomed pass's class + score.
    return classes[best_i], float(scores[best_i])


# 6. LIGHTWEIGHT TRACKER  (fixes: double counting, no tracking,
#    stale boxes, snapshot spam, class flicker)


def iou(box_a, box_b):
    ya, xa, yb, xb = box_a
    yc, xc, yd, xd = box_b

    inter_y1, inter_x1 = max(ya, yc), max(xa, xc)
    inter_y2, inter_x2 = min(yb, yd), min(xb, xd)

    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_area = inter_h * inter_w

    area_a = max(0.0, yb - ya) * max(0.0, xb - xa)
    area_b = max(0.0, yd - yc) * max(0.0, xd - xc)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


class Track:
    _next_id = 1

    def __init__(self, box, cls, score):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.class_votes = Counter({cls: 1})
        self.last_score = score
        self.hits = 1
        self.misses = 0
        self.counted = False

    def update(self, box, cls, score):
        self.box = box
        self.class_votes[cls] += 1
        self.last_score = score
        self.hits += 1
        self.misses = 0

    @property
    def class_id(self):
        return self.class_votes.most_common(1)[0][0]


def update_tracks(tracks, detections):
    """detections: list of (box, score, class_id) for this processed frame."""

    pairs = []
    for ti, track in enumerate(tracks):
        for di, (box, score, cls) in enumerate(detections):
            score_iou = iou(track.box, box)
            if score_iou >= IOU_MATCH_THRESH:
                pairs.append((score_iou, ti, di))
    pairs.sort(key=lambda p: p[0], reverse=True)

    used_tracks, used_dets = set(), set()
    for _, ti, di in pairs:
        if ti in used_tracks or di in used_dets:
            continue
        box, score, cls = detections[di]
        tracks[ti].update(box, cls, score)
        used_tracks.add(ti)
        used_dets.add(di)

    for ti, track in enumerate(tracks):
        if ti not in used_tracks:
            track.misses += 1

    for di, (box, score, cls) in enumerate(detections):
        if di not in used_dets:
            tracks.append(Track(box, cls, score))

    tracks[:] = [t for t in tracks if t.misses <= MAX_MISSES]
    return tracks


# 7. LIVE WEBCAM LOOP


SNACK_LOG_DIR = os.path.join(
    PROJECT_DIR,
    "snack_log"
)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Could not open Cam.")
    raise RuntimeError(f"Camera {CAM_INDEX} is not available.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
tracks = []
junk_score = 0
healthy_score = 0
frame_count = 0
t_last = time.time()
fps = 0.0

print("\nSnack Patrol v2 is watching.")
print("Press 'q' to quit, 'r' to reset scores.\n")

while True:

    ret, frame = cap.read()
    if not ret:
        print("Could not read from webcam.")
        break

    frame_count += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    if frame_count % PROCESS_EVERY_N == 0:

        boxes, scores, classes = run_inference(frame_rgb)

        candidates = []
        for box, score, cls in zip(boxes, scores, classes):
            if cls not in WATCHED_IDS or score < LOW_THRESH:
                continue

            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

            if score >= CONFIRM_THRESH:
                candidates.append((box, float(score), int(cls)))
            elif area < SMALL_AREA_FRAC:
                # Possibly distant/small - give it a focused second look
                verified = zoom_and_reverify(frame_rgb, box)
                if verified is not None:
                    v_cls, v_score = verified
                    candidates.append((box, v_score, v_cls))
            # else: low confidence AND not small -> treat as noise, discard

        tracks = update_tracks(tracks, candidates)

        # Score + snapshot EXACTLY ONCE per confirmed track
        for track in tracks:
            if track.hits >= MIN_HITS_TO_CONFIRM and not track.counted:
                label, _ = judge(track.class_id)
                if label == "JUNK FOOD":
                    junk_score += 1
                elif label == "HEALTHY":
                    healthy_score += 1
                track.counted = True

                if label is not None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = os.path.join(
    SNACK_LOG_DIR,
    "{}_track{}_{}.jpg".format(
        ts,
        track.id,
        label.replace(" ", "")
    )
)
                    cv2.imwrite(fname, frame)


    # Draw current tracks
   
    for track in tracks:
        if track.misses > 0:
            continue  # don't draw boxes we haven't reconfirmed recently

        label, color = judge(track.class_id)
        if label is None:
            continue

        ymin, xmin, ymax, xmax = track.box
        p1 = (int(xmin * w), int(ymin * h))
        p2 = (int(xmax * w), int(ymax * h))

        cv2.rectangle(frame, p1, p2, color, 2)
        status = "" if track.counted else " (confirming...)"
        text = "{} {}%{}".format(label, int(track.last_score * 100), status)
        cv2.putText(frame, text, (p1[0], max(p1[1] - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if any(judge(t.class_id)[0] == "JUNK FOOD" and t.misses == 0 for t in tracks):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        cv2.putText(frame, "SNACK ALERT!", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Junk: {}  Healthy: {}  (unique items, 'r' to reset)".format(junk_score, healthy_score),
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-6))
    t_last = now
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("Snack Patrol v2", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        junk_score = 0
        healthy_score = 0
        tracks = []
        print("Scores reset.")


# 8. WRAP UP


cap.release()
cv2.destroyAllWindows()
sess.close()

print("\n======================================")
print("Snack Patrol v2 session ended")
print("\nUnique healthy items counted:", healthy_score)
print("Unique junk food items counted:", junk_score)
print("Snapshots saved to: snack_log/")
print("======================================")