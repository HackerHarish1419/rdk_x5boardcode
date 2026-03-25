import os
import glob
import time
import numpy as np
import cv2
import multiprocessing
import requests
from bpu_infer_lib_x5 import Infer
from simple_tracker import SimpleIOUTracker

# ==================== Configuration ====================

API_URL = "https://ai.kaaval-ai.com/recognize-plate"
API_KEY = "SECRET_KEY"  # Replace with actual secret key or read from env
CAMERA_ID = "CAM-EDGE-01"
LOCATION = "RDK X5 Deploy"

# Max storage for evidences (in Megabytes)
MAX_STORAGE_MB = 1000 
OUTPUT_DIR = "evidences"

# COCO-like model mapping (as provided by the user)
DETECT_CLASSES_COCO = {0: 'helmet', 1: 'no_helmet', 3: 'motorcycle'}
# If the model has exactly 3 classes (0: rider, 1: helmet, 2: no_helmet)
DETECT_CLASSES_3 = {0: 'rider', 1: 'helmet', 2: 'no_helmet'}

INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.65
NMS_THRESHOLD = 0.45

# ==================== Global/Queue Setup ====================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# multiprocessing.Queue works across process boundaries
upload_queue = multiprocessing.Queue()

# Track Best Frames: {track_id: {'proof': None, 'crops': [(score, crop_img)]}}
# 'crops' will hold up to 3 tuples, sorted by score.
violator_tracking = {}

# ==================== Storage Management ====================

def check_and_clear_storage(directory, max_mb=1000):
    """If directory exceeds max_mb, deletes oldest files until it's under the limit."""
    total_size = 0
    files_with_times = []
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            size = stat.st_size
            total_size += size
            files_with_times.append((filepath, stat.st_mtime, size))
            
    total_mb = total_size / (1024 * 1024.0)
    
    if total_mb > max_mb:
        print(f"[Storage] Limit reached ({total_mb:.1f}MB > {max_mb}MB). Cleaning up...")
        # Sort by oldest first
        files_with_times.sort(key=lambda x: x[1])
        
        freed = 0
        for filepath, _, size in files_with_times:
            try:
                os.remove(filepath)
                freed += size
                total_mb -= (size / (1024 * 1024.0))
                if total_mb <= max_mb * 0.9: # Drop to 90% capacity to avoid continuous wiping
                    break
            except Exception as e:
                print(f"[Storage] Failed to delete {filepath}: {e}")
                
        print(f"[Storage] Cleaned up {freed/(1024*1024.0):.1f}MB.")

# ==================== Upload Worker (Separate Process) ====================

def upload_worker(q, api_url, api_key, camera_id, location):
    """
    Runs in a completely separate OS process.
    Sends all images for a track as a single bundled multipart POST request.
    Bundle naming: track_{id}_bundle containing track_{id}_proof, track_{id}_crop0, etc.
    """
    import requests as req  # Import inside process for safety
    print("[Uploader] Separate upload process started (PID: %d)." % os.getpid())
    while True:
        try:
            item = q.get()
            if item is None:  # Poison pill to exit
                print("[Uploader] Received shutdown signal. Exiting.")
                break
                
            track_id = item['track_id']
            proof_path = item['proof']
            crop_paths = item['crops']
            bundle_id = f"track_{track_id}_bundle"

            # Build multipart file list: [ ("files", (filename, file_obj, mime)) , ... ]
            file_handles = []
            multipart_files = []

            # Add proof image
            if proof_path and os.path.exists(proof_path):
                fh = open(proof_path, 'rb')
                file_handles.append(fh)
                multipart_files.append(("files", (f"track_{track_id}_proof.jpg", fh, "image/jpeg")))

            # Add crop images
            for c_idx, c_path in enumerate(crop_paths):
                if os.path.exists(c_path):
                    fh = open(c_path, 'rb')
                    file_handles.append(fh)
                    multipart_files.append(("files", (f"track_{track_id}_crop{c_idx}.jpg", fh, "image/jpeg")))

            if not multipart_files:
                print(f"[Uploader] No files found for {bundle_id}. Skipping.")
                continue

            print(f"[Uploader] Sending {bundle_id} ({len(multipart_files)} files)...")

            headers = {"x-api-key": api_key}
            data = {
                "camera_id": camera_id,
                "location": location,
                "bundle_id": bundle_id,
            }

            try:
                response = req.post(api_url, headers=headers, files=multipart_files, data=data, timeout=30)

                if response.status_code == 200:
                    json_data = response.json()
                    plate = json_data.get('best_plate', {}).get('plate', 'N/A')
                    print(f"[Uploader] {bundle_id} received -> Plate: {plate}")

                    # Delete all uploaded files from disk
                    for path in [proof_path] + crop_paths:
                        if path and os.path.exists(path):
                            os.remove(path)
                else:
                    print(f"[Uploader] {bundle_id} HTTP Error {response.status_code}: {response.text}")

            except Exception as e:
                print(f"[Uploader] {bundle_id} request failed: {e}")
            finally:
                # Always close file handles
                for fh in file_handles:
                    fh.close()

        except Exception as e:
            print(f"[Uploader] Main loop error: {e}")

# ==================== Image Processing Utilities ====================

def bgr2nv12(bgr_img: np.ndarray) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    area = h * w
    yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
    nv12 = np.zeros_like(yuv420p)
    nv12[:area] = y
    nv12[area:] = uv_packed
    return nv12

def letterbox(img: np.ndarray, new_shape=(640, 640), color=(127, 127, 127)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess(img: np.ndarray):
    img_resized, ratio, (dw, dh) = letterbox(img, new_shape=INPUT_SIZE)
    nv12 = bgr2nv12(img_resized)
    return nv12, ratio, (dw, dh)

# ==================== Inference & Post-processing ====================

def postprocess(outputs, orig_shape, ratio, pad):
    data = np.squeeze(outputs[0])
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    if data.shape[0] < data.shape[1]:
        data = data.T

    total_cls = data.shape[1] - 4
    if total_cls <= 0: return np.array([]), np.array([]), np.array([])
    cx, cy, bw, bh = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    mapping = DETECT_CLASSES_3 if total_cls <= 3 else DETECT_CLASSES_COCO
    wanted = np.array([c for c in mapping.keys() if c < total_cls], dtype=int)
    
    cls_probs = data[:, 4 + wanted]               
    max_scores = cls_probs.max(axis=1)
    best_idx = cls_probs.argmax(axis=1)
    class_ids = wanted[best_idx]                   

    keep = max_scores >= CONF_THRESHOLD
    if not np.any(keep): return np.array([]), np.array([]), np.array([])

    cx, cy, bw, bh = cx[keep], cy[keep], bw[keep], bh[keep]
    max_scores, class_ids = max_scores[keep], class_ids[keep]

    x1 = cx - bw * 0.5
    y1 = cy - bh * 0.5
    x2 = cx + bw * 0.5
    y2 = cy + bh * 0.5
    boxes_lb = np.stack([x1, y1, x2, y2], axis=1)

    dw, dh = pad
    boxes = boxes_lb.copy()
    boxes[:, [0, 2]] = np.clip((boxes_lb[:, [0, 2]] - dw) / ratio, 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip((boxes_lb[:, [1, 3]] - dh) / ratio, 0, orig_shape[0])

    final_boxes, final_scores, final_ids = [], [], []
    for cid in wanted:
        mask = class_ids == cid
        if not np.any(mask): continue
        cb = boxes[mask]
        cs = max_scores[mask]
        xywh = np.stack([cb[:, 0], cb[:, 1], cb[:, 2] - cb[:, 0], cb[:, 3] - cb[:, 1]], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(xywh, cs.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
        if len(idxs) > 0:
            idxs = np.array(idxs).flatten()
            final_boxes.append(cb[idxs])
            final_scores.append(cs[idxs])
            final_ids.append(np.full(len(idxs), cid, dtype=int))

    if not final_boxes: return np.array([]), np.array([]), np.array([])
    return np.concatenate(final_boxes), np.concatenate(final_scores), np.concatenate(final_ids)

def detect_frame(model, frame):
    orig_shape = frame.shape[:2]
    nv12, ratio, pad = preprocess(frame)
    # The shims `Infer.infer([nv12])` is expected based on bpu_infer_lib_x5.py
    outputs = model.infer([nv12])
    boxes, scores, class_ids = postprocess(outputs, orig_shape, ratio, pad)
    return boxes, scores, class_ids

# ==================== Processing Loops ====================

def get_frame_generator(input_path):
    if os.path.isdir(input_path):
        IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        files = sorted(files)
        print(f"Directory detected. Found {len(files)} images.")
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                yield img
    else:
        # Check if it's a live network stream
        is_live = str(input_path).startswith(('rtsp://', 'http://', 'https://'))
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Failed to open video/stream: {input_path}")
            return
            
        if is_live:
            # Force 1-frame buffer for live streams to avoid latency queueing
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Live Stream detected. Processing {input_path} with 1-frame buffer.")
        else:
            print(f"Video file detected. Processing {input_path}")
            
        while cap.isOpened():
            # For live streams, we might want to grab/retrieve to flush the buffer
            # but setting BUFFERSIZE to 1 usually suffices for OpenCV modern versions.
            ret, frame = cap.read()
            if not ret: 
                if is_live:
                    print("Stream connection lost. Reconnecting in 5s...")
                    time.sleep(5)
                    cap.release()
                    cap = cv2.VideoCapture(input_path)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue
                else:
                    break
            yield frame
            
        cap.release()

def run_pipeline(model, input_path):
    frame_gen = get_frame_generator(input_path)

    tracker = SimpleIOUTracker(iou_threshold=0.3, max_missed=30, min_hits=2)
    uploaded_ids = set()

    # Model specific constants
    # Assuming 'helmet_best2.bin' is a 3-class model where 0=Rider, 1=Helmet, 2=NoHelmet
    # Or COCO-based where 3=Motorcycle, 0=Helmet, 1=NoHelmet. 
    # We will track the enclosing bounding box. If 3-class: Rider (0). If COCO: Motorcycle (3).
    # Since riders are directly inside the bounding box in 3-class, we use 0.
    RIDER_CLASS = 0
    NO_HELMET_CLASS = 2

    # If you are using the older 80-class mode where motorcycle=3, change to:
    # RIDER_CLASS = 3
    # NO_HELMET_CLASS = 1

    fps = 30.0
    frame_count = 0

    print("Pipeline started.")

    try:
        for frame in frame_gen:
            clean_frame = frame.copy()

            boxes, scores, class_ids = detect_frame(model, frame)
            
            # Extract riders/motorcycles for tracking
            tracker_input = []
            for i in range(len(boxes)):
                if class_ids[i] == RIDER_CLASS:
                    tracker_input.append((int(class_ids[i]), float(scores[i]), 
                                          boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]))
            
            # Standard simple IO tracker update
            active_tracks = tracker.update(tracker_input)
            
            # Map out no_helmets for intersecting
            no_helmets = []
            for i in range(len(boxes)):
                if class_ids[i] == NO_HELMET_CLASS:
                    no_helmets.append(boxes[i])

            # Check inside active tracks
            for track_id, (cls_id, conf, x1, y1, x2, y2) in active_tracks.items():
                if track_id in uploaded_ids:
                    continue  # Already exhausted/uploaded

                has_violation = False
                for nh_box in no_helmets:
                    # Stricter intersection: no-helmet center should be in top 50% of Rider/Motorcycle
                    nh_cx = (nh_box[0] + nh_box[2]) / 2.0
                    nh_cy = (nh_box[1] + nh_box[3]) / 2.0
                    
                    v_cy = (y1 + y2) / 2.0
                    
                    if (x1 < nh_cx < x2) and (nh_cy < v_cy):
                        has_violation = True
                        break

                if has_violation:
                    h, w = frame.shape[:2]
                    px1 = max(0, int(x1))
                    py1 = max(0, int(y1))
                    px2 = min(w, int(x2))
                    py2 = min(h, int(y2))
                    
                    rider_crop = clean_frame[py1:py2, px1:px2]
                    if rider_crop.size == 0: continue
                    
                    # Score by Area: A larger box means the vehicle is closer to the camera.
                    # This physically guarantees more pixels on the number plate, 
                    # avoiding the motion-blur traps of Laplacian variance.
                    score = (px2 - px1) * (py2 - py1)
                    
                    if track_id not in violator_tracking:
                        violator_tracking[track_id] = {'proof_img': None, 'crops': []}
                        
                    best_crops = violator_tracking[track_id]['crops']
                    
                    # Store 1 proof frame immediately on first violation
                    if violator_tracking[track_id]['proof_img'] is None:
                        violator_tracking[track_id]['proof_img'] = clean_frame.copy()

                    # Array of 3 crops: [ (score, crop), (score, crop), (score, crop) ]
                    # Insert if list < 3 or if score is strictly better than the worst in list
                    if len(best_crops) < 3:
                        best_crops.append((score, rider_crop.copy()))
                        best_crops.sort(key=lambda x: x[0], reverse=True) # Highest score first
                    elif score > best_crops[-1][0]:
                        best_crops[-1] = (score, rider_crop.copy())
                        best_crops.sort(key=lambda x: x[0], reverse=True)

            # Check lost tracks and send to upload queue
            lost_tracks = tracker.get_lost_tracks(min_missed=30)
            for tid in lost_tracks:
                if tid in violator_tracking and tid not in uploaded_ids:
                    uploaded_ids.add(tid)
                    
                    # We are ready to serialize and queue
                    check_and_clear_storage(OUTPUT_DIR, MAX_STORAGE_MB)
                    
                    # Save proof
                    proof_path = os.path.join(OUTPUT_DIR, f"track_{tid}_proof.jpg")
                    cv2.imwrite(proof_path, violator_tracking[tid]['proof_img'])
                    
                    crop_paths = []
                    for c_idx, (c_score, c_img) in enumerate(violator_tracking[tid]['crops']):
                        c_path = os.path.join(OUTPUT_DIR, f"track_{tid}_crop{c_idx}.jpg")
                        # Write raw un-enhanced crop at 100% JPEG Quality
                        cv2.imwrite(c_path, c_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        crop_paths.append(c_path)
                    
                    print(f"Submitting ID {tid} to uploader (Crops: {len(crop_paths)})...")
                    upload_queue.put({
                        'track_id': tid,
                        'proof': proof_path,
                        'crops': crop_paths
                    })
                    
                    # Clean RAM
                    del violator_tracking[tid]

            frame_count += 1
            if frame_count % 100 == 0: print(f"Processed {frame_count} frames...")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Flush the remainders
        for tid, data in violator_tracking.items():
            if tid not in uploaded_ids:
                check_and_clear_storage(OUTPUT_DIR, MAX_STORAGE_MB)
                
                proof_path = os.path.join(OUTPUT_DIR, f"track_{tid}_proof.jpg")
                cv2.imwrite(proof_path, data['proof_img'])
                
                crop_paths = []
                for c_idx, (c_score, c_img) in enumerate(data['crops']):
                    c_path = os.path.join(OUTPUT_DIR, f"track_{tid}_crop{c_idx}.jpg")
                    # Write raw un-enhanced crop at 100% JPEG Quality
                    cv2.imwrite(c_path, c_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    crop_paths.append(c_path)
                    
                upload_queue.put({
                    'track_id': tid,
                    'proof': proof_path,
                    'crops': crop_paths
                })

        # Wait for uploads to finish
        print("Waiting for upload process to finish...")
        upload_queue.put(None)
        worker_process.join(timeout=60)
        if worker_process.is_alive():
            print("[Warning] Upload process did not exit cleanly. Terminating.")
            worker_process.terminate()
        print("Done.")

# ==================== Main Entry ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("ALPR Pipeline")
    parser.add_argument("--model", type=str, default="helmet_best2.bin")
    parser.add_argument("--input", type=str, required=True, help="Video file, image dir, or stream")
    args = parser.parse_args()

    # Start the upload worker as a SEPARATE OS PROCESS
    worker_process = multiprocessing.Process(
        target=upload_worker,
        args=(upload_queue, API_URL, API_KEY, CAMERA_ID, LOCATION),
        daemon=True
    )
    worker_process.start()
    print(f"[Main] Detection process PID: {os.getpid()}")
    print(f"[Main] Upload process PID:    {worker_process.pid}")

    print(f"Loading Model: {args.model}")
    model = Infer(args.model)
    print("Model loaded. Starting pipeline...")
    
    run_pipeline(model, args.input)
