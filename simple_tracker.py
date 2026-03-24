"""
Lightweight IoU-based multi-object tracker for RDK X5.
Replaces Ultralytics' ByteTrack with a simple, dependency-free tracker.

Each tracked object gets a unique integer ID that persists across frames.
"""

import numpy as np
from collections import OrderedDict


def iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def iou_matrix(boxes_a, boxes_b):
    """Compute IoU matrix between two sets of boxes."""
    n = len(boxes_a)
    m = len(boxes_b)
    matrix = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            matrix[i, j] = iou(boxes_a[i], boxes_b[j])
    return matrix


class TrackedObject:
    """Represents a single tracked object."""

    def __init__(self, track_id, box, class_id, score):
        self.track_id = track_id
        self.box = box          # [x1, y1, x2, y2]
        self.class_id = class_id
        self.score = score
        self.age = 0            # frames since creation
        self.hits = 1           # total frames matched
        self.missed = 0         # consecutive frames without match

    def update(self, box, class_id, score):
        self.box = box
        self.class_id = class_id
        self.score = score
        self.hits += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1


class SimpleIOUTracker:
    """
    Simple IoU-based tracker suitable for edge deployment.
    
    Matches detections to existing tracks using IoU.
    Uses greedy assignment (highest IoU first).
    
    Args:
        iou_threshold: Minimum IoU to consider a match (default 0.3)
        max_missed: Maximum consecutive frames a track can be missed before removal (default 30)
        min_hits: Minimum hits before a track is considered confirmed (default 1)
    """

    def __init__(self, iou_threshold=0.3, max_missed=30, min_hits=1):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.tracks = OrderedDict()  # track_id → TrackedObject
        self.next_id = 1

    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: list of (class_id, score, x1, y1, x2, y2)
        
        Returns:
            active_tracks: dict of {track_id: (class_id, score, x1, y1, x2, y2)}
                           Only confirmed (hits >= min_hits) and not-missed tracks.
        """
        if not detections:
            # Mark all tracks as missed
            to_remove = []
            for tid, track in self.tracks.items():
                track.mark_missed()
                if track.missed > self.max_missed:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return {}

        det_boxes = [d[2:6] for d in detections]
        det_classes = [d[0] for d in detections]
        det_scores = [d[1] for d in detections]

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].box for tid in track_ids]

        matched_tracks = set()
        matched_dets = set()

        if track_boxes and det_boxes:
            # Compute IoU matrix
            cost_matrix = iou_matrix(track_boxes, det_boxes)

            # Greedy matching: pick highest IoU pairs first
            while True:
                if cost_matrix.size == 0:
                    break
                max_iou = cost_matrix.max()
                if max_iou < self.iou_threshold:
                    break

                ti, di = np.unravel_index(cost_matrix.argmax(), cost_matrix.shape)
                tid = track_ids[ti]
                self.tracks[tid].update(det_boxes[di], det_classes[di], det_scores[di])
                matched_tracks.add(tid)
                matched_dets.add(di)

                # Invalidate this row and column
                cost_matrix[ti, :] = 0
                cost_matrix[:, di] = 0

        # Handle unmatched tracks
        to_remove = []
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].mark_missed()
                if self.tracks[tid].missed > self.max_missed:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        # Create new tracks for unmatched detections
        for di in range(len(detections)):
            if di not in matched_dets:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = TrackedObject(
                    new_id, det_boxes[di], det_classes[di], det_scores[di]
                )

        # Increment age for all tracks
        for track in self.tracks.values():
            track.age += 1

        # Return confirmed, active tracks
        active = {}
        for tid, track in self.tracks.items():
            if track.hits >= self.min_hits and track.missed == 0:
                active[tid] = (track.class_id, track.score, *track.box)
        return active

    def get_all_track_ids(self):
        """Return set of all current track IDs."""
        return set(self.tracks.keys())

    def get_lost_tracks(self, min_missed=30):
        """Return track IDs that haven't been seen for min_missed frames."""
        lost = set()
        for tid, track in self.tracks.items():
            if track.missed >= min_missed:
                lost.add(tid)
        return lost
