from collections import defaultdict

import numpy as np


def pairwise_ious(bboxes1, bboxes2):
    """
    Calculates pairwise IoUs between two sets of bounding boxes.
    Args:
        bboxes1: np.array of shape (N, 4) in format (x1, y1, x2, y2).
        bboxes2: np.array of shape (M, 4) with the same semantics.
    Return:
        np.array of shape (N, M) containing pairwise IoUs for the given bboxes.
    """
    areas1, areas2 = map(lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]),
                         (bboxes1, bboxes2))

    lt = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])
    rb = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])

    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = areas1[:, None] + areas2 - inter

    return inter / union


def parse_annotations(annotation_dict):
    """
    A function that parses dict with annotations that belong to a certain
    dataset split.
    Args:
        annotations_dict (dict): Annotations dict.
    Returns:
        dict(list): Dict where key is filename and are bounding boxes with
            corresponding colors.
    """
    img_anns = defaultdict(list)

    for annotation in annotation_dict["annotations"]:
        image_id = annotation["image_id"]
        x0, y0, w, h = annotation["bbox"]
        bbox = [x0, y0, x0 + w, y0 + h]
        color = annotation["color"]
        img_anns[image_id].append((bbox, color))

    return img_anns
