from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from mmdet.datasets import CocoDataset
import numpy as np
from scipy.optimize import linear_sum_assignment

from color_tagging.utils.data_utils import (MASK_PALETTE, CLASSES)
from color_tagging.utils.metrics_utils import pairwise_ious, parse_annotations


@DATASETS.register_module()
class ColorFashionDataset(CocoDataset):
    CLASSES = CLASSES
    PALETTE = MASK_PALETTE[1:]

    def _parse_ann_info(self, img_info, ann_info):
        annotations = super()._parse_ann_info(img_info, ann_info)

        color_anns = []

        for ann in ann_info:
            color_anns.append(ann["color"])

        annotations["colors"] = color_anns

        return annotations

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.
        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.
        Returns:
            dict[str, float]: COCO style evaluation and color metrics.
        """

        metrics = metric if isinstance(metric, list) else [metric]

        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        bbox_results, color_results = list(map(list, zip(*results)))

        result_files, tmp_dir = self.format_results(bbox_results,
                                                    jsonfile_prefix)

        eval_results = self.evaluate_det_segm(bbox_results, result_files,
                                              coco_gt, metrics, logger,
                                              classwise, proposal_nums,
                                              iou_thrs, metric_items)
        if color_results[0]["det_colors"].size > 0:
            color_accuracy = self.evaluate_color(color_results, coco_gt,
                                                 logger)
            eval_results.update(color_accuracy)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def evaluate_color(self, results, coco_gt, logger):
        """
        Method that evaluates color metrics during training.

        Args:
            model_results (list[dict]): Color resuts obtained from model.
            coco_gt (COCO): COCO API object with ground truth annotation.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: Color metrics (only accuracy is implemented).

        """
        filenames = []
        for image in coco_gt.dataset["images"]:
            filenames.append(image["file_name"])

        gt_anns = parse_annotations(coco_gt.dataset)

        pred_col = []
        gt_col = []
        for filename, result in zip(filenames, results):
            gt_boxes, gt_colors = list(map(list, zip(*gt_anns[filename])))
            gt_boxes = np.array(gt_boxes)
            ious = pairwise_ious(gt_boxes, result["det_bboxes"][:, :4])
            matched_gt, matched_pred = linear_sum_assignment(ious)

            color_res = result["det_colors"].tolist()
            pred_col.extend([color_res[x] for x in matched_pred.tolist()])
            gt_col.extend([gt_colors[x] for x in matched_gt.tolist()])

        pred_col = np.array(pred_col)
        gt_col = np.array(gt_col)
        accuracy = (pred_col == gt_col).mean()

        msg = f"Color accuracy: {accuracy}."
        print_log(msg, logger)

        return dict(color_acc=accuracy)
