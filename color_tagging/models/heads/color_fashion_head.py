from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import anchor_inside_flags, multi_apply, reduce_mean, unmap
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.models.dense_heads.yolof_head import YOLOFHead, levels_to_images
from mmdet.models.builder import HEADS, build_loss
import torch
import torch.nn as nn


@HEADS.register_module()
class ColorFashionHead(YOLOFHead):

    def __init__(self,
                 num_classes,
                 num_colors,
                 in_channels,
                 num_cls_convs=2,
                 num_reg_convs=4,
                 num_col_convs=2,
                 loss_col=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    reduction='mean',
                    loss_weight=1.0),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 **kwargs):
        """
        Color Fashion Head

        Args:
            num_classes (int): The number of object classes (w/o background)
            num_colors (int): The number rof color classes.
            in_channels (List[int]): The number of input channels per scale.
            cls_num_convs (int): The number of convolutions of cls branch.
                Default 2.
            reg_num_convs (int): The number of convolutions of reg branch.
                Default 4.
            num_col_convs (int): The numbeor of convolutions of color branch.
            norm_cfg (dict): Dictionary to construct and config norm layer.
        """
        super().__init__(num_classes, in_channels, num_cls_convs=num_cls_convs,
                         num_reg_convs=num_reg_convs, norm_cfg=norm_cfg,
                         **kwargs)
        self.num_colors = num_colors
        self.loss_col = build_loss(loss_col)
        self.num_col_convs = num_col_convs
        self.use_sigmoid_col = loss_col.get('use_sigmoid', False)
        # If we use loss that requires softmax (cross entropy loss),
        # then we consider the background as an additional class.
        if self.use_sigmoid_col:
            self.col_out_channels = num_colors
        else:
            self.col_out_channels = num_colors + 1

        self._init_col_layers()

    def _init_col_layers(self):
        col_subnet = []
        for i in range(self.num_col_convs):
            col_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        self.col_subnet = nn.Sequential(*col_subnet)
        self.col_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * self.col_out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward_single(self, feature):
        normalized_cls_score, bbox_reg = super().forward_single(feature)

        col_score = self.col_pred(self.col_subnet(feature))

        return normalized_cls_score, bbox_reg, col_score

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None,
                      gt_colors=None, gt_bboxes_ignore=None, proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_colors (Tensor): Ground truth color for each bbox,
                shape (num_gts, num_colors)
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (lforwaist[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_colors, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'col_scores'))
    def loss(self, cls_scores, bbox_preds, col_scores, gt_bboxes, gt_labels,
             gt_colors, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (batch, num_anchors * num_classes, h, w)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (batch, num_anchors * 4, h, w)
            col_scores (list[Tensor]): Color scores for each scale level
                Has shape (batch, num_anchors * num_colors, h, w)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_colors (list[Tensor]): color indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == 1
        assert self.prior_generator.num_levels == 1

        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        # The output level is always 1
        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]

        cls_scores_list = levels_to_images(cls_scores)
        bbox_preds_list = levels_to_images(bbox_preds)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            gt_colors_list=gt_colors,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (batch_labels, batch_label_weights, batch_colors, batch_color_weights,
         num_total_pos, num_total_neg, batch_bbox_weights,
         batch_pos_predicted_boxes, batch_target_boxes) = cls_reg_targets

        flatten_labels = batch_labels.reshape(-1)
        batch_label_weights = batch_label_weights.reshape(-1)
        cls_score = cls_scores[0].permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)

        flatten_colors = batch_colors.reshape(-1)
        batch_color_weights = batch_color_weights.reshape(-1)
        col_score = col_scores[0].permute(0, 2, 3,
                                          1).reshape(-1, self.col_out_channels)

        num_total_samples = (num_total_pos +
                             num_total_neg) if self.sampling else num_total_pos
        num_total_samples = reduce_mean(
            cls_score.new_tensor(num_total_samples)).clamp_(1.0).item()

        # classification loss
        loss_cls = self.loss_cls(
            cls_score,
            flatten_labels,
            batch_label_weights,
            avg_factor=num_total_samples)

        # color classification loss)
        loss_col = self.loss_col(
            col_score,
            flatten_colors,
            batch_color_weights,
            avg_factor=num_total_samples)

        # regression loss
        if batch_pos_predicted_boxes.shape[0] == 0:
            # no pos sample
            loss_bbox = batch_pos_predicted_boxes.sum() * 0
        else:
            loss_bbox = self.loss_bbox(
                batch_pos_predicted_boxes,
                batch_target_boxes,
                batch_bbox_weights.float(),
                avg_factor=num_total_samples)

        return dict(loss_cls=loss_cls, loss_col=loss_col, loss_bbox=loss_bbox)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    gt_colors_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.
        Args:
            cls_scores_list (list[Tensor])： Classification scores of
                each image. each is a 4D-tensor, the shape is
                (h * w, num_anchors * num_classes).
            bbox_preds_list (list[Tensor])： Bbox preds of each image.
                each is a 4D-tensor, the shape is (h * w, num_anchors * 4).
            anchor_list (list[Tensor]): Anchors of each image. Each element of
                is a tensor of shape (h * w * num_anchors, 4).
            valid_flag_list (list[Tensor]): Valid flags of each image. Each
               element of is a tensor of shape (h * w * num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            gt_labels_list (list[Tensor]): Ground truth colors of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - batch_labels (Tensor): Label of all images. Each element \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - batch_label_weights (Tensor): Label weights of all images \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - batch_colors (Tensor): Color of all images. Each element \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - batch_color_weights (Tensor): Color weights of all images \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        if gt_colors_list is None:
            gt_colors_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            gt_colors_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_colors, all_color_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        batch_labels = torch.stack(all_labels, 0)
        batch_label_weights = torch.stack(all_label_weights, 0)

        batch_colors = torch.stack(all_colors, 0)
        batch_color_weights = torch.stack(all_color_weights, 0)

        res = (batch_labels, batch_label_weights, batch_colors,
               batch_color_weights, num_total_pos, num_total_neg)
        for i, rests in enumerate(rest_results):  # user-added return values
            rest_results[i] = torch.cat(rests, 0)

        return res + tuple(rest_results)

    def _get_targets_single(self,
                            bbox_preds,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            gt_colors,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.
        Args:
            bbox_preds (Tensor): Bbox prediction of the image, which
                shape is (h * w ,4)
            flat_anchors (Tensor): Anchors of the image, which shape is
                (h * w * num_anchors ,4)
            valid_flags (Tensor): Valid flags of the image, which shape is
                (h * w * num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_colors (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
        Returns:
            tuple:
                labels (Tensor): Labels of image, which shape is
                    (h * w * num_anchors, ).
                label_weights (Tensor): Label weights of image, which shape is
                    (h * w * num_anchors, ).
                colors (Tensor): Colors of image, which shape is
                    (h * w * num_anchors, ).
                color_weights (Tensor): Color weights of image, which shape is
                    (h * w * num_anchors, ).
                pos_inds (Tensor): Pos index of image.
                neg_inds (Tensor): Neg index of image.
                sampling_result (obj:`SamplingResult`): Sampling result.
                pos_bbox_weights (Tensor): The Weight of using to calculate
                    the bbox branch loss, which shape is (num, ).
                pos_predicted_boxes (Tensor): boxes predicted value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
                pos_target_boxes (Tensor): boxes target value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 8
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        bbox_preds = bbox_preds.reshape(-1, 4)
        bbox_preds = bbox_preds[inside_flags, :]

        # decoded bbox
        decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        assign_result = self.assigner.assign(
            decoder_bbox_preds, anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)

        pos_bbox_weights = assign_result.get_extra_property('pos_idx')
        pos_predicted_boxes = assign_result.get_extra_property(
            'pos_predicted_boxes')
        pos_target_boxes = assign_result.get_extra_property('target_boxes')

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        colors = anchors.new_full((num_valid_anchors, ), self.num_colors,
                                  dtype=torch.long)
        color_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

            if gt_colors is None:
                colors[pos_inds] = 0
            else:
                colors[pos_inds] = gt_colors[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                color_weights[pos_inds] = 1.0
            else:
                color_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)

            colors = unmap(
                colors, num_total_anchors, inside_flags,
                fill=self.num_colors)
            color_weights = unmap(color_weights, num_total_anchors,
                                  inside_flags)

        return (labels, label_weights, colors, color_weights, pos_inds,
                neg_inds, sampling_result, pos_bbox_weights,
                pos_predicted_boxes, pos_target_boxes)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'col_scores'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   col_scores,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.
        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            col_scores (list[Tensor]): Classification color scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_colors, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor, Tensor]]: Each item in result_list is
                2-tuple. The first item is an (n, 5) tensor, where the first 4
                columns are bounding box positions (tl_x, tl_y, br_x, br_y) and
                the 5-th column is a score between 0 and 1. The second item is
                a (n,) tensor where each item is the predicted class label of
                the corresponding box. The third item is a (n,) tensor where
                each item is the predicted color of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            col_score_list = select_single_mlvl(col_scores, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              col_score_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           col_score_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            col_score_list (list[Tensor]): Box color scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple[Tensor]: Results of detected bboxes, labels and colors. If
                with_nms is False and mlvl_score_factor is None,
                return mlvl_bboxes and mlvl_scores, else return mlvl_bboxes,
                mlvl_scores and mlvl_score_factor. Usually with_nms is False
                is used for aug test. If with_nms is True, then return the
                following format:
                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
                - det_colors (Tensor): Predicted colors of the corresponding \
                    box with shape [num_bboxes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_colors = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None

        zipped = zip(cls_score_list, bbox_pred_list, col_score_list,
                     score_factor_list, mlvl_priors)
        for level_idx, (cls_score, bbox_pred, col_score,
                        score_factor, priors) in enumerate(zipped):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            col_score = col_score.permute(1, 2,
                                          0).reshape(-1, self.col_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            if self.use_sigmoid_col:
                scores_col = col_score.sigmoid()
            else:
                scores_col = col_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            labels_col = scores_col[keep_idxs].argmax(dim=1)

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_colors.append(labels_col)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       mlvl_colors, img_meta['scale_factor'],
                                       cfg, rescale, with_nms,
                                       mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           mlvl_colors,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.
        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.
        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.
        Returns:
            tuple[Tensor]: Results of detected bboxes, labels and colors. If
                with_nms is False and mlvl_score_factor is None,
                return mlvl_bboxes and mlvl_scores, else return mlvl_bboxes,
                mlvl_scores and mlvl_score_factor. Usually with_nms is False
                is used for aug test. If with_nms is True, then return the
                following format:
                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
                - det_colors (Tensor): Predicted colors of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        mlvl_colors = torch.cat(mlvl_colors)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels, mlvl_colors

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            det_colors = mlvl_colors[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels, det_colors
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels
