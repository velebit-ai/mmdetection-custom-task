from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadAnnotations


@PIPELINES.register_module()
class LoadColorAnnotations(LoadAnnotations):
    """Load multiple types of annotations.
    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_colors (bool): Whether to parse and load color annotations.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
            Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        denorm_bbox (bool): Whether to convert bbox from relative value to
            absolute value. Only used in OpenImage Dataset.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_colors=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):
        super().__init__(with_bbox=with_bbox, with_label=with_label,
                         with_mask=with_mask, with_seg=with_seg,
                         poly2mask=poly2mask, denorm_bbox=denorm_bbox,
                         file_client_args=file_client_args)
        self.with_colors = with_colors

    def _load_colors(self, results):
        """Private function to load color annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_colors'] = ann_info['colors'].copy()

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)

        if self.with_colors:
            results = self._load_colors(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_colors={self.with_colors}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str
