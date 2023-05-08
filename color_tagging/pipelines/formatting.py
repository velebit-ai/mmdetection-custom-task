from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import to_tensor, DefaultFormatBundle


@PIPELINES.register_module()
class ModifiedDefaultFormatBundle(DefaultFormatBundle):

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        results = super().__call__(results)

        keys = ['gt_colors']

        for key in keys:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))

        return results
