"""测试共用指标口径：在同一对 RGB 图像上分别评测亮度和三通道误差。"""

DISTORTION_NAMES = ('psnr_y', 'ssim_y', 'psnr_rgb', 'ssim_rgb')
METRIC_NAMES = (*DISTORTION_NAMES, 'lpips')
METRIC_PROTOCOL = 'dual-channel-v1: PSNR-Y/SSIM-Y (YCbCr), PSNR-RGB/SSIM-RGB; LPIPS float RGB; FID from PNG'


def build_quality_metrics(device, include_lpips=True, channels=('y', 'rgb')):
    """显式指定两个通道口径，避免库默认值变化；RGB SSIM 对三通道结果求均值。"""
    import pyiqa
    metrics = {
        f'{name}_{channel}': pyiqa.create_metric(
            name, test_y_channel=(channel == 'y'), color_space='ycbcr', device=device
        ).eval()
        for channel in channels for name in ('psnr', 'ssim')
    }
    if include_lpips:
        metrics['lpips'] = pyiqa.create_metric('lpips', device=device).eval()
    return metrics


def compute_quality_metrics(prediction, target, metrics):
    """生成器输出 [B,3,H,W]、[-1,1]；两种口径共享 FP32 转换、裁剪和归一化。"""
    prediction = prediction.float().clamp(-1, 1) * 0.5 + 0.5
    target = target.float().clamp(-1, 1) * 0.5 + 0.5
    return {name: float(metric(prediction, target).item()) for name, metric in metrics.items()}


def format_quality_metrics(values):
    """日志明确标注 Y/RGB，防止把不同口径的数值误作可比结果。"""
    labels = {'psnr_y':'PSNR-Y', 'ssim_y':'SSIM-Y', 'psnr_rgb':'PSNR-RGB', 'ssim_rgb':'SSIM-RGB', 'lpips':'LPIPS'}
    return ' | '.join(f'{labels[name]}={values[name]:.4f}' for name in METRIC_NAMES if name in values)
