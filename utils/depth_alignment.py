import torch
EPS = 1e-8
def scale_shift_linear(rendered_depth, predicted_depth, mask, fuse=True):
    """
    Optimize a scale and shift parameter in the least squares sense, such that rendered_depth and predicted_depth match.
    Formally, solves the following objective:

    min     || (d * a + b) - d_hat ||
    a, b

    where d = 1 / predicted_depth, d_hat = 1 / rendered_depth

    :param rendered_depth: torch.Tensor (H, W)
    :param predicted_depth:  torch.Tensor (H, W)
    :param mask: torch.Tensor (H, W) - 1: valid points of rendered_depth, 0: invalid points of rendered_depth (ignore)
    :param fuse: whether to fuse shifted/scaled predicted_depth with the rendered_depth

    :return: scale/shift corrected depth
    """
    if mask.sum() == 0:
        return predicted_depth

    rendered_disparity = 1 / (rendered_depth[mask].unsqueeze(-1) + EPS)
    predicted_disparity = 1 / (predicted_depth[mask].unsqueeze(-1) + EPS)

    X = torch.cat([predicted_disparity, torch.ones_like(predicted_disparity)], dim=1)
    XTX_inv = (X.T @ X).inverse()
    XTY = X.T @ rendered_disparity
    # print('XTX_inv:', XTX_inv)
    # print('XTY:', XTY)
    AB = XTX_inv @ XTY
    # print('AB:', AB)

    fixed_disparity = (1 / (predicted_depth+EPS)) * AB[0] + AB[1]
    fixed_depth = 1 / (fixed_disparity + EPS)

    if fuse:
        fused_depth = torch.where(mask, rendered_depth, fixed_depth)
        return fused_depth
    else:
        return fixed_depth