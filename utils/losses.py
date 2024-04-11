import torch

mse2psnr = lambda x:-10. * torch.log(x) / (torch.log(torch.tensor([10.0])).item())

def dist_loss(t, w, conf=None):
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2
    # ut = t
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    loss = loss_inter + loss_intra

    if conf is not None:
        loss = loss * conf

    return loss.mean()