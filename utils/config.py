import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    parser.add_argument("--depth_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--sd_model_path", type=str, default=None)
    parser.add_argument('--controlnet_model_path', type=str, default=None)
    parser.add_argument("--semantic_model_path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--revision", type=str, default=None, required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument('--controlnet_conditioning_scale', type=float, default=1.)

    parser.add_argument('--finetune_style', type=str2bool, default=False)
    parser.add_argument('--finetune', type=str2bool, default=False)
    parser.add_argument("--finetune_sky", type=str2bool, default=True)
    parser.add_argument("--finetune_depth", type=str2bool, default=True)
    parser.add_argument("--chunk_size", type=int, default=16384)
    parser.add_argument("--finetune_num_rays", type=int, default=16384)
    parser.add_argument("--num_train_step", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--random_angle", type=float, default=15.)
    parser.add_argument("--pre_generation", type=str2bool, default=True)
    parser.add_argument("--use_clip_loss", type=str2bool, default=True)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--checkpoint_freq", type=int, default=10000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--render_video', type=str2bool, default=False)
    parser.add_argument('--layout_path', type=str, default=None)
    parser.add_argument('--use_8bit_adam', type=str2bool, default=False)
    parser.add_argument('--evaluate', type=str2bool, default=False)
    parser.add_argument('--text_prompt', type=str, default='a residential scene')
    parser.add_argument('--train_scale_factor', type=float, default=2.)
    parser.add_argument('--num_nerf_samples', type=int, default=32)
    parser.add_argument('--grid_scale', type=float, default=400)
    parser.add_argument('--obj_scale', type=float, default=1.)
    parser.add_argument('--use_viewdirs', type=str2bool, default=False)
    parser.add_argument('--use_objviewdirs', type=str2bool, default=False)
    parser.add_argument('--obj_log2_T', type=int, default=14)
    parser.add_argument('--sky_rep', type=str, default='grid')
    parser.add_argument('--empty_bkgd', type=str2bool, default=False)

    return parser.parse_args()