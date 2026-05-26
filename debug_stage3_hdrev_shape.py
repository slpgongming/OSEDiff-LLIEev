import argparse

import torch

from dataloaders.sde_event_dataset import SDEEventDataset
from osediff_hdrev import OSEDiffHDRevGen, tensor_stats
from train_osediff_lle_hdrev import parse_train_timestep_range


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--hdrev_pretrained_root", type=str, default="checkpoints/HDRev")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--event_norm", action="store_true", default=True)
    parser.add_argument("--no_event_norm", dest="event_norm", action="store_false")
    parser.add_argument("--fixed_prompt", type=str, default="high quality, bright, detailed photo")
    parser.add_argument("--neg_prompt", type=str, default="underexposed, dark, low-light, noisy")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--train_timestep_range", default="999", type=parse_train_timestep_range)
    parser.add_argument("--lambda_latent_align", type=float, default=0.05)
    return parser.parse_args()


def main(args):
    # Reuse the training dataset path names expected by SDEEventDataset.
    args.train_data_root = args.data_root
    args.test_data_root = args.data_root

    dataset = SDEEventDataset(split="train", args=args)
    sample = dataset[0]
    batch = {
        "conditioning_pixel_values": sample["conditioning_pixel_values"].unsqueeze(0),
        "hdrev_pixel_values": sample["hdrev_pixel_values"].unsqueeze(0),
        "event_voxel": sample["event_voxel"].unsqueeze(0),
        "output_pixel_values": sample["output_pixel_values"].unsqueeze(0),
        "prompt": [args.fixed_prompt],
        "neg_prompt": [args.neg_prompt],
    }

    model = OSEDiffHDRevGen(args)
    model.eval()

    with torch.no_grad():
        _, hdrev_features = model.hdrev_encoder(
            batch["hdrev_pixel_values"].to(model.device),
            batch["event_voxel"].to(model.device),
        )
        start_latent = model.event_latent_projector(hdrev_features)
        target_latent = model.encode_target_latent(batch["output_pixel_values"])
        output, x_denoised, *_, model_pred, sampled_timesteps = model(batch, args=args, return_target_latent=False)

    print("===== Stage 3 HDRev shape smoke test =====")
    print(f"low_rgb [-1,1]:      {tuple(batch['conditioning_pixel_values'].shape)}")
    print(f"hdrev_rgb [0,1]:     {tuple(batch['hdrev_pixel_values'].shape)}")
    print(f"event_voxel:         {tuple(batch['event_voxel'].shape)}")
    for idx, feat in enumerate(hdrev_features):
        print(f"hdrev_feature[{idx}]: {tuple(feat.shape)}")
    print(f"start_latent:        {tuple(start_latent.shape)} stats={tensor_stats(start_latent)}")
    print(f"target_latent:       {tuple(target_latent.shape)} stats={tensor_stats(target_latent)}")
    print(f"model_pred:          {tuple(model_pred.shape)} stats={tensor_stats(model_pred)}")
    print(f"x_denoised:          {tuple(x_denoised.shape)} stats={tensor_stats(x_denoised)}")
    print(f"output:              {tuple(output.shape)} stats={tensor_stats(output)}")
    print(f"sampled_timestep:    {sampled_timesteps.detach().cpu().tolist()}")
    print(f"event_sparsity:      {(batch['event_voxel'] != 0).float().mean().item():.8f}")


if __name__ == "__main__":
    main(parse_args())
