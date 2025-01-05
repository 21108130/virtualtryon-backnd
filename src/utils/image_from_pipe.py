import os
from tqdm import tqdm
import torch

import torchvision.transforms as T
from diffusers import DiffusionPipeline
from torch.utils.data import DataLoader
import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.image_composition import compose_img, compose_img_dresscode


@torch.inference_mode()
def generate_images_from_mgd_pipe(
    test_order: bool,
    pipe: DiffusionPipeline,
    test_dataloader: DataLoader,
    save_name: str,
    dataset: str,
    output_dir: str,
    guidance_scale: float = 7.5,
    guidance_scale_pose: float = 7.5,
    guidance_scale_sketch: float = 7.5,
    sketch_cond_rate: float = 1.0,
    start_cond_rate: float = 0.0,
    no_pose: bool = False,
    disentagle: bool = False,
    seed: int = 1234,
) -> None:
    """
    Generates images from the given test dataloader and saves them to the output directory.
    """

    assert save_name != "", "save_name must be specified"
    assert output_dir != "", "output_dir must be specified"

    path = os.path.join(output_dir, f"{save_name}_{test_order}", "images")
    os.makedirs(path, exist_ok=True)

    generator = torch.Generator("cuda").manual_seed(seed)

    for batch in tqdm(test_dataloader):
        # Debugging: Print batch information
        print(f"Processing batch {test_order}")
        print(f"Saving images to: {path}")
        print(f"Batch keys: {batch.keys()}")  # Check available keys in batch

        model_img = batch["image"]
        mask_img = batch["inpaint_mask"].type(torch.float32)
        prompts = batch["original_captions"]  # List of prompts
        pose_map = batch["pose_map"]
        sketch = batch["im_sketch"]
        ext = ".jpg"

        # Debugging: Validate `pipe`
        print(f"Type of `pipe`: {type(pipe)}")
        print(f"Is `pipe` callable? {callable(pipe)}")
        assert callable(pipe), "`pipe` must be callable. Check MGDPipe implementation."

        if disentagle:
            generated_images = pipe(
                prompt=prompts,
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                sketch=sketch,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                sketch_cond_rate=sketch_cond_rate,
                guidance_scale_pose=guidance_scale_pose,
                guidance_scale_sketch=guidance_scale_sketch,
                start_cond_rate=start_cond_rate,
                no_pose=no_pose,
            ).images
        else:
            generated_images = pipe(
                prompt=prompts,
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                sketch=sketch,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                sketch_cond_rate=sketch_cond_rate,
                start_cond_rate=start_cond_rate,
                no_pose=no_pose,
            ).images

        for i, generated_image in enumerate(generated_images):
            model_i = model_img[i] * 0.5 + 0.5
            if dataset == "vitonhd":
                final_img = compose_img(model_i, generated_image, batch["im_parse"][i])
            else:  # dataset == Dresscode
                face = batch["stitch_label"][i].to(model_img.device)
                face = T.functional.resize(
                    face,
                    size=(512, 384),
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                final_img = compose_img_dresscode(
                    gt_img=model_i,
                    fake_img=T.functional.to_tensor(generated_image).to(model_img.device),
                    im_head=face,
                )

            # Save the final image
            final_img = T.functional.to_pil_image(final_img)
            save_path = os.path.join(path, batch["im_name"][i].replace(".jpg", ext))
            final_img.save(save_path)
            print(f"Saved image to {save_path}")

