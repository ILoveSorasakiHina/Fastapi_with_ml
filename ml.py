from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image



token_path = Path("token.txt")
token = token_path.read_text().strip()


pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    revision="fp16", 
    torch_dtype=torch.float32,
    user_auth_token = token,
    )



prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt).images[0]

def obtain_image(
        prompt: str,
        *,
        seed: int | None = None,
        num_inference: int = 50,
        guidance_scale: float = 7.5,
) -> Image:
    generator = None if seed is None else torch.Generator("cpu")
    print(f"Using device:{pipe.device}")
    image: Image = pipe(
        prompt,
        #圖像品質，值越低品質越高，但可能不符合提示詞要求。
        guidance_scale = guidance_scale,
        num_inference = num_inference,
        generator = generator,
    ).images[0]

    return image


# image = obtain_image(prompt,num_inference = 5, seed = 1024)


