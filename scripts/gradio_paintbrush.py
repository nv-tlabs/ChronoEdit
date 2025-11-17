import time
import gradio as gr
import torch
import numpy as np
import tempfile
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers import UniPCMultistepScheduler
from chronoedit_diffusers.pipeline_chronoedit import ChronoEditPipeline
from chronoedit_diffusers.transformer_chronoedit import ChronoEditTransformer3DModel
from transformers import CLIPVisionModel


start = time.time()

model_id = "nvidia/ChronoEdit-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(
	model_id,
	subfolder="image_encoder",
	torch_dtype=torch.float32
)
print("✓ Loaded image encoder")

vae = AutoencoderKLWan.from_pretrained(
	model_id,
	subfolder="vae",
	torch_dtype=torch.bfloat16
)
print("✓ Loaded VAE")

transformer = ChronoEditTransformer3DModel.from_pretrained(
	model_id,
	subfolder="transformer",
	torch_dtype=torch.bfloat16
)
print("✓ Loaded transformer")

pipe = ChronoEditPipeline.from_pretrained(
	model_id,
	image_encoder=image_encoder,
	transformer=transformer,
	vae=vae,
	torch_dtype=torch.bfloat16
)
print("✓ Created pipeline")

pipe.load_lora_weights("nvidia/ChronoEdit-14B-Diffusers-Paint-Brush-Lora", weight_name="paintbrush_lora_diffusers.safetensors", adapter_name="paintbrush")
pipe.load_lora_weights("nvidia/ChronoEdit-14B-Diffusers", weight_name="lora/chronoedit_distill_lora.safetensors", adapter_name="distill")
pipe.fuse_lora(adapter_names=["paintbrush", "distill"], lora_scale=1.0)

pipe.scheduler = UniPCMultistepScheduler.from_config(
	pipe.scheduler.config,
	flow_shift=2.0
)

pipe.to("cuda")
# pipe.enable_model_cpu_offload()
end = time.time()
print(f"Model loaded in {end - start:.2f}s.")


def calculate_dimensions(image,  mod_value):
    """
    Calculate output dimensions based on resolution settings.
    
    Args:
        image: PIL Image
        mod_value: Modulo value for dimension alignment
        
    Returns:
        Tuple of (width, height)
    """
    
    # Get max area from preset or override 
    target_area = 720 * 1280
    
    # Calculate dimensions maintaining aspect ratio
    aspect_ratio = image.height / image.width
    calculated_height = round(np.sqrt(target_area * aspect_ratio)) // mod_value * mod_value
    calculated_width = round(np.sqrt(target_area / aspect_ratio)) // mod_value * mod_value
    
    return calculated_width, calculated_height


# @spaces.GPU
def run_inference(
	image,
	prompt: str,
	num_inference_steps: int = 8,
	guidance_scale: float = 1.0,
):
	final_prompt = f"{prompt}"

	if isinstance(image, dict):
		image = image["composite"]
	image = load_image(image)
	mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
	width, height = calculate_dimensions(
		image,
		mod_value
	)
	print(f"Output dimensions: {width}x{height}")
	image = image.resize((width, height))
	num_frames = 5

	start = time.time()
	output = pipe(
		image=image,
		prompt=final_prompt,
		height=height,
		width=width,
		num_frames=num_frames,
		num_inference_steps=num_inference_steps,
		guidance_scale=guidance_scale,
		enable_temporal_reasoning=False,
		num_temporal_reasoning_steps=0,
	).frames[0]
	end = time.time()
	print(f"Generated video in {end - start:.2f}s")

	image_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
	output_path_image = image_tmp.name
	image_tmp.close()
	Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save(output_path_image)

	log_text = (
		f"Prompt: {prompt}\n"
		f"Final prompt: {final_prompt}\n"
		f"Guidance: {guidance_scale}, Steps: {num_inference_steps}\n"
	)
	print(log_text)

	return output_path_image


def build_ui() -> gr.Blocks:
	with gr.Blocks(title="ChronoEdit", theme=gr.themes.Soft()) as demo:

		gr.Markdown("""
		# 🎨 ChronoEdit Paint Brush Demo
		This demo is built on ChronoEdit-14B with a paintbrush LoRA. You can make edits simply by drawing a quick sketch on the input image.  
		[[Project Page]](https://research.nvidia.com/labs/toronto-ai/chronoedit/) | 
		[[Code]](https://github.com/nv-tlabs/ChronoEdit) |
		[[Technical Report]](https://arxiv.org/abs/2510.04290) | 
		[[Model]](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers)
		""")
		
		with gr.Row():
			with gr.Column(scale=1):
				image = gr.ImageEditor(type="pil", brush=gr.Brush(default_color="black", default_size=8), label="Input Image")
				with gr.Column(scale=1):
					gr.Markdown("""
					_Trigger prompt: "Turn the pencil sketch in the image into an actual object that is consistent with the image’s content. The user wants to change the sketch to {}."_
					""")
					prompt = gr.Textbox(label="Prompt", lines=4, value="Turn the pencil sketch in the image into an actual object that is consistent with the image’s content. The user wants to change the sketch to ")
					run_btn = gr.Button("Start Generation", variant="primary")
				
					with gr.Accordion("Advanced options", open=False):
						num_inference_steps = gr.Slider(minimum=4, maximum=75, step=1, value=8, label="Num Inference Steps")
						guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.5, value=1.0, label="Guidance Scale")
			
			with gr.Column(scale=1):
				output_image = gr.Image(label="Generated Image")

		def _on_run(image, prompt, num_inference_steps, guidance_scale):
			image_out_path = run_inference(
				image=image,
				prompt=prompt,
				num_inference_steps=num_inference_steps,
				guidance_scale=guidance_scale,
			)
			return image_out_path

		run_btn.click(
			_on_run,
			inputs=[image, prompt, num_inference_steps, guidance_scale],
			outputs=[output_image]
		)

	return demo


if __name__ == "__main__":
	demo = build_ui()
	demo.launch(share=True)