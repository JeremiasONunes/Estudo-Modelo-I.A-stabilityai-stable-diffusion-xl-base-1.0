from huggingface_hub import InferenceClient
client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token="Bearer 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")

# output is a PIL.Image object
image = client.text_to_image("Astronaut riding a horse")
image.save("/c:/Users/Jerem/OneDrive/Área de Trabalho/SDXL/generated_image.png")