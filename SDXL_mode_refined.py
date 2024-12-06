from PIL import Image
import os
from diffusers import DiffusionPipeline
import torch

# Carrega o pipeline do modelo base do Stable Diffusion XL.
# Este modelo é responsável por gerar os latentes iniciais (representação comprimida da imagem).
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # Nome do modelo base.
    torch_dtype=torch.float16,                  # Define o tipo de dado como float16 para economizar memória.
    variant="fp16",                             # Variante para otimizações de desempenho.
    use_safetensors=True                        # Usa o formato "safetensors" para segurança e eficiência.
)

# Carrega o pipeline do modelo refinador do Stable Diffusion XL.
# Este modelo refina os latentes gerados pelo modelo base, melhorando detalhes e qualidade.
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",  # Nome do modelo refinador.
    text_encoder_2=base.text_encoder_2,            # Reutiliza o segundo codificador de texto do modelo base.
    vae=base.vae,                                  # Reutiliza o autoencoder variacional (VAE) do modelo base.
    torch_dtype=torch.float16,                     # Define o tipo de dado como float16 para economizar memória.
    use_safetensors=True,                          # Usa o formato "safetensors" para segurança e eficiência.
    variant="fp16"                                 # Variante para otimizações de desempenho.
)

# Move o modelo base para o dispositivo CUDA (GPU), necessário para processamento acelerado.
base.to("cuda")

# Move o modelo refinador para o dispositivo CUDA (GPU).
refiner.to("cuda")

# Define o prompt que descreve a imagem que queremos gerar.
# Neste caso, estamos solicitando "Um leão majestoso pulando de uma grande pedra à noite".
prompt = "A majestic lion jumping from a big stone at night"

# Gera os latentes iniciais usando o modelo base.
# - prompt: descrição textual da imagem desejada.
# - num_inference_steps: número de passos de inferência para gerar a imagem (maior número = mais detalhes).
# - denoising_end: porcentagem de ruído a ser removida nesta etapa (neste caso, 80% do processo é feito pelo modelo base).
latents = base(prompt=prompt, num_inference_steps=40, denoising_end=0.8).images

# Refina os latentes gerados pelo modelo base usando o refinador.
# - prompt: mesma descrição textual, para manter a consistência com o modelo base.
# - num_inference_steps: número de passos de inferência (mesmo valor usado no modelo base).
# - denoising_start: inicia o refinamento a partir de 80% do processo, completando os últimos 20%.
# - image: latentes gerados pelo modelo base, que serão processados para criar a imagem final.
final_image = refiner(prompt=prompt, num_inference_steps=40, denoising_start=0.8, image=latents).images[0]

# Salva a imagem gerada no mesmo diretório que o código.
output_path = os.path.join(os.path.dirname(__file__), "generated_image.png")
final_image.save(output_path)