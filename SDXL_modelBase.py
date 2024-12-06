# Importa a classe DiffusionPipeline da biblioteca diffusers, usada para carregar e executar o modelo.
from diffusers import DiffusionPipeline

# Importa o PyTorch, necessário para manipulação de tensores e uso de GPU.
import torch

# Importa a biblioteca os para manipulação de caminhos de diretórios.
import os

# Importa a biblioteca PIL para salvar a imagem.
from PIL import Image

# Carrega o pipeline do modelo Stable Diffusion XL Base.
# Esse pipeline inclui todas as etapas necessárias para gerar imagens a partir de prompts textuais.
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # Nome do modelo base na plataforma Hugging Face.
    torch_dtype=torch.float16,                  # Define o tipo de dado como float16 para reduzir o uso de memória.
    use_safetensors=True,                       # Usa o formato "safetensors" para carregamento eficiente e seguro.
    variant="fp16"                              # Variante otimizada para GPUs com suporte a precisões mistas.
)

# Move o pipeline para a GPU (dispositivo CUDA) para acelerar a inferência.
pipe.to("cuda")

# Define o prompt que descreve a imagem a ser gerada.
# Neste caso, o prompt é "Um astronauta montando um cavalo verde".
prompt = "An astronaut riding a green horse"

# Gera a imagem a partir do prompt fornecido.
# O pipeline interpreta o texto e cria uma imagem correspondente.
image = pipe(prompt=prompt).images[0]

# Obtém o caminho do diretório atual.
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define o caminho completo para salvar a imagem.
image_path = os.path.join(current_directory, "generated_image.png")

# Salva a imagem no caminho especificado.
image.save(image_path)