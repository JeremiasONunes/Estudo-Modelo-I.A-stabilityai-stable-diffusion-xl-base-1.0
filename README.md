
# Stable Diffusion XL (SDXL) 1.0 - Base Model

## Visão Geral do Projeto

O **Stable Diffusion XL (SDXL) 1.0** é um modelo de geração de imagens baseado em difusão, desenvolvido pela **Stability AI**. 
Ele permite a criação e modificação de imagens a partir de descrições textuais. O modelo combina técnicas avançadas para 
produzir resultados visualmente impressionantes, sendo amplamente utilizado em pesquisa, arte e design.

### Principais Características

- **Modelo:** Baseado em difusão latente.
- **Desenvolvedor:** Stability AI.
- **Codificadores de Texto:** OpenCLIP-ViT/G e CLIP-ViT/L.
- **Licença:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).
- **Código Fonte:** Disponível no [GitHub](https://github.com/Stability-AI/generative-models).
- **Uso Recomandado:** Pesquisa em modelos generativos, ferramentas criativas e educacionais.


### Difusão Latente

A difusão latente é uma técnica avançada utilizada em modelos generativos, especialmente em inteligência artificial para criação de imagens e outros conteúdos. Essa abordagem combina conceitos de aprendizado profundo e processos estocásticos, como o método de difusão, para gerar dados em alta qualidade.

## Funcionamento do Modelo

O SDXL opera de duas maneiras principais:

1. **Modelo Base (Standalone):**
   - Gera latentes iniciais de saída a partir de descrições textuais.
   - Permite o uso direto para tarefas que não exigem refinamento visual extremo.

2. **Pipeline Completa:** 
   - **Etapa 1:** Geração de latentes com o modelo base.
   - **Etapa 2:** Refinamento dos latentes usando um modelo especializado (Refiner), otimizando os detalhes visuais.

Além disso, técnicas como SDEdit ("img2img") permitem ajustes e modificações em imagens existentes, 
ampliando as possibilidades criativas.

### SDEdit ("img2img")

O SDEdit (ou "img2img") é uma técnica baseada em difusão que permite editar imagens existentes de maneira criativa ou refinada, mantendo aspectos visuais desejados enquanto aplica alterações baseadas em um texto ou outras entradas. O nome "img2img" vem de "imagem para imagem", indicando que o processo começa com uma imagem fornecida pelo usuário e a transforma com base em instruções adicionais.

## Requisitos de Instalação

Para começar a usar o SDXL, instale as dependências necessárias:

```bash
pip install diffusers transformers accelerate safetensors invisible_watermark --upgrade
```

Se necessário, configure o modelo para **offloading de CPU** ou use otimizações de desempenho do PyTorch para reduzir o tempo de inferência.

## Exemplos de Uso

### Não é necessario fazer um clone do projeto

Não é necessário clonar o modelo diretamente para sua máquina, pois o DiffusionPipeline da biblioteca diffusers permite carregar os modelos diretamente da plataforma Hugging Face. Quando você usa o método from_pretrained, ele faz o download automático dos arquivos necessários e os armazena localmente (normalmente em um cache, como ~/.cache/huggingface).

### Como funciona o carregamento com from_pretrained:

Primeiro uso:

Quando você executa o comando:
```python
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
```
O script verifica se o modelo já foi baixado. Se não estiver disponível localmente, ele faz o download do repositório público do Hugging Face.
Todos os arquivos necessários, como pesos do modelo, configuradores e vocabulários, são armazenados em cache.
Uso subsequente:

O pipeline usa o cache local e não realiza um novo download, tornando o processo mais rápido.

### Quando clonar o repositório?
Clonar o repositório completo pode ser útil em casos específicos:

- **Modificação do Código Fonte:**
Se você deseja modificar ou entender como o pipeline é implementado internamente.
Exemplo: Ajustar detalhes no modelo ou integrar funcionalidades personalizadas.
Treinamento/Finetuning:
Se você planeja treinar o modelo novamente ou ajustá-lo com dados personalizados, o clone do repositório pode fornecer scripts e exemplos úteis.
Ambientes sem Conexão com a Internet:
Se o ambiente onde o modelo será usado não tiver acesso à internet, você pode clonar o repositório e carregar os pesos manualmente.
Clonando o repositório, se necessário:
Para clonar o repositório oficial do modelo:

```
git clone https://github.com/Stability-AI/generative-models.git
```
Depois de clonar, você pode configurar o pipeline para carregar o modelo dos arquivos locais:

```
pipe = DiffusionPipeline.from_pretrained("/caminho/para/o/modelo")
```
### Recomendação
Na maioria dos casos, usar from_pretrained diretamente do Hugging Face é suficiente e mais conveniente. A menos que precise de algo muito específico, você não precisa clonar o modelo na máquina.

### Apenas o Modelo Base:

```python
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

```

### Pipeline Completa (Base + Refinamento):

```python
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
```
### utilizando uma API do huggingface.

```python
import requests
import io
import os  # Importa o módulo 'os' para manipular caminhos de arquivos
from PIL import Image

from huggingface_hub import InferenceClient
client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token="Bearer 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")

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"}

# Função para fazer a consulta à API e obter os bytes da imagem
def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Verifica se a resposta foi bem-sucedida
        if response.status_code == 200:
            return response.content
        else:
            print(f"Erro na solicitação: {response.status_code}")
            print(response.text)  # Exibe a mensagem de erro retornada pela API
            return None
    except Exception as e:
        print(f"Erro ao conectar com a API: {e}")
        return None

# Função principal para gerar e salvar a imagem
def gerar_imagem():
    # Solicita a descrição da imagem ao usuário
    descricao = input("Digite a descrição da imagem que você quer gerar: ")

    # Gera a imagem com base na descrição fornecida
    image_bytes = query({
        "inputs": descricao,
    })

    # Verifica se a resposta da API foi válida
    if image_bytes is None:
        print("Falha ao gerar a imagem.")
        return

    try:
        # Converte os bytes para uma imagem
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"Erro ao abrir a imagem: {e}")
        return

    # Caminho para salvar a imagem na pasta atual no seu Windows
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Define o nome do arquivo da imagem
    nome_arquivo = descricao[:10].replace(" ", "_") + "_imagem.png"  # Evita nomes de arquivos muito longos
    caminho_imagem = os.path.join(current_directory, nome_arquivo)

    # Salva a imagem no caminho especificado
    image.save(caminho_imagem)

    print(f"A imagem foi salva em: {caminho_imagem}")

# Executa a função para gerar e salvar a imagem com uma descrição de exemplo
# Executa a função para gerar e salvar a imagem
while True:
    print("Digite 1 para gerar uma nova imagem")
    print("Digite 0 para sair")
    opcao = int(input())
    if opcao == 1:
        gerar_imagem()
    elif opcao == 0:
        break
    else:
        print("Opção inválida. Por favor, digite 1 ou 0.")


```

## Limitações do Modelo

Embora seja poderoso, o modelo possui algumas limitações:

- **Fotorealismo:** Não atinge perfeição visual.
- **Texto em Imagens:** Não gera texto legível.
- **Tarefas Complexas:** Desafios com composição complexa, como "um cubo vermelho sobre uma esfera azul".
- **Faces e Pessoas:** Representações humanas podem ser imprecisas.
- **Codificação:** Perdas podem ocorrer durante a auto-codificação.

## Áreas de Aplicação

### Uso Direto

- **Criação Artística:** Geração de obras de arte e designs únicos.
- **Ferramentas Criativas:** Aplicações em educação e desenvolvimento criativo.
- **Pesquisa:** Exploração dos limites e potenciais de modelos generativos.

### Uso Não Recomendado

- Representação factual de pessoas ou eventos, devido à natureza generativa do modelo.

## Contribuições e Contato

O SDXL é um modelo de código aberto. Para contribuir, acesse o [repositório no GitHub](https://github.com/Stability-AI/generative-models). Dúvidas ou sugestões podem ser enviadas diretamente à equipe de desenvolvimento.

---

Este README fornece uma visão abrangente do SDXL, detalhando seu uso e características. Explore as possibilidades criativas e aplique em seus projetos!


![Imagem Gerada](An_astronaut_riding_green_horse.jpg)

![Imagem Gerada](super_mari_imagem.png)