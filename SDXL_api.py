import requests
import io
import os  # Importa o módulo 'os' para manipular caminhos de arquivos
from PIL import Image

from huggingface_hub import InferenceClient
client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token="Bearer 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")

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

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
