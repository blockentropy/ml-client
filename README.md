<img src="1on_whitex4.png" alt="Entropy ML Clients" width="400"/>
 Entropy ML Clients.  OpenAI compatible clients for distributed and decentralized compute

## Latest News

- **2023/12:** Initial release of ML endpoints

## About

This repository contains custom endpoints for use in distributed and decentralized compute. Most of these endpoints are compatible with OpenAI, including the LLM client, image client, and embedding clients. There are additional custom endpoints that support state-of-the-art research features, allowing the Entropy API to achieve competitive performance results with GPT-4 models.

### OpenAI Compatible Endpoints

- `llm_client.py`: LLM chat compatible endpoint. Supported models include Yi, Starling, Mixtral, Mistral, Phind, Llama, and more.
- `image_client.py`: Image compatible endpoint. Supported models include SD1.5 diffusion models, and SDXL diffusion models. Also added support for [IP adapters](https://github.com/tencent-ailab/IP-Adapter) for images/edits endpoint.
- `embedding_client.py`: Vector embedding compatible endpoint. Supported models include the BGE embedding models.

_Note:_ Our router is compatible with vLLM endpoints as well. See [vLLM project](https://github.com/vllm-project/vllm). vLLM has support for multiple clients. Differences include support for HuggingFace models, GPTQ quantization, and basic authentication with the Entropy router. They also handle the prompt template of different models within the custom code.

### Custom Research Endpoints

- `rerank_client.py`: This rerank endpoint takes in an input and a list of several output strings, then returns a rank of the best outputs. Based on research by [LLM-Blender](https://github.com/yuchenlin/LLM-Blender). The API endpoint is `v1/rank`.
- `compress_client.py`: This compression endpoint takes in an input and compresses it, maintaining the structure and meaning of the original input. Based on research by [LLMLingua](https://github.com/microsoft/LLMLingua). The API endpoints are `v1/compress` and `v1/compresslong`.

### Conda Environments

- `llm_environment.yml`: Conda environment needed for the LLMs.
- `guardrails_environment.yml`: Conda environment needed to set up guardrails for those interested in trustworthy, safe, and controllable LLM conversations. Based on research by [NVIDIA NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails).

### Config

- `config.ini.sample`: Configuration file that points to model directories, port, upload destination (for image generation). Rename to `config.ini`.

### Installation

The endpoints were installed on Ubuntu 22 and 23 Linux-based machines. The port can be overridden from the `config.ini` file by using the cmd line `--port` option. Recommended method of installing miniconda is here, [miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

```bash
conda env create -f llm_environment.yml
conda activate bellm

python3 llm_client.py
```

Flash attention will need to be installed after the fact. On Ubuntu 23, do the following

```bash
sudo apt update
sudo apt install nvidia-cudnn nvidia-cuda-toolkit

pip3 install flash-attn --no-build-isolation
```

### Troubleshooting

If you have trouble finding CUDA_HOME, or the cuda toolkit when trying to compile flash attention on Ubuntu 22, you can try the following.

```bash
wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/cuda.gpg
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt update
sudo apt install cuda-toolkit-11-8
pip install flash-attn --no-build-isolation
```

Ubuntu 23 does not require the adding of any repos for the toolkit.
