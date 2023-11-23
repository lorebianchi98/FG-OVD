FROM nvcr.io/nvidia/pytorch:22.09-py3
RUN pip install transformers==4.28.1
RUN pip install sentencepiece
RUN pip install accelerate
RUN mkdir -p .cache/huggingface/datasets
RUN mkdir .cache/huggingface/hub
RUN export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"
RUN export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/hub"
RUN pip install debugpy
RUN pip install bitsandbytes