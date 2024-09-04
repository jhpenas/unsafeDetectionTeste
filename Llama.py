import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import requests
from io import BytesIO

def analisarImagemLLAMA(image, prompt="Briefly describe the image"):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["mm_projector", "vision_model"],
    )

    model_id = "qresearch/llama-3-vision-alpha-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
    ).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )


    return(
        tokenizer.decode(
            model.answer_question(image, "Briefly describe the image", tokenizer),
            skip_special_tokens=True,
        )
    )

url = "https://huggingface.co/qresearch/llama-3-vision-alpha-hf/resolve/main/assets/demo-2.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
#image = Image.open('PPE/test/images/15h_img_263_jpg.rf.322a8288620372c00832997189849870.jpg')
analisarImagemLLAMA(image)