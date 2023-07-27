import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

peft_model_id = "/home/prasanta/falcon7b/falcon7b/results/checkpoint-2000/" # This should be the patha of the directory where model has been fine tuned
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    load_in_8bit=True,
    device_map = {"": 0},
    trust_remote_code=True,
    load_in_4bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

def generate(
        instruction,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs
):
    prompt = instruction + "\n### Response:\n"
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].lstrip("\n")

instruction = "Write a python code to scrape data from website."

print(generate(instruction))