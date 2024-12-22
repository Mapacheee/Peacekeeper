from transformers import AutoTokenizer, AutoModel
import torch

model_name = "AI-Safeguard/Ivy-VL-llava"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt_text = "Describe the image"
tokens = tokenizer(prompt_text,
                   return_tensors="pt",
                   padding=True,
                   truncation=True)
prompt_input_ids = tokens["input_ids"]
prompt_image = torch.zeros((1, 3, 224, 224))

torch.onnx.export(
    model,
    (prompt_image, prompt_input_ids),
    "Peacekeeper.onnx",
    input_names=["image", "input_ids"],
    output_names=["output"],
    dynamic_axes={"image": {0: "batch_size"}, "input_ids": {0: "batch_size"}},
    opset_version=13
)

print("carga lista")