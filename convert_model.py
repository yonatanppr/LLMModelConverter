import coremltools as ct
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np

def convert_model():
    """
    Downloads a distilled summarization model and converts it for Core ML
    using a FIXED input shape to guarantee conversion.
    """
    model_name = "sshleifer/distilbart-cnn-6-6"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torchscript=True)
    model.eval()

    print("Tracing model...")
    example_text = "An example document to be summarized by the model."
    example_input = tokenizer(example_text, return_tensors="pt")
    traced_model = torch.jit.trace(model, (example_input.input_ids, example_input.attention_mask))

    # --- THE MAIN CHANGE IS HERE ---
    # Instead of a dynamic dimension, we define a fixed sequence length.
    # 512 is a good balance between context length and performance.
    fixed_seq_len = 512

    input_ids = ct.TensorType(name="input_ids", shape=(1, fixed_seq_len), dtype=np.int32)
    attention_mask = ct.TensorType(name="attention_mask", shape=(1, fixed_seq_len), dtype=np.int32)

    print("Converting model to Core ML...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[input_ids, attention_mask],
        convert_to="mlprogram", # Using the modern mlprogram format
        compute_units=ct.ComputeUnit.ALL
    )

    coreml_model.short_description = "Summarizes English text using a distilled BART model."
    coreml_model.input_description["input_ids"] = f"Tokenized input text (padded/truncated to {fixed_seq_len})."
    coreml_model.input_description["attention_mask"] = "Mask to ignore padding tokens."

    output_path = "ClerkSummarizer.mlpackage"
    coreml_model.save(output_path)
    print(f"Successfully converted and saved model to: {output_path}")

if __name__ == "__main__":
    convert_model()