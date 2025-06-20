import coremltools as ct
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import numpy as np


def convert_model():
    """
    Downloads and correctly converts the 't5-small' model to predict the next token.
    This version correctly traces the model before conversion.
    """
    model_name = "t5-small"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       torchscript=True)  # Tracing requires torchscript=True
    model.eval()

    # 1. Prepare example inputs for both the encoder and decoder.
    task_prefix = "summarize: "
    example_text = "An example document."

    # Encoder input
    input_ids = tokenizer(task_prefix + example_text, return_tensors="pt").input_ids

    # Decoder input (the start token)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.long)

    # 2. Trace the model to get a TorchScript object. This is the required format.
    print("Tracing model...")
    traced_model = torch.jit.trace(model, (input_ids, None,
                                           decoder_input_ids))  # Pass None for attention_mask during this trace

    # 3. Define the input specifications for the Core ML model
    fixed_seq_len = 512
    decoder_len = ct.RangeDim(lower_bound=1, upper_bound=fixed_seq_len, default=1)

    input_ids_spec = ct.TensorType(name="input_ids", shape=(1, fixed_seq_len), dtype=np.int32)
    decoder_input_ids_spec = ct.TensorType(name="decoder_input_ids", shape=(1, decoder_len), dtype=np.int32)

    print("Converting model to Core ML...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[input_ids_spec, decoder_input_ids_spec],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL
    )

    coreml_model.short_description = "Generates next-token logits using the T5-small model."

    output_path = "ClerkT5NextToken.mlpackage"
    coreml_model.save(output_path)
    print(f"Successfully converted and saved model to: {output_path}")


if __name__ == "__main__":
    convert_model()