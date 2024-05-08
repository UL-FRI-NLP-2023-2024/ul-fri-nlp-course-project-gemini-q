from torch import bfloat16
import transformers


if __name__ == "__main__":
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()

    test_prompt = "Tvoja mama je"
    result = generate_text(test_prompt)
    print(result[0]['generated_text'])
