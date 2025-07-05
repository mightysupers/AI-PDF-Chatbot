from llama_cpp import Llama

# ✅ Path to your downloaded GGUF model file
model_path = "models/llama-2-7b.Q8_0.gguf"

# ✅ Load the model (this may take a few seconds)
llm = Llama(model_path=model_path, n_ctx=2048)

# ✅ Test prompt
response = llm("Q: What is artificial intelligence?\nA:", max_tokens=100)

# ✅ Print result
print("\nResponse:")
print(response["choices"][0]["text"].strip())
