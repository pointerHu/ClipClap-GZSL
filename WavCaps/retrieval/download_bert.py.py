from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# 简单测试一下
test_text = "Hello world"
encoded = tokenizer(test_text, return_tensors="pt")
output = model(**encoded)
print(f"Model output shape: {output.last_hidden_state.shape}")
print("模型加载成功!")