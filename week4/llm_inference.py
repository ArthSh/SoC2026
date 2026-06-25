"""
Week 4 - Introduction to Large Language Models

This script demonstrates:
1. Loading a pretrained language model
2. Tokenizing text
3. Generating text using Hugging Face Transformers
"""

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

prompt = "Artificial Intelligence is transforming industrial quality assurance by"

result = generator(
    prompt,
    max_length=80,
    num_return_sequences=1,
    temperature=0.8
)

print("Prompt:")
print(prompt)

print("\nGenerated Text:\n")
print(result[0]["generated_text"])
