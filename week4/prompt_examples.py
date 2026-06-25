"""
Week 4 - Prompt Engineering Examples

Simple examples demonstrating how prompt wording
changes LLM outputs.
"""

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompts = [

    "Explain machine learning in one sentence.",

    "List three applications of computer vision in manufacturing.",

    "Describe why multimodal AI is useful in industrial quality assurance."

]

for prompt in prompts:

    print("="*60)

    print("Prompt:")

    print(prompt)

    output = generator(

        prompt,

        max_length=70,

        do_sample=True,

        temperature=0.7

    )

    print("\nGenerated Response:\n")

    print(output[0]["generated_text"])

    print()
