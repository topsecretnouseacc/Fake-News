import argparse
from transformers import pipeline

def generate_fake_news(category: str, max_length: int = 200) -> str:
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"Write a short fake news article about {category}:"
    outputs = generator(prompt, max_length=max_length, num_return_sequences=1)
    return outputs[0]["generated_text"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        type=str,
        default="technology",
        help="Fake news örneği üretilecek kategori"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Üretilecek metnin maksimum token uzunluğu"
    )
    args = parser.parse_args()

    article = generate_fake_news(args.category, args.max_length)
    print("\n--- Fake News Sample ---\n")
    print(article)
