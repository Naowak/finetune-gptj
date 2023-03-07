from transformers import GPTJForCausalLM, AutoTokenizer
import argparse

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

# Load model
print("Loading the model...")
model = GPTJForCausalLM.from_pretrained(args.model).half().to("cuda")
tokenizer = AutoTokenizer.from_pretrained(args.model)
print(f"{args.model} loaded !")


while True:

    # Get input from user
    question = str(input("\n\nQuestion to ask to the model:\n"))
    if len(question) == 0:
        continue
    
    # Construct text var
    text = (
        "La réponse doit répondre à la question. La réponse doit être précise, argumentée, et doit suivre un cheminement logique.\n"
        "Si la question est ambigüe, ou ne traite pas de sujets politiques ou sociétaux, la réponse doit être \"Désolé, je suis un modèle spécialisé sur les sujets politiques et sociétaux, je ne peux pas répondre à cette question.\"\n"
        "=== QUESTION ===\n"
        f"{question}\n"
        "=== REPONSE ===\n"
    )

    # Print
    print("\nText generated:\n")

    # Generate until <|endoftext|> in sequence
    while not "<|endoftext|>" in text:

        # Generate 16 tokens
        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        length = 16 + ids.shape[1]

        gen_tokens = model.generate(
            ids,
            do_sample=True,
            max_length=length,
            temperature=0.9,
            top_p=0.95,
            use_cache=True,
        )

        # Retrieve generated text and print it
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        print(gen_text[len(text):])

        # Update text
        text = gen_text


