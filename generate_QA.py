from transformers import GPTJForCausalLM, AutoTokenizer
import argparse
import time

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
    
    MAX_CONTEXT_LENGTH = 1024
    MAX_GEN_LENGTH = 256

    # Generate until <|endoftext|> in sequence
    while not "<|endoftext|>" in text:

        # Generate 16 tokens
        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        if ids.shape[1] > MAX_CONTEXT_LENGTH :
            ids = ids[-MAX_CONTEXT_LENGTH:]

        length = 1 + ids.shape[1]

        gen_tokens = model.generate(
            ids,
            temperature=0.7, # The more the temperature is, the more the outputs will be random (0 to 1+)
            top_p=0.95, # Sum of probability to take into account (the most likelihood words) (0 to 1)
            top_k=40, # Length of the set of words to pick in (the most likelihood words) (1 to 50+)
            rep=0.25, # Penalty the model has to generate repetition (0 to 1 : 0 is no penalty)
            max_length=MAX_GEN_LENGTH,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Retrieve generated text and print it
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        for c in gen_text:
            time.sleep(0.001)
            print(c, end='')

        # Update text
        text = gen_text


