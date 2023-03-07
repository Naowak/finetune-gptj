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

# Define constants
MAX_LENGTH = 1024
GEN_LENGTH = 250
CONTEXT_LENGTH = MAX_LENGTH - GEN_LENGTH

# Run 
while True:

    # Get input from user
    question = str(input("\n\nVotre question pour Overton :\n"))
    if len(question) == 0:
        continue

    print('\n\nOverton réfléchit...\n\n')
    
    # Construct text var
    text = (
        #"La réponse doit précise, argumentée, logique et véridique."
        "=== QUESTION ===\n"
        f"{question}\n"
        "=== REPONSE ===\n"
    )

    # Retrieve tokens
    ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")[:,-CONTEXT_LENGTH:]
    max_length = ids.shape[1] + GEN_LENGTH

    # Generate    
    res_tokens = model.generate(
        ids,
        temperature=0.8, # The more the temperature is, the more the outputs will be random (0 to 1+)
        top_p=0.95, # Sum of probability to take into account (the most likelihood words) (0 to 1)
        #top_k=50, # Length of the set of words to pick in (the most likelihood words) (1 to 50+)
        #rep=0.25, # Penalty the model has to generate repetition (0 to 1 : 0 is no penalty)
        max_length=max_length,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Print result
    gen_text = tokenizer.batch_decode(res_tokens)[0]
    print("Réponse :\n")
    print(gen_text)

    