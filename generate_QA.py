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
GEN_LENGTH = 8
CONTEXT_LENGTH = MAX_LENGTH - GEN_LENGTH

# Run 
while True:

    # Get input from user
    question = str(input("\n\nVotre question pour Overton :\n"))
    if len(question) == 0:
        continue
    
    # Construct text var
    text = (
        #"La réponse doit précise, argumentée, logique et véridique."
        "=== QUESTION ===\n"
        f"{question}\n"
        "=== REPONSE ===\n"
    )

    # Retrieve tokens
    ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    if ids.shape[1] > CONTEXT_LENGTH :
        ids = ids[:,-CONTEXT_LENGTH:]

    # Print
    flag = False
    print("Overton réfléchis...\n\n")
    
    # Generate until <|endoftext|> in sequence
    while not "<|endoftext|>" in text:

        res_tokens = model.generate(
            ids,
            temperature=0.8, # The more the temperature is, the more the outputs will be random (0 to 1+)
            top_p=0.95, # Sum of probability to take into account (the most likelihood words) (0 to 1)
            #top_k=50, # Length of the set of words to pick in (the most likelihood words) (1 to 50+)
            #rep=0.25, # Penalty the model has to generate repetition (0 to 1 : 0 is no penalty)
            max_length=MAX_LENGTH,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Print beginning of answer
        if not flag:
            print('Réponse :\n')
            flag = True

        # Retrieve gen tokens
        gen_tokens = res_tokens[:,-GEN_LENGTH:]

        # Retrieve generated text and print it
        gen_text = tokenizer.batch_decode()[0]
        print(gen_text, flush=True, end='\n')

        if "<|endoftext|>" in gen_text:
            break

        # Update ids
        ids = ids.cat([ids, gen_tokens], dim=1)[:,-CONTEXT_LENGTH:]


