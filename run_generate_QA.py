from transformers import GPTJForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()


model = GPTJForCausalLM.from_pretrained(args.model).half().to("cuda")
tokenizer = AutoTokenizer.from_pretrained(args.model)


while True:
    question = str(input("\n\nQuestion to ask to the model: "))
    if len(question) == 0:
        continue

    text = (
        "=== QUESTION ===\n",
        question + '\n',
        '=== REPONSE ===\n',
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    # add the length of the prompt tokens to match with the mesh-tf generation
    max_length = 400 + ids.shape[1]

    gen_tokens = model.generate(
        ids,
        do_sample=True,
        #min_length=max_length,
        max_length=max_length,
        temperature=0.9,
        top_p=0.95,
        use_cache=True,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print("\nText generated:\n")
    print(gen_text)
