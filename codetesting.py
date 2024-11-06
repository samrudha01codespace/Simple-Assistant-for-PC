#Just train your model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load the model and tokenizer
model_name = "gpt2"  # Replace with the actual model name you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load different Wikimedia Wikipedia datasets
datasets = [
    load_dataset("wikimedia/wikipedia", "20231101.ab"),
    load_dataset("wikimedia/wikipedia", "20231101.ace"),
    load_dataset("wikimedia/wikipedia", "20231101.ady"),
    load_dataset("wikimedia/wikipedia", "20231101.en"),  # Adding English Wikipedia
    load_dataset("wikimedia/wikipedia", "20231101.es"),  # Adding Spanish Wikipedia
    load_dataset("wikimedia/wikipedia", "20231101.fr"),  # Adding French Wikipedia  # Adding a dataset of books
    load_dataset("cnn_dailymail", "3.0.0")  # Adding CNN/Daily Mail news articles
]

# Flatten the dataset list for easier access
all_examples = []
for ds in datasets:
    all_examples.extend(ds['train'])

# Example of generating text based on user input or dataset examples
user_input = input("Enter your prompt or press Enter to use dataset examples: ")

if user_input:
    # Use user input as the prompt
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated response:", response)
else:
    # Use dataset examples if no user input is provided
    for example in all_examples:
        prompt = example['text']  # Modify 'text' to match the appropriate column
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
