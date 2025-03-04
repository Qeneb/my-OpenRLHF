import re
import torch
from prepare_dataset import extract_answer_from_model_output

# TODO: 可以改进，当前这个并不能很好地提取最后一个数字，留给模型思考的token并不多
def extract_last_number(text):
   """
   Extracts the last number appearing in the text.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The last number in the text, or None if no number is found.

   Explanation:
       1. Removes dollar signs and percent symbols from the text.
       2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
       3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
       4. Returns the found number as a float, or None if no match is found.
   """
   text = text.replace('$', '').replace('%', '')
   pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
   match = re.search(pattern, text)
   return float(match.group(1)) if match else None

def extract_single_number(text):
   """
   Extracts a single number from text if exactly one number is present.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.

   Explanation:
       1. Uses regex to find all numbers in the text (including negative numbers and decimals).
       2. If exactly one number is found, returns it as a float.
       3. If zero or multiple numbers are found, returns None.
   """
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None

def evaluate_model(model, tokenizer, eval_examples, device):
   """
   Evaluates the model on a set of examples and prints detailed results.

   Args:
       model: The language model to evaluate.
       tokenizer: The tokenizer for encoding inputs and decoding outputs.
       eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
       device: The device (CPU or GPU) to run evaluation on.

   Returns:
       float: The accuracy percentage (correct predictions / total examples * 100).

   Explanation:
       1. Sets the model to evaluation mode.
       2. For each example in the evaluation set:
          - Encodes the prompt and generates a response using the model.
          - Extracts the predicted answer from the generated response.
          - Compares the predicted answer with the expected answer using multiple methods:
            a. Exact string matching
            b. Single number extraction and comparison
            c. Last number extraction and comparison
          - Prints detailed information about each example.
       3. Calculates and returns the overall accuracy.
       4. Returns the model to training mode.
   """
   model.eval()
   correct = 0
   total = len(eval_examples)
   print("\n" + "="*50)
   print("EVALUATION ON", total, "EXAMPLES")
   print("="*50)

   for example in eval_examples:
       # Get the prompt and expected answer
       full_prompt = example["prompt"]
       expected = example["answer"]

       # Tokenize and generate response
       inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
       with torch.no_grad():
           outputs = model.generate(
               inputs,
               max_new_tokens=512,
               temperature=0.7,
               num_return_sequences=1,
               pad_token_id=tokenizer.pad_token_id,
               eos_token_id=tokenizer.eos_token_id,
               forced_eos_token_id=tokenizer.eos_token_id,
               early_stopping=False,
           )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)

       try:
           # Extract answer and check correctness
           predicted = extract_answer_from_model_output(response)

           # Try different matching methods
           if predicted == expected:  # Exact match
               is_correct = True
           else:
               # Try single number matching
               pred_num = extract_single_number(str(predicted))
               exp_num = extract_single_number(str(expected))
               if pred_num is not None and exp_num is not None and pred_num == exp_num:
                   is_correct = True
               else:
                   # Try last number matching
                   pred_num = extract_last_number(str(predicted))
                   exp_num = extract_last_number(str(expected))
                   is_correct = (pred_num is not None and exp_num is not None and
                               pred_num == exp_num)

           # Update counter for correct answers
           if is_correct:
               correct += 1

           # Print evaluation details
           print("\nPrompt:")
           print(full_prompt)
           print("\nExpected Answer:")
           print(expected)
           print("\nExtracted Answer:")
           print(predicted)
           print("\nFull Generated Response:")
           print(response)
           print("\nCorrect:", "✓" if is_correct else "✗")
           print("-"*50)

       except Exception as e:
           print("\nFailed to parse model output for prompt:")
           print(full_prompt)
           print("Error:", e)
           print("-"*50)

   # Calculate and print final accuracy
   accuracy = (correct / total) * 100
   print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
   print("="*50)

   # Return model to training mode
   model.train()
   return accuracy

if __name__ == '__main__':
    # Hugging Face libraries for transformer models
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prepare_dataset import prepare_dataset
    import random
    import os

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"

    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model downloaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    all_data = prepare_dataset("train")
    random.shuffle(all_data)
    size_of_eval_data = 1 # change to a smaller value to save time or to a larger number for a more reliable estimate
    eval_data = all_data[:size_of_eval_data]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    accuracy = evaluate_model(model, tokenizer, eval_data, device)

    print(f"Pre-GRPO Accuracy: {accuracy:.2f}%")