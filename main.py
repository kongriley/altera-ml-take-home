import transformers as tr
import torch
import math
from tqdm import tqdm

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)


def contrastive_generation(amateur, expert, prompt, max_tokens) -> str:
	device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

	alpha = 0.1
	log_alpha = math.log(alpha)

	amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur, use_cache=True, torch_dtype=torch.bfloat16, device_map="auto")
	expert_model = tr.AutoModelForCausalLM.from_pretrained(expert, use_cache=True, torch_dtype=torch.bfloat16, device_map="auto")

	amateur_past_key_values = None
	expert_past_key_values = None
	
	inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
	out = ""

	for i in tqdm(range(max_tokens)):
		current_inputs = inputs if i == 0 else inputs[:, -1:]

		with torch.no_grad():
			amateur_outputs = amateur_model(input_ids=current_inputs, past_key_values=amateur_past_key_values, use_cache=True)
			expert_outputs = expert_model(input_ids=current_inputs, past_key_values=expert_past_key_values, use_cache=True)

		amateur_past_key_values = amateur_outputs.past_key_values
		expert_past_key_values = expert_outputs.past_key_values

		amateur_logits = amateur_outputs.logits[:, -1, :]
		expert_logits = expert_outputs.logits[:, -1, :]

		amateur_logprobs = torch.log_softmax(amateur_logits, dim=-1)
		expert_logprobs = torch.log_softmax(expert_logits, dim=-1)

		score = expert_logprobs - amateur_logprobs
		score[expert_logprobs < (log_alpha + torch.max(amateur_logprobs, dim=-1).values)] = -float("inf")

		next_token = torch.argmax(score, dim=-1)
		
		# stop generation if we see <|im_end|> or <|endoftext|> token
		if next_token.item() in (151645, 151643):
			break
	
		out += tokenizer.decode(next_token)
		inputs = next_token.unsqueeze(0)

	return out

if __name__ == "__main__":
	print(contrastive_generation(amateur_path, expert_path, prompt, 256))