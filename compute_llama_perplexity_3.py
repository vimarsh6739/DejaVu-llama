from datasets import load_dataset
import evaluate

perplexity = evaluate.load("perplexity", module_type="metric")
input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] # doctest: +SKIP
input_texts = [s for s in input_texts if s!='']
results = perplexity.compute(model_id='/shared/vsathia2/model/', predictions=input_texts, batch_size=4)
print(list(results.keys()))
print(round(results["mean_perplexity"], 2))
