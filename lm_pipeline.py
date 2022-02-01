from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
import sys
import codecs
import pickle
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

if __name__ == '__main__':

    model_name, beam, input_file, output_file, = sys.argv[1:]

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    text_generator = TextGenerationPipeline(model, tokenizer, device = 0)

    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line:
                continue
            items = line.strip().split('\t')
            event, relation = items[:2]
            re = text_generator(f"{event} {relation}", max_length=100, num_beams=int(beam), num_return_sequences = int(beam))
            results.append(re)
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"saved to {output_file}")