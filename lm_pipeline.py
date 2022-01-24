from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
import sys

if __name__ == '__main__':

    model_name, beam, input_file, = sys.argv[1:]

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    text_generator = TextGenerationPipeline(model, tokenizer)

    print(text_generator("PersonX去酒吧 <oWant>", max_length=100, num_beams=beams))