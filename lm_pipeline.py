from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

if __name__ == '__main__':

    model_name, beam, = sys.argv[1:]

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    text_generator = TextGenerationPipeline(model, tokenizer)

    print(text_generator("PersnX坐车 <oWant>", max_length=100, num_beams=int(beam), num_return_sequences = int(beam)))