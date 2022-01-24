from transformers import (GPT2LMHeadModel, 
                    BertTokenizer, 
                    TextGenerationPipeline,
                    Trainer,
                    default_data_collator)
import os
import sys
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from datasets import load_from_disk
from lyc.utils import get_tokenizer
from lyc.train import get_base_hf_args

class ZhComet(GPT2LMHeadModel):

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        labels_weights=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none' if labels_weights is not None else 'mean', ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if labels_weights is not None:
                labels_weights = labels_weights.repeat(shift_logits.shape[1], 1).transpose(1,0).contiguous().view(-1)
                loss = (loss * labels_weights).mean()

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

def make_dataset(model_type, tokenizer, max_length, data_path='data/zh/'):
    path = os.path.join(data_path, f'{model_type}_dataset')
    print(data_path, path)
    ds = load_from_disk(path)

    def preprocess(examples):
        line = examples['line']
        text = line[0] +' <'+line[1] +'> ' + line[2] +'<END>'
        result = tokenizer(text, max_length = max_length, truncation='longest_first', padding='max_length')
        result['labels'] = result["input_ids"].copy()
        if model_type == 'distill':
            result['labels_weights'] = examples['score']
        return result

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    return ds


def customize_tokenizer(tokenizer):
    categories = []
    categories += ["oEffect"]
    categories += ["oReact"]
    categories += ["oWant"]
    categories += ["xAttr"]
    categories += ["xEffect"]
    categories += ["xIntent"]
    categories += ["xNeed"]
    categories += ["xReact"]
    categories += ["xWant"]

    effect_types = ['<' + i + '>' for i in categories]
    special_tokens = ['PersonX', 'PersonY', '<END>'] + effect_types

    tokenizer.add_tokens(special_tokens, special_tokens=True)
    pad_id = tokenizer.pad_token_id

    return tokenizer, pad_id

if __name__ == '__main__':
    # model_name: gpt model path or name,
    # model_type: baseline / distill
    model_name, model_type, output_path, logging_dir, data_path, = sys.argv[1:]
    max_length = 128
    assert model_type in ['baseline', 'distill']

    tokenizer = get_tokenizer('bert-base-chinese')
    tokenizer, pad_id = customize_tokenizer(tokenizer)
    end_id = tokenizer.convert_tokens_to_ids('<END>')

    ds = make_dataset(model_type, tokenizer, max_length, data_path=data_path)

    # preparing model
    model = ZhComet.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.eos_token_id = end_id

    args = get_base_hf_args(
        output_dir=output_path,
        train_batch_size=32,
        epochs = 5,
        logging_steps=500,
        logging_dir = logging_dir,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()

    