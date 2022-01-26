import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())
sys.path.append('/user/HS502/yl02706/comet-commonsense/')

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full-categories_None/1e-05_adam_64_15500.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")
    parser.add_argument("--input_file", type=str, help="prediction input file, each line should be in this format: <event>\t<category>\t<sample_method>")
    parser.add_argument("--output_file", type=str, help="output file path")

    args = parser.parse_args()

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    sampler = interactive.set_sampler(opt, args.sampling_algorithm, data_loader)
    results = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line:
                continue
            event, relation = line.strip().split('\t')
            outputs = interactive.get_conceptnet_sequence(
                event, model, sampler, data_loader, text_encoder, relation)
            print(outputs)
            outputs = outputs[relation]
            results.append(outputs)
    # with open(args.output_file, 'w', encoding='utf-8') as f:
    #     for r in results:
    #         f.wirte(','.join(r)+'\n')
    with open(args.output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'write results to {args.output_file}')