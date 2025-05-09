import torch
import numpy as np
import time
import os

from model_AQI import GGTN
from prepare_data import prepare_data_AQI_infer
from utils import *
from config import *

def main(args):
    model_name = f"model_{args.x_start_time}_{args.time_scope}_SD_{args.do_standard}"
    save_dir = os.path.join("saved_model", model_name)
    
    if not os.path.exists(save_dir):
        print(f"Model path {save_dir} not found.")
        return
    
    model_path = os.path.join(save_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    checkpoint = torch.load(model_path)

    param_mismatch = False
    mismatch_params = []

    check_params = [
        'x_start_time', 'time_scope', 'fill_strategy', 'do_standard',
        'time_mode', 'temporal_model', 'hop', 'hidden_dim', 
        'n_heads', 'trans_layers', 'tcn_layers', 't_block', 'concat_mask'
    ]
    
    for param in check_params:
        if param in checkpoint and getattr(args, param) != checkpoint[param]:
            param_mismatch = True
            mismatch_params.append(param)
    
    if param_mismatch:
        print("No matched parameters found.")
        # for param in mismatch_params:
        #     print(f"\t{param}: {checkpoint[param]}")
        print("Needed param:")
        for param in mismatch_params:
            print(f"\t{param}: {getattr(args, param)}")
        return

    train_x, train_x_mask, train_y, adj_matrix, train_spots_list, AQI_mean, AQI_std = prepare_data_AQI_infer(
        args.x_start_time, 
        args.time_scope, 
        args.all_spot, 
        args.data_dir, 
        args.location_file, 
        fill_strategy=args.fill_strategy, 
        do_standard=args.do_standard, 
        mode="infer",
        infer_spot=args.infer_spot
    )

    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")

    if args.concat_mask:
        train_x = np.concatenate([train_x, train_x_mask], axis=-1) # [N_train, T, 2F]

    # 训练节点索引
    train_idx = list(range(train_x.shape[0]))
    # 总节点数 = 训练节点数 + 1（推理节点）
    num_nodes = len(train_x) + 1
    # 推理节点索引
    infer_idx = num_nodes - 1

    input_dim = train_x.shape[-1]
    horizon = train_y.shape[1]

    train_idx_tensor = torch.tensor(train_idx).to(device)
    train_x = torch.tensor(train_x).unsqueeze(0).float().to(device)
    
    adj = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

    model = GGTN(
        num_nodes=num_nodes,
        train_idx=train_idx_tensor,
        in_dim=input_dim,
        time_mode=args.time_mode,
        hid_dim=args.hidden_dim,
        hop=args.hop,
        horizon=horizon,
        dropout=args.dropout,
        temporal_model=args.temporal_model,
        n_heads=args.n_heads,
        trans_layers=args.trans_layers,
        tcn_layers=args.tcn_layers,
        t_block=args.t_block
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        pred = model(train_x, adj)
        infer_pred = pred[0, infer_idx, :].cpu().numpy()

    if args.do_standard:
        print(f"AQI_mean: {AQI_mean}, AQI_std: {AQI_std}")
        infer_pred = infer_pred * AQI_std + AQI_mean

    print("\nInfer Result:")
    if args.infer_spot:
        print(f"Infer Pos: [{args.infer_spot[0]}, {args.infer_spot[1]}]")
    
    print("\nAQI:")
    print(infer_pred)


if __name__ == "__main__":
    args = get_args()
    print("-"*50)
    print("\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    print("-"*50)
    
    set_random_seed(args.seed)
    t1 = time.time()
    main(args)
    t2 = time.time()
    print("\nTime Cost (s): {:.2f}".format(t2 - t1))
