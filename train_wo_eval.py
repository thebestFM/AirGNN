import torch
import torch.optim as optim
import numpy as np
import time
import os

from model_AQI import GGTN
from prepare_data import prepare_data_AQI_infer
from utils import *
from config import *

def main(args):
    train_x, train_x_mask, train_y, adj_matrix, train_spots_list = prepare_data_AQI_infer(
        args.x_start_time, 
        args.time_scope, 
        args.all_spot, 
        args.data_dir, 
        args.location_file, 
        fill_strategy=args.fill_strategy, 
        do_standard=args.do_standard, 
        mode="train"
    )

    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")

    if args.concat_mask:
        train_x = np.concatenate([train_x, train_x_mask], axis=-1) # [N_train, T, 2F]

    # 训练节点索引
    train_idx = list(range(train_x.shape[0]))
    # 总节点数 = 训练节点数 + 1（推理节点）
    num_nodes = len(train_x) + 1

    input_dim = train_x.shape[-1]
    horizon = train_y.shape[1]

    train_idx_tensor = torch.tensor(train_idx).to(device)
    train_x = torch.tensor(train_x).unsqueeze(0).float().to(device)
    train_y = torch.tensor(train_y).unsqueeze(0).float().to(device)
    
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, verbose=True
    )

    model_name = f"model_{args.x_start_time}_{args.time_scope}_SD_{args.do_standard}"
    save_dir = os.path.join("saved_model", model_name)
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')
    no_better_count = 0
    best_model_state = None

    for epoch in range(1, args.num_epochs_wo_eval + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x, adj)[:, train_idx, :]
        true = train_y

        loss = masked_mae(pred, true)
        loss.backward()
        optimizer.step()

        scheduler.step(loss)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()
            no_better_count = 0
            print(f"Epoch {epoch:04d}: Better Train Loss = {loss.item():.4f}")
        else:
            no_better_count += 1
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:04d}: Train Loss = {loss.item():.4f}")

        if args.patience > 0 and no_better_count >= args.patience:
            print(f"Early Stop at Epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_path = os.path.join(save_dir, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_spots': args.all_spot,
        'x_start_time': args.x_start_time,
        'time_scope': args.time_scope,
        'fill_strategy': args.fill_strategy,
        'do_standard': args.do_standard,
        'time_mode': args.time_mode, 
        'temporal_model': args.temporal_model,
        'hop': args.hop,
        'hidden_dim': args.hidden_dim,
        'n_heads': args.n_heads,
        'trans_layers': args.trans_layers,
        'tcn_layers': args.tcn_layers,
        't_block': args.t_block,
        'concat_mask': args.concat_mask,    
    }, model_path)

    print("\nTraining complete.")
    print(f"The best model has been saved at: {model_path}")


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
