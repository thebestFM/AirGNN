import torch
import torch.optim as optim
import numpy as np
import time
import os
import pickle
import hashlib

from model_AQI import GGTN
from prepare_data import prepare_data_AQI
from utils import *
from config import *

def get_cache_filename(args):
    params = {
        'x_start_time': args.x_start_time,
        'time_scope': args.time_scope,
        'all_spot': args.all_spot,
        'test_spot': args.test_spot,
        'val_spot': args.val_spot,
        'data_dir': args.data_dir,
        'location_file': args.location_file,
        'fill_strategy': args.fill_strategy,
        'do_standard': args.do_standard
    }

    params_str = str(sorted(params.items()))
    hash_obj = hashlib.md5(params_str.encode())
    return f"data_cache_{hash_obj.hexdigest()}.pkl"


def main(args):
    cache_dir = "processed_data"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_data_path = os.path.join(cache_dir, get_cache_filename(args))

    if os.path.exists(cache_data_path):
        print(f"Loading processed data from: {cache_data_path}")
        with open(cache_data_path, 'rb') as f:
            data = pickle.load(f)
            train_x, train_x_mask, train_y, val_y, test_y, adj_matrix, all_spots_list = data
    else:
        print("Processing data...")
        train_x, train_x_mask, train_y, val_y, test_y, adj_matrix, all_spots_list = prepare_data_AQI(
            args.x_start_time, 
            args.time_scope, 
            args.all_spot, 
            args.test_spot, 
            args.val_spot, 
            args.data_dir, 
            args.location_file, 
            fill_strategy=args.fill_strategy, 
            do_standard=args.do_standard
        )

        print(f"Processed data has been saved at: {cache_data_path}")
        with open(cache_data_path, 'wb') as f:
            data = (train_x, train_x_mask, train_y, val_y, test_y, adj_matrix, all_spots_list)
            pickle.dump(data, f)
    
    num_train = train_y.shape[0]
    num_val = val_y.shape[0]
    num_test = test_y.shape[0]

    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")

    if args.concat_mask:
        train_x = np.concatenate([train_x, train_x_mask], axis=-1) # [N_train, T, 2F]

    train_idx = list(range(num_train))
    val_idx = list(range(num_train, num_train + num_val))
    test_idx = list(range(num_train + num_val, num_train + num_val + num_test))
    
    num_nodes = num_train + num_val + num_test

    input_dim = train_x.shape[-1]
    horizon = train_y.shape[1]

    train_idx_tensor = torch.tensor(train_idx).to(device)
    train_x = torch.tensor(train_x).unsqueeze(0).float().to(device)
    
    train_y = torch.tensor(train_y).unsqueeze(0).float().to(device)
    val_y = torch.tensor(val_y).unsqueeze(0).float().to(device)
    test_y = torch.tensor(test_y).unsqueeze(0).float().to(device)

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)

    val_perf = {
        'mae': float('inf'),
        'rmse': float('inf'),
        'r2': float('-inf'),
    }
    test_perf = {
        'mae': float('inf'),
        'rmse': float('inf'),
        'r2': float('-inf'),
    }

    no_better_count = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x, adj)[:, train_idx, :]
        true = train_y

        loss = masked_mae(pred, true)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(train_x, adj)

            val_mae = masked_mae(y_pred[:, val_idx, :], val_y).item()
            scheduler.step(val_mae)

            val_rmse = masked_rmse(y_pred[:, val_idx, :], val_y).item()
            val_r2 = masked_r2(y_pred[:, val_idx, :], val_y).item()

            test_mae = masked_mae(y_pred[:, test_idx, :], test_y).item()
            test_rmse = masked_rmse(y_pred[:, test_idx, :], test_y).item()
            test_r2 = masked_r2(y_pred[:, test_idx, :], test_y).item()

        improved = False

        if val_mae < val_perf['mae']:
            val_perf['mae'] = val_mae
            test_perf['mae'] = test_mae
            improved = True

        if val_rmse < val_perf['rmse']:
            val_perf['rmse'] = val_rmse
            test_perf['rmse'] = test_rmse
            improved = True

        if val_r2 > val_perf['r2']:
            val_perf['r2'] = val_r2
            test_perf['r2'] = test_r2
            improved = True

        if improved:
            no_better_count = 0
            print(f"Better Val at Epoch {epoch:04d}: "
                f"Train Loss = {loss.item():.4f}, "
                f"Val MAE = {val_mae:.4f}, RMSE = {val_rmse:.4f}, R2 = {val_r2:.4f}, "
                f"Test MAE = {test_mae:.4f}, RMSE = {test_rmse:.4f}, R2 = {test_r2:.4f}")
        else:
            no_better_count += 1

        if no_better_count >= args.patience:
            print(f"Early Stop at Epoch {epoch}")
            break

    print("\nTraining complete.")
    print(f"Best Val:")
    for metric, value in val_perf.items():
        print(f"\t{metric.upper()}: {value:.4f}")
    print(f"\nLast Test:")
    for metric, value in test_perf.items():
        print(f"\t{metric.upper()}: {value:.4f}")


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
