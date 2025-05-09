import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_dir', type=str, default="history_data")
    parser.add_argument('--location_file', type=str, default="location.csv")
    parser.add_argument('--all_spot', nargs='+', default=[
        "1001A", "1003A", "1004A", "1005A", "1006A", "1007A", "1008A", "1009A", "1010A", "1011A",
        "3281A", "3418A",
        "3671A", "3672A", "3673A", "3674A", "3675A",
        "3694A", "3695A", "3696A", "3697A"
    ])
    parser.add_argument('--test_spot', nargs='+', default=['3697A', '3696A'])
    parser.add_argument('--val_spot', nargs='+', default=['3695A', '3694A'])
    parser.add_argument('--x_start_time', type=str, default="2023010100")
    parser.add_argument('--time_scope', type=str, default='days', choices=['days', 'month', 'year'])
    parser.add_argument('--fill_strategy', type=str, default='mean', help="NaN value fill strategy", choices=['mean', 'linear'])
    parser.add_argument('--concat_mask', action='store_true', help="Whether to concatenate filled mask")
    parser.add_argument('--infer_spot', nargs='+', type=float, default=[116.46226, 39.91054])
    parser.add_argument('--do_standard', action='store_true')

    # Model
    parser.add_argument('--time_mode', type=str, default='none', choices=['none', 'days', 'weeks', 'months'])
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension size")
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--temporal_model", default="tcn", choices=["gru", "transformer", "tcn"])
    
    # for temporal_model == "transformer"
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--trans_layers", type=int, default=2)
    
    # for temporal_model == "tcn"
    parser.add_argument("--tcn_layers", type=int, default=4)
    parser.add_argument("--t_block", default="separable", choices=['basic', 'glu', 'separable', 'inception'])

    # Training 
    parser.add_argument('--seed', type=int, default=2025, help="Random seed")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--num_epochs', type=int, default=10000, help="Max training epochs")
    parser.add_argument('--num_epochs_wo_eval', type=int, default=3000)

    parser.add_argument('--patience', type=int, default=100, help="Early stopping patience")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to be used, -1 for CPU")

    return parser.parse_args()
