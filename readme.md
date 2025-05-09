# GGTN

**GGTN** (Geographically Generalizable Temporal Network) is a graduation project focusing on temporal forecasting models for **geographic locations without historical data**.

## ⚡ Quick Start

1. **Quickly Download Data**:  
   Download the `history_data/` directory and place it in the project root from:  
   [Google Drive Link](https://drive.google.com/file/d/19pbTwdU_hoNw35TnhJjpOOliQP8XoTed/view?usp=drive_link)

2. **Weekly Updated Data**:  
   Weekly updated air quality data for China can be obtained from:  
   [https://quotsoft.net/air/](https://quotsoft.net/air/) → `中国空气质量数据` Section

## ▶️ Run Training

```bash
python train.py \
  --x_start_time 2025022700 \
  --time_scope days \
  --do_standard \
  --hop 2 \
  --dropout 0.0 \
  --tcn_layers 5 \
  --seed 2025 \
  --lr 5e-3 \
  --wd 5e-5
```

- `--time_scope` supports: `days`, `month`, or `year`, defining the input/output time span.
- `--x_start_time` is the starting timestamp of the input sequence.  
  The model predicts the target variable in the next `time_scope` period based on a 7-dimensional input over the previous `time_scope` period.

---

Feel free to modify parameters and experiment with different model architectures.
