import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
from geopy.distance import geodesic

METRICS = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']

def gaussian_similarity(dist_km, sigma=5.0):
    return np.exp(- (dist_km ** 2) / (2 * sigma ** 2))


def prepare_data_infer(
    x_start_time: str = "2025033000",
    x_end_time: str = "2025033123",
    target_start_time: str = "2025040100",
    target_end_time: str = "2025040111",
    all_spot: List[str] = [
        "1001A", "1002A", "1003A", "1004A", "1005A", "1006A", "1007A", "1008A", "1009A", "1010A", "1011A", "1012A",
        "3281A", "3417A", "3418A",
        "3671A", "3672A", "3673A", "3674A", "3675A",
        "3694A", "3695A", "3696A", "3697A"
    ],
    data_dir: str = "history_data",
    location_file: str = "location.csv",
    fill_strategy: str = "mean",
    mode: str = "train",
    infer_spot: List[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    
    def validate_timestamp(ts: str) -> datetime:
        return datetime.strptime(ts, "%Y%m%d%H")

    def time_range(start: datetime, end: datetime) -> List[str]:
        result = []
        while start <= end:
            result.append(start.strftime("%Y%m%d%H"))
            start += timedelta(hours=1)
        return result

    def date_list(start: datetime, end: datetime) -> List[str]:
        current = start.date()
        end_date = end.date()
        days = []
        while current <= end_date:
            days.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return days

    x_start = validate_timestamp(x_start_time)
    x_end = validate_timestamp(x_end_time)
    t_start = validate_timestamp(target_start_time)
    t_end = validate_timestamp(target_end_time)

    if not (x_start < x_end < t_start < t_end):
        raise ValueError("Timestamps must be: x_start < x_end < target_start < target_end")

    train_spots = all_spot.copy()

    x_times = time_range(x_start, x_end)
    y_times = time_range(t_start, t_end)
    all_times = set(x_times + y_times)
    required_days = sorted(set(date_list(x_start, x_end) + date_list(t_start, t_end)))

    data_dict = {spot: {} for spot in all_spot}
    for day in required_days:
        file_path = os.path.join(data_dir, f"china_sites_{day}.csv")
        if not os.path.exists(file_path):
            print(f"[Warning] Missing file: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df = df[df['type'].isin(METRICS)]
        df['timestamp'] = df['hour'].apply(lambda h: f"{day}{int(h):02d}")
        for _, row in df.iterrows():
            ts = row['timestamp']
            if ts not in all_times:
                continue
            pol = row['type']
            for spot in all_spot:
                if ts not in data_dict[spot]:
                    data_dict[spot][ts] = [np.nan] * 7
                try:
                    idx = METRICS.index(pol)
                    val = float(row.get(spot, np.nan))
                    data_dict[spot][ts][idx] = val
                except:
                    continue

    def build_tensor(spot_list: List[str], time_seq: List[str]) -> np.ndarray:
        tensor = []
        for s in spot_list:
            seq = [data_dict[s].get(ts, [np.nan]*7) for ts in time_seq]
            tensor.append(seq)
        return np.array(tensor)

    def clean_data(x: np.ndarray, y: np.ndarray, spot_list: List[str], tag: str):
        keep_indices = []
        for i in range(x.shape[0]):
            if np.isnan(x[i]).all() or np.isnan(y[i]).all():
                print(f"[Warning] Drop spot '{spot_list[i]}' from {tag} due to all-NaN in x or y.")
                continue
            keep_indices.append(i)
        x = x[keep_indices]
        y = y[keep_indices]
        spot_list = [spot_list[i] for i in keep_indices]

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                if np.isnan(x[i, :, j]).all() or np.isnan(y[i, :, j]).all():
                    print(f"[Info] Spot '{spot_list[i]}' ({METRICS[j]}) in {tag} has all-NaN values; replaced with 0.0.")
                    x[i, :, j] = 0.0
                    y[i, :, j] = 0.0

        def interpolate(arr):
            mask = ~np.isnan(arr) # 原始非nan
            for i in range(arr.shape[0]):
                for j in range(arr.shape[2]):
                    series = arr[i, :, j]
                    nan_mask = np.isnan(series)
                    if nan_mask.all():
                        continue
                    not_nan = np.where(~nan_mask)[0]
                    for k in range(1, len(not_nan)):
                        l, r = not_nan[k - 1], not_nan[k]
                        if r - l > 1:
                            if fill_strategy == "mean":
                                fill_value = (series[l] + series[r]) / 2
                                series[l + 1:r] = fill_value
                            elif fill_strategy == "linear":
                                step = (series[r] - series[l]) / (r - l)
                                for m in range(1, r - l):
                                    series[l + m] = series[l] + step * m
                    
                    series[:not_nan[0]] = series[not_nan[0]]
                    series[not_nan[-1]+1:] = series[not_nan[-1]]
                    arr[i, :, j] = series
            new_mask = mask.astype(np.float32)
            return arr, new_mask

        x, x_mask = interpolate(x)
        y, _ = interpolate(y)
        return x, y, x_mask, spot_list

    train_x = build_tensor(train_spots, x_times)
    train_y = build_tensor(train_spots, y_times)
    train_x, train_y, train_x_mask, train_spots = clean_data(train_x, train_y, train_spots, "train")

    # 读取位置信息
    loc_df = pd.read_csv(location_file)
    coord_map = {
        row['监测点编码']: (float(row['经度']), float(row['纬度']))
        for _, row in loc_df.iterrows()
        if row['监测点编码'] in train_spots
    }
    
    # 根据模式决定推理点的坐标
    if mode == "infer" and infer_spot is not None:
        # 使用传入的infer_spot作为推理点
        infer_longitude, infer_latitude = infer_spot
    else:
        # 计算训练点的经纬度均值，作为推理点
        longitudes = [coord[0] for coord in coord_map.values()]
        latitudes = [coord[1] for coord in coord_map.values()]
        infer_longitude = np.mean(longitudes)
        infer_latitude = np.mean(latitudes)
    
    # 创建包含训练点和推理点的坐标数组
    coords = np.array([coord_map[s] for s in train_spots] + [(infer_longitude, infer_latitude)])
    
    # 计算邻接矩阵
    num = len(coords)
    adj_matrix = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        for j in range(num):
            d = geodesic((coords[i][1], coords[i][0]), (coords[j][1], coords[j][0])).km
            adj_matrix[i, j] = gaussian_similarity(d)

    return train_x, train_x_mask, train_y, adj_matrix, train_spots


def prepare_data_AQI_infer(
    x_start_time: str = "2025033000",
    time_scope: str = "days",
    all_spot: List[str] = [
        "1001A", "1002A", "1003A", "1004A", "1005A", "1006A", "1007A", "1008A", "1009A", "1010A", "1011A", "1012A",
        "3281A", "3417A", "3418A",
        "3671A", "3672A", "3673A", "3674A", "3675A",
        "3694A", "3695A", "3696A", "3697A"
    ],
    data_dir: str = "history_data",
    location_file: str = "location.csv",
    fill_strategy: str = "mean",
    do_standard: bool = True,
    mode: str = "train",
    infer_spot: List[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    - train_x: [n_train_spots, T, 7]
    - train_x_mask: [n_train_spots, T, 7]
    - train_y: [n_train_spots, horizon]
    - adj_matrix: [n_train_spots+1, n_train_spots+1]  # 包含推理点
    - train_spots: list
    """
    
    def validate_timestamp(ts: str) -> datetime:
        return datetime.strptime(ts, "%Y%m%d%H")

    def time_range(start: datetime, end: datetime, gap_hours: int = 1) -> List[str]:
        result = []
        current = start
        while current < end:
            result.append(current.strftime("%Y%m%d%H"))
            current += timedelta(hours=gap_hours)
        return result

    def date_list(start: datetime, end: datetime) -> List[str]:
        current = start.date()
        end_date = end.date()
        days = []
        while current <= end_date:
            days.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return days
    
    # 验证 x_start_time 格式
    try:
        x_start = validate_timestamp(x_start_time)
    except ValueError:
        raise ValueError(f"Invalid timestamp format: {x_start_time}. Expected format: YYYYMMDDHH")
    
    # 根据 time_scope 计算时间范围
    if time_scope == "days":
        # 检查是否为某日的0点
        if x_start.hour != 0:
            raise ValueError(f"For 'days' scope, x_start_time must be at 00:00 (got {x_start.hour:02d}:00)")
        
        # x 范围: 从 x_start_time 开始的 48 小时，每小时取样
        x_end = x_start + timedelta(hours=48)
        x_times = time_range(x_start, x_end)
        
        # target 范围: 从 x 结束后的 48 小时，每小时取样
        t_start = x_end
        t_end = t_start + timedelta(hours=48)
        y_times = time_range(t_start, t_end)
        
    elif time_scope == "month":
        # 检查是否为某月第一天的0点
        if x_start.day != 1 or x_start.hour != 0:
            raise ValueError(f"For 'month' scope, x_start_time must be the first day of month at 00:00 (got day {x_start.day}, hour {x_start.hour:02d})")
        
        # 计算当月的最后一天
        if x_start.month == 12:
            next_month_year = x_start.year + 1
            next_month = 1
        else:
            next_month_year = x_start.year
            next_month = x_start.month + 1
        
        # x 范围: 从 x_start_time 开始的那个月，每天取 0, 6, 12, 18 点
        x_end = datetime(next_month_year, next_month, 1, 0)
        
        # 生成 x 的时间点列表
        x_times = []
        current = x_start
        while current < x_end:
            for hour in [0, 6, 12, 18]:
                time_point = datetime(current.year, current.month, current.day, hour)
                if time_point >= x_start and time_point < x_end:
                    x_times.append(time_point.strftime("%Y%m%d%H"))
            current += timedelta(days=1)
        
        # target 范围: 下一个月，每天取 0, 6, 12, 18 点
        t_start = x_end
        
        # 计算下下个月的第一天
        if next_month == 12:
            next_next_month_year = next_month_year + 1
            next_next_month = 1
        else:
            next_next_month_year = next_month_year
            next_next_month = next_month + 1
        
        t_end = datetime(next_next_month_year, next_next_month, 1, 0)
        
        # 生成 target 的时间点列表
        y_times = []
        current = t_start
        while current < t_end:
            for hour in [0, 6, 12, 18]:
                time_point = datetime(current.year, current.month, current.day, hour)
                if time_point >= t_start and time_point < t_end:
                    y_times.append(time_point.strftime("%Y%m%d%H"))
            current += timedelta(days=1)
        
    elif time_scope == "year":
        # 检查是否为某月第一天的0点
        if x_start.day != 1 or x_start.hour != 0:
            raise ValueError(f"For 'year' scope, x_start_time must be the first day of month at 00:00 (got day {x_start.day}, hour {x_start.hour:02d})")
        
        # x 范围: 从 x_start_time 开始直到下一年这个月第一天的0点前
        x_end = datetime(x_start.year + 1, x_start.month, 1, 0)
        
        # target 范围: x 后的一年
        t_start = x_end
        t_end = datetime(x_start.year + 2, x_start.month, 1, 0)
        
        # 对于 year 模式，我们需要每小时的数据来计算每天的均值
        # 先获取所有小时的时间点
        x_hourly_times = time_range(x_start, x_end)
        y_hourly_times = time_range(t_start, t_end)
        
        # 将小时时间点按天分组，用于后续计算每天的均值
        x_daily_groups = {}
        for ts in x_hourly_times:
            day = ts[:8]  # YYYYMMDD
            if day not in x_daily_groups:
                x_daily_groups[day] = []
            x_daily_groups[day].append(ts)
        
        y_daily_groups = {}
        for ts in y_hourly_times:
            day = ts[:8]  # YYYYMMDD
            if day not in y_daily_groups:
                y_daily_groups[day] = []
            y_daily_groups[day].append(ts)
        
        # 每天只保留一个时间点（0点）作为代表，实际数据会在后面计算均值
        x_times = [day + "00" for day in sorted(x_daily_groups.keys())]
        y_times = [day + "00" for day in sorted(y_daily_groups.keys())]
        
    else:
        raise ValueError(f"Invalid time_scope: {time_scope}. Expected one of: 'days', 'month', 'year'")

    # 所有点都是训练点
    train_spot = all_spot.copy()

    all_times = set(x_times + y_times)
    if time_scope == "year":
        # 对于 year 模式，我们需要所有小时的数据
        all_times = set(x_hourly_times + y_hourly_times)
    
    required_days = sorted(set([ts[:8] for ts in all_times]))  # YYYYMMDD

    # 读取数据
    data_dict = {spot: {} for spot in all_spot}
    for day in required_days:
        file_path = os.path.join(data_dir, f"china_sites_{day}.csv")
        if not os.path.exists(file_path):
            print(f"[Warning] Missing file: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df = df[df['type'].isin(METRICS)]
        df['timestamp'] = df['hour'].apply(lambda h: f"{day}{int(h):02d}")
        for _, row in df.iterrows():
            ts = row['timestamp']
            if ts not in all_times and time_scope != "year":
                continue
            pol = row['type']
            for spot in all_spot:
                if ts not in data_dict[spot]:
                    data_dict[spot][ts] = [np.nan] * 7
                try:
                    idx = METRICS.index(pol)
                    val = float(row.get(spot, np.nan))
                    data_dict[spot][ts][idx] = val
                except:
                    continue

    def build_tensor(spot_list: List[str], time_seq: List[str]) -> np.ndarray:
        if time_scope != "year":
            # 对于 days 和 month 模式，直接构建张量
            tensor = []
            for s in spot_list:
                seq = [data_dict[s].get(ts, [np.nan]*7) for ts in time_seq]
                tensor.append(seq)
            return np.array(tensor)
        else:
            # 对于 year 模式，需要计算每天的均值
            tensor = []
            for s in spot_list:
                daily_seq = []
                for day_ts in time_seq:  # day_ts 格式为 YYYYMMDD00
                    day = day_ts[:8]  # YYYYMMDD
                    # 获取这一天所有小时的数据
                    if day in x_daily_groups:
                        hourly_data = [data_dict[s].get(ts, [np.nan]*7) for ts in x_daily_groups[day]]
                    else:
                        hourly_data = [data_dict[s].get(ts, [np.nan]*7) for ts in y_daily_groups[day]]
                    
                    # 转换为 numpy 数组并计算每个指标的均值
                    hourly_data = np.array(hourly_data)
                    daily_mean = np.nanmean(hourly_data, axis=0)
                    daily_seq.append(daily_mean.tolist())
                tensor.append(daily_seq)
            return np.array(tensor)

    def clean_data(x: np.ndarray, y: np.ndarray, spot_list: List[str], tag: str):
        keep_indices = []
        for i in range(x.shape[0]):
            if np.isnan(x[i]).all() or np.isnan(y[i]).all():
                print(f"[Warning] Drop spot '{spot_list[i]}' from {tag} due to all-NaN in x or y.")
                continue
            keep_indices.append(i)
        x = x[keep_indices]
        y = y[keep_indices]
        spot_list = [spot_list[i] for i in keep_indices]

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                if np.isnan(x[i, :, j]).all() or np.isnan(y[i, :, j]).all():
                    print(f"[Info] Spot '{spot_list[i]}' ({METRICS[j]}) in {tag} has all-NaN values; replaced with 0.0.")
                    x[i, :, j] = 0.0
                    y[i, :, j] = 0.0

        def interpolate(arr):
            mask = ~np.isnan(arr) # 原始非nan
            for i in range(arr.shape[0]):
                for j in range(arr.shape[2]):
                    series = arr[i, :, j]
                    nan_mask = np.isnan(series)
                    if nan_mask.all():
                        continue
                    not_nan = np.where(~nan_mask)[0]
                    for k in range(1, len(not_nan)):
                        l, r = not_nan[k - 1], not_nan[k]
                        if r - l > 1:
                            if fill_strategy == "mean":
                                fill_value = (series[l] + series[r]) / 2
                                series[l + 1:r] = fill_value
                            elif fill_strategy == "linear":
                                step = (series[r] - series[l]) / (r - l)
                                for m in range(1, r - l):
                                    series[l + m] = series[l] + step * m
                    
                    series[:not_nan[0]] = series[not_nan[0]]
                    series[not_nan[-1]+1:] = series[not_nan[-1]]
                    arr[i, :, j] = series
            new_mask = mask.astype(np.float32)
            return arr, new_mask

        x, x_mask = interpolate(x)
        y, _ = interpolate(y)

        y_AQI = y[:, :, 0] # [n_spots, horizon]
        
        return x, y_AQI, x_mask, spot_list

    train_x = build_tensor(train_spot, x_times)
    train_y = build_tensor(train_spot, y_times)
    train_x, train_y, train_x_mask, train_spot = clean_data(train_x, train_y, train_spot, "train")
    
    if do_standard:
        AQI_mean = 0.0
        AQI_std = 0.0
        
        # Per Air Metric - Time: x+y - Spot: train
        for j in range(7):
            all_data = []

            # x for 7 metrics
            all_data.append(train_x[:, :, j].flatten())

            if j == 0: # y for AQI
                all_data.append(train_y.flatten())

            all_data = np.concatenate([data for data in all_data if data.size > 0])
            mean = np.mean(all_data)
            std = np.std(all_data)
            
            if std == 0:
                print(f"[Warning] Standard deviation for {METRICS[j]} is 0, skipping normalization")
                continue

            train_x[:, :, j] = (train_x[:, :, j] - mean) / std

            if j == 0:
                AQI_mean = mean
                AQI_std = std
                train_y = (train_y - mean) / std
                
            print(f"Normalized {METRICS[j]}: mean={mean:.4f}, std={std:.4f}")

    # 读取位置信息
    loc_df = pd.read_csv(location_file)
    coord_map = {
        row['监测点编码']: (float(row['经度']), float(row['纬度']))
        for _, row in loc_df.iterrows()
        if row['监测点编码'] in train_spot
    }
    
    # 根据模式决定推理点的坐标
    if mode == "infer" and infer_spot is not None:
        # 使用传入的infer_spot作为推理点
        infer_longitude, infer_latitude = infer_spot
    else:
        # 计算训练点的经纬度均值，作为推理点
        longitudes = [coord[0] for coord in coord_map.values()]
        latitudes = [coord[1] for coord in coord_map.values()]
        infer_longitude = np.mean(longitudes)
        infer_latitude = np.mean(latitudes)
    
    # 创建包含训练点和推理点的坐标数组
    coords = np.array([coord_map[s] for s in train_spot] + [(infer_longitude, infer_latitude)])
    
    # 计算邻接矩阵
    num = len(coords)
    adj_matrix = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        for j in range(num):
            d = geodesic((coords[i][1], coords[i][0]), (coords[j][1], coords[j][0])).km
            adj_matrix[i, j] = gaussian_similarity(d)

    if mode == "infer":
        return train_x, train_x_mask, train_y, adj_matrix, train_spot, AQI_mean, AQI_std
    
    return train_x, train_x_mask, train_y, adj_matrix, train_spot


def prepare_data_AQI(
    x_start_time: str = "2025033000",
    time_scope: str = "days",
    all_spot: List[str] = [
        "1001A", "1002A", "1003A", "1004A", "1005A", "1006A", "1007A", "1008A", "1009A", "1010A", "1011A", "1012A",
        "3281A", "3417A", "3418A",
        "3671A", "3672A", "3673A", "3674A", "3675A",
        "3694A", "3695A", "3696A", "3697A"
    ],
    test_spot: List[str] = ['3697A', '3696A'],
    val_spot: List[str] = ['3695A', '3694A'],
    data_dir: str = "history_data",
    location_file: str = "location.csv",
    fill_strategy: str = "mean",
    do_standard: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    - train_x: [n_train_spots, T, 7]
    - train_x_mask: [n_train_spots, T, 7]
    - train_y: [n_train_spots, horizon]
    - val_y: [n_val_spots, horizon]
    - test_y: [n_test_spots, horizon]
    - adj_matrix: [n_all_spots, n_all_spots]
    - all_spots: list
    """
    
    def validate_timestamp(ts: str) -> datetime:
        return datetime.strptime(ts, "%Y%m%d%H")

    def time_range(start: datetime, end: datetime, gap_hours: int = 1) -> List[str]:
        result = []
        current = start
        while current < end:
            result.append(current.strftime("%Y%m%d%H"))
            current += timedelta(hours=gap_hours)
        return result

    def date_list(start: datetime, end: datetime) -> List[str]:
        current = start.date()
        end_date = end.date()
        days = []
        while current <= end_date:
            days.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return days
    
    # 验证 x_start_time 格式
    try:
        x_start = validate_timestamp(x_start_time)
    except ValueError:
        raise ValueError(f"Invalid timestamp format: {x_start_time}. Expected format: YYYYMMDDHH")
    
    # 根据 time_scope 计算时间范围
    if time_scope == "days":
        # 检查是否为某日的0点
        if x_start.hour != 0:
            raise ValueError(f"For 'days' scope, x_start_time must be at 00:00 (got {x_start.hour:02d}:00)")
        
        # x 范围: 从 x_start_time 开始的 48 小时，每小时取样
        x_end = x_start + timedelta(hours=48)
        x_times = time_range(x_start, x_end)
        
        # target 范围: 从 x 结束后的 48 小时，每小时取样
        t_start = x_end
        t_end = t_start + timedelta(hours=48)
        y_times = time_range(t_start, t_end)
        
    elif time_scope == "month":
        # 检查是否为某月第一天的0点
        if x_start.day != 1 or x_start.hour != 0:
            raise ValueError(f"For 'month' scope, x_start_time must be the first day of month at 00:00 (got day {x_start.day}, hour {x_start.hour:02d})")
        
        # 计算当月的最后一天
        if x_start.month == 12:
            next_month_year = x_start.year + 1
            next_month = 1
        else:
            next_month_year = x_start.year
            next_month = x_start.month + 1
        
        # x 范围: 从 x_start_time 开始的那个月，每天取 0, 6, 12, 18 点
        x_end = datetime(next_month_year, next_month, 1, 0)
        
        # 生成 x 的时间点列表
        x_times = []
        current = x_start
        while current < x_end:
            for hour in [0, 6, 12, 18]:
                time_point = datetime(current.year, current.month, current.day, hour)
                if time_point >= x_start and time_point < x_end:
                    x_times.append(time_point.strftime("%Y%m%d%H"))
            current += timedelta(days=1)
        
        # target 范围: 下一个月，每天取 0, 6, 12, 18 点
        t_start = x_end
        
        # 计算下下个月的第一天
        if next_month == 12:
            next_next_month_year = next_month_year + 1
            next_next_month = 1
        else:
            next_next_month_year = next_month_year
            next_next_month = next_month + 1
        
        t_end = datetime(next_next_month_year, next_next_month, 1, 0)
        
        # 生成 target 的时间点列表
        y_times = []
        current = t_start
        while current < t_end:
            for hour in [0, 6, 12, 18]:
                time_point = datetime(current.year, current.month, current.day, hour)
                if time_point >= t_start and time_point < t_end:
                    y_times.append(time_point.strftime("%Y%m%d%H"))
            current += timedelta(days=1)
        
    elif time_scope == "year":
        # 检查是否为某月第一天的0点
        if x_start.day != 1 or x_start.hour != 0:
            raise ValueError(f"For 'year' scope, x_start_time must be the first day of month at 00:00 (got day {x_start.day}, hour {x_start.hour:02d})")
        
        # x 范围: 从 x_start_time 开始直到下一年这个月第一天的0点前
        x_end = datetime(x_start.year + 1, x_start.month, 1, 0)
        
        # target 范围: x 后的一年
        t_start = x_end
        t_end = datetime(x_start.year + 2, x_start.month, 1, 0)
        
        # 对于 year 模式，我们需要每小时的数据来计算每天的均值
        # 先获取所有小时的时间点
        x_hourly_times = time_range(x_start, x_end)
        y_hourly_times = time_range(t_start, t_end)
        
        # 将小时时间点按天分组，用于后续计算每天的均值
        x_daily_groups = {}
        for ts in x_hourly_times:
            day = ts[:8]  # YYYYMMDD
            if day not in x_daily_groups:
                x_daily_groups[day] = []
            x_daily_groups[day].append(ts)
        
        y_daily_groups = {}
        for ts in y_hourly_times:
            day = ts[:8]  # YYYYMMDD
            if day not in y_daily_groups:
                y_daily_groups[day] = []
            y_daily_groups[day].append(ts)
        
        # 每天只保留一个时间点（0点）作为代表，实际数据会在后面计算均值
        x_times = [day + "00" for day in sorted(x_daily_groups.keys())]
        y_times = [day + "00" for day in sorted(y_daily_groups.keys())]
        
    else:
        raise ValueError(f"Invalid time_scope: {time_scope}. Expected one of: 'days', 'month', 'year'")

    # 验证其他参数
    for s in test_spot + val_spot:
        if s not in all_spot:
            raise ValueError(f"Invalid spot: {s}")
    if set(test_spot) & set(val_spot):
        raise ValueError("test_spot and val_spot must not overlap")

    train_spot = [s for s in all_spot if s not in val_spot and s not in test_spot]

    all_times = set(x_times + y_times)
    if time_scope == "year":
        # 对于 year 模式，我们需要所有小时的数据
        all_times = set(x_hourly_times + y_hourly_times)
    
    required_days = sorted(set([ts[:8] for ts in all_times]))  # YYYYMMDD

    # 读取数据
    data_dict = {spot: {} for spot in all_spot}
    for day in required_days:
        file_path = os.path.join(data_dir, f"china_sites_{day}.csv")
        if not os.path.exists(file_path):
            print(f"[Warning] Missing file: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df = df[df['type'].isin(METRICS)]
        df['timestamp'] = df['hour'].apply(lambda h: f"{day}{int(h):02d}")
        for _, row in df.iterrows():
            ts = row['timestamp']
            if ts not in all_times and time_scope != "year":
                continue
            pol = row['type']
            for spot in all_spot:
                if ts not in data_dict[spot]:
                    data_dict[spot][ts] = [np.nan] * 7
                try:
                    idx = METRICS.index(pol)
                    val = float(row.get(spot, np.nan))
                    data_dict[spot][ts][idx] = val
                except:
                    continue

    def build_tensor(spot_list: List[str], time_seq: List[str]) -> np.ndarray:
        if time_scope != "year":
            # 对于 days 和 month 模式，直接构建张量
            tensor = []
            for s in spot_list:
                seq = [data_dict[s].get(ts, [np.nan]*7) for ts in time_seq]
                tensor.append(seq)
            return np.array(tensor)
        else:
            # 对于 year 模式，需要计算每天的均值
            tensor = []
            for s in spot_list:
                daily_seq = []
                for day_ts in time_seq:  # day_ts 格式为 YYYYMMDD00
                    day = day_ts[:8]  # YYYYMMDD
                    # 获取这一天所有小时的数据
                    if day in x_daily_groups:
                        hourly_data = [data_dict[s].get(ts, [np.nan]*7) for ts in x_daily_groups[day]]
                    else:
                        hourly_data = [data_dict[s].get(ts, [np.nan]*7) for ts in y_daily_groups[day]]
                    
                    # 转换为 numpy 数组并计算每个指标的均值
                    hourly_data = np.array(hourly_data)
                    daily_mean = np.nanmean(hourly_data, axis=0)
                    daily_seq.append(daily_mean.tolist())
                tensor.append(daily_seq)
            return np.array(tensor)

    def clean_data(x: np.ndarray, y: np.ndarray, spot_list: List[str], tag: str):
        keep_indices = []
        for i in range(x.shape[0]):
            if np.isnan(x[i]).all() or np.isnan(y[i]).all():
                print(f"[Warning] Drop spot '{spot_list[i]}' from {tag} due to all-NaN in x or y.")
                continue
            keep_indices.append(i)
        x = x[keep_indices]
        y = y[keep_indices]
        spot_list = [spot_list[i] for i in keep_indices]

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                if np.isnan(x[i, :, j]).all() or np.isnan(y[i, :, j]).all():
                    print(f"[Info] Spot '{spot_list[i]}' ({METRICS[j]}) in {tag} has all-NaN values; replaced with 0.0.")
                    x[i, :, j] = 0.0
                    y[i, :, j] = 0.0

        def interpolate(arr):
            mask = ~np.isnan(arr) # 原始非nan
            for i in range(arr.shape[0]):
                for j in range(arr.shape[2]):
                    series = arr[i, :, j]
                    nan_mask = np.isnan(series)
                    if nan_mask.all():
                        continue
                    not_nan = np.where(~nan_mask)[0]
                    for k in range(1, len(not_nan)):
                        l, r = not_nan[k - 1], not_nan[k]
                        if r - l > 1:
                            if fill_strategy == "mean":
                                fill_value = (series[l] + series[r]) / 2
                                series[l + 1:r] = fill_value
                            elif fill_strategy == "linear":
                                step = (series[r] - series[l]) / (r - l)
                                for m in range(1, r - l):
                                    series[l + m] = series[l] + step * m
                    
                    series[:not_nan[0]] = series[not_nan[0]]
                    series[not_nan[-1]+1:] = series[not_nan[-1]]
                    arr[i, :, j] = series
            new_mask = mask.astype(np.float32)
            return arr, new_mask

        x, x_mask = interpolate(x)
        y, _ = interpolate(y)

        y_AQI = y[:, :, 0] # [n_spots, horizon]
        
        return x, y_AQI, x_mask, spot_list

    train_x = build_tensor(train_spot, x_times)
    train_y = build_tensor(train_spot, y_times)
    train_x, train_y, train_x_mask, train_spot = clean_data(train_x, train_y, train_spot, "train")

    val_x = build_tensor(val_spot, x_times)
    val_y = build_tensor(val_spot, y_times)
    val_x, val_y, _, val_spot = clean_data(val_x, val_y, val_spot, "val")

    test_x = build_tensor(test_spot, x_times)
    test_y = build_tensor(test_spot, y_times)
    test_x, test_y, _, test_spot = clean_data(test_x, test_y, test_spot, "test")

    all_spots = train_spot + val_spot + test_spot
    
    if do_standard:
        # Per Air Metric - Time: x+y - Spot: train+val+test
        for j in range(7):
            all_data = []

            # x for 7 metrics
            all_data.append(train_x[:, :, j].flatten())
            all_data.append(val_x[:, :, j].flatten())
            all_data.append(test_x[:, :, j].flatten())

            if j == 0: # y for AQI
                all_data.append(train_y.flatten())
                all_data.append(val_y.flatten())
                all_data.append(test_y.flatten())

            all_data = np.concatenate([data for data in all_data if data.size > 0])
            mean = np.mean(all_data)
            std = np.std(all_data)
            
            if std == 0:
                print(f"[Warning] Standard deviation for {METRICS[j]} is 0, skipping normalization")
                continue

            train_x[:, :, j] = (train_x[:, :, j] - mean) / std
            val_x[:, :, j] = (val_x[:, :, j] - mean) / std
            test_x[:, :, j] = (test_x[:, :, j] - mean) / std
            

            if j == 0:
                train_y = (train_y - mean) / std
                val_y = (val_y - mean) / std
                test_y = (test_y - mean) / std
                
            print(f"[Info] Normalized {METRICS[j]}: mean={mean:.4f}, std={std:.4f}")

    loc_df = pd.read_csv(location_file)
    coord_map = {
        row['监测点编码']: (row['经度'], row['纬度'])
        for _, row in loc_df.iterrows()
        if row['监测点编码'] in all_spots
    }
    coords = np.array([coord_map[s] for s in all_spots])
    num = len(coords)
    adj_matrix = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        for j in range(num):
            d = geodesic((coords[i][1], coords[i][0]), (coords[j][1], coords[j][0])).km
            adj_matrix[i, j] = gaussian_similarity(d)

    return train_x, train_x_mask, train_y, val_y, test_y, adj_matrix, all_spots


if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, test_y, train_x_mask, val_x_mask, test_x_mask, adj_matrix, all_spots_used = prepare_data()
    print("train_x.shape:", train_x.shape)
    print("train_x:", train_x)
    print("train_y.shape:", train_y.shape)
    print("train_y:", train_y)
    print("val_x.shape:", val_x.shape)
    print("val_x:", val_x)
    print("val_y.shape:", val_y.shape)
    print("val_y:", val_y)
    print("test_x.shape:", test_x.shape)
    print("test_x:", test_x)
    print("test_y.shape:", test_y.shape)
    print("test_y:", test_y)
    print("adj_matrix.shape:", adj_matrix.shape)
    print("adj_matrix:", adj_matrix)
    print("spots used:", all_spots_used)
