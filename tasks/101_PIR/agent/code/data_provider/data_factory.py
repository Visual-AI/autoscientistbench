from data_provider.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader
data_dict = {"ETTh1": Dataset_ETT_hour}

def data_provider(args, flag, data=None):
    Data = data_dict[args.data] if not data else data_dict[data]
    timeenc = 0 if args.embed != 'timeF' else 0

    assert flag in ('train', 'val', 'retrieval')
    shuffle_flag = True
    drop_last = False
    freq = args.freq

    stride = args.retrieval_stride if flag == 'retrieval' else 1

    if args.data == 'm4':
        drop_last = False
    size = [args.seq_len, args.label_len, args.pred_len]
    Data = data_dict[args.data]
    batch_size = args.batch_size
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=size,
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        percent=args.percent,
        output_index=args.output_index,
        stride=stride
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
