from utils import dataset

def choose(args):
    if args.dataset == 'scene':
        print('Data Preparation of scene')
        file_name = ["./data/scene_data.csv", "./data/scene_label.csv", "./data/scene_com_label.csv"]
        train_loader, test_loader = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 6
        input_dim = 294
    elif args.dataset == "yeast":
        print('Data Preparation of yeast')
        file_name = ["./data/yeast_data.csv", "./data/yeast_label.csv", "./data/yeast_com_label.csv"]
        train_loader, test_loader = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 14
        input_dim = 103

    return train_loader, test_loader, num_class, input_dim