import torch
import pickle
import csv
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def numpy_to_tensor(arr, use_gpu=True):
    if use_gpu:
        return torch.tensor(arr).to(device).float()
    else:
        return torch.tensor(arr).float()


def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def list2str(a_list):
    return ''.join([str(i) for i in a_list])


def pickle_load(f):
    return pickle.load(open(f, 'rb'))


def pickle_dump(obj, f):
    pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def json_read(path):
    with open(path, 'r') as f:
        content = json.load(f)
    f.close()
    return content


def json_write(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)
    f.close()


def csv_read(path):
    content = list()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            content.append(row)
    f.close()
    header = content[0]
    content = content[1:]
    return header, content

