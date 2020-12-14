import random
import numpy as np
import json
from tqdm import tqdm
import os


def taint(data, n=None, ratio=None, seed=None):
    # data: json data of a file
    # ratio: ratio of words to be tainted
    # count words in data
    total_words = 0
    for item in data["form"]:
        total_words += len([w for w in item['words'] if w["text"].strip() != ""])
    if not total_words:
        print("No words in data.")
        return

    if ratio:
        n = int(ratio * total_words)
    elif not n and not ratio:
        print("Must pass at least one value for either ratio or n.")
        raise AssertionError

    random.seed(seed)

    tainted_ids = random.sample(range(total_words), min(total_words, n))
    word_count = 0
    for i, item in enumerate(data["form"]):
        for j, word in enumerate(item["words"]):
            if word["text"].strip() == "":
                continue
            if word_count in tainted_ids:
                data["form"][i]['words'][j]['text'] = '<tainted>'
            word_count += 1
    return data


def tear(data, n_split=2):
    word_boxes = []
    for item in data["form"]:
        # left, top, right, bottom
        word_boxes.extend([w["box"] for w in item["words"] if w["text"].strip() != ""])
    if not len(word_boxes):
        print("No words in data.")
        return

    # define split location
    # find the min left & top and max right & bottom
    word_boxes = np.array(word_boxes)
    min_left = min(word_boxes[:, 0])
    min_top = min(word_boxes[:, 1])
    max_right = max(word_boxes[:, 2])
    max_bottom = max(word_boxes[:, 3])
    # candidate split points

    _splits = {}
    for i in range(n_split):
        for j in range(n_split):
            _left = min_left + i * (max_right - min_left) / n_split
            _right = _left + (max_right - min_left) / n_split
            _top = min_top + j * (max_bottom - min_top) / n_split
            _bottom = _top + (max_bottom - min_top) / n_split
            _splits[(_left, _top, _right, _bottom)] = []

    # put words into _splits
    for wb in word_boxes:
        wb_center = [(wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2]
        for _sb in _splits:
            # if the center of word box is inside split
            if wb_center[0] >= _sb[0] and wb_center[0] <= _sb[2] and wb_center[1] >= _sb[1] and wb_center[1] <= _sb[3]:
                _splits[_sb].append(wb)
                break
        else:
            raise LookupError

    # adjust _splits margin according to word boxes
    split_boxes = []
    for _, _word_boxes_in_split in _splits.items():
        # may exist empty split
        if not len(_word_boxes_in_split):
            continue
        _word_boxes_in_split = np.array(_word_boxes_in_split)
        left = min(_word_boxes_in_split[:, 0])
        top = min(_word_boxes_in_split[:, 1])
        right = max(_word_boxes_in_split[:, 2])
        bottom = max(_word_boxes_in_split[:, 3])
        split_boxes.append([left, top, right, bottom])

    print(split_boxes)
    for i, item in enumerate(data["form"]):
        for j, word in enumerate(item["words"]):
            if word["text"].strip() == "":
                continue
            wb = word["box"]
            wb_center = ((wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2)
            for sb_idx, sb in enumerate(split_boxes):
                if wb_center[0] >= sb[0] and wb_center[0] <= sb[2] and wb_center[1] >= sb[1] and wb_center[1] <= sb[3]:
                    data["form"][i]['words'][j]['box'] = [wb[0] - sb[0], wb[1] - sb[1], wb[2] - sb[0], wb[3] - sb[1]]
                    data["form"][i]['words'][j]['split'] = sb_idx
                    break
            else:
                raise LookupError

    return data

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def generate_taint_dataset(data_dir, output_dir, ratio, seed=4321):
    all_files = os.listdir(data_dir)
    for file in tqdm(all_files, total=len(all_files)):
        if file[-5:] != '.json': continue
        input_file = os.path.join(data_dir, file)
        with open(input_file, 'r', encoding="utf8") as rf:
            data = json.load(rf)

        data_tainted = taint(data, ratio=ratio, seed=seed)

        output_file = os.path.join(output_dir, file)
        with open(output_file, 'w', encoding="utf8") as wf:
            json.dump(data_tainted, wf, indent=4, cls=NumpyEncoder)


def generate_tear_dataset(data_dir, output_dir, n_split=2):
    all_files = os.listdir(data_dir)
    for file in tqdm(all_files, total=len(all_files)):
        if file[-5:] != '.json': continue
        input_file = os.path.join(data_dir, file)
        with open(input_file, 'r', encoding="utf8") as rf:
            data = json.load(rf)

        data_torn = tear(data, n_split=n_split)

        output_file = os.path.join(output_dir, file)
        with open(output_file, 'w', encoding="utf8") as wf:
            json.dump(data_torn, wf, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    #with open('dataset/testing_data/annotations/82092117.json', 'r') as rf:
    #    data = json.load(rf)
    #print(data)
    #data_torn = tear(data, n_split=2)
    #print(data_torn)
    for ratio in np.arange(0.1, 0.6, 0.1):
        if not os.path.isdir('/data4/jiayun/FUNSD/taint/{}/training_data/annotations'.format(ratio)):
            os.makedirs('/data4/jiayun/FUNSD/taint/{}/training_data/annotations'.format(ratio))
        if not os.path.isdir('/data4/jiayun/FUNSD/taint/{}/testing_data/annotations'.format(ratio)):
            os.makedirs('/data4/jiayun/FUNSD/taint/{}/testing_data/annotations'.format(ratio))
        generate_taint_dataset('/data4/jiayun/FUNSD/dataset/training_data/annotations', '/data4/jiayun/FUNSD/taint/{}/training_data/annotations'.format(ratio), ratio=ratio)
        generate_taint_dataset('/data4/jiayun/FUNSD/dataset/testing_data/annotations', '/data4/jiayun/FUNSD/taint/{}/testing_data/annotations'.format(ratio), ratio=ratio)
        print("{} taint done.".format(ratio))
    #generate_tear_dataset('/data4/jiayun/FUNSD/dataset/training_data/annotations', '/data4/jiayun/FUNSD/torn/training_data/annotations')
    #generate_tear_dataset('/data4/jiayun/FUNSD/dataset/testing_data/annotations', '/data4/jiayun/FUNSD/torn/testing_data/annotations')