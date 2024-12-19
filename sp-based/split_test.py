import json


with open('./datasets/val.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(len(data))
    if len(data) != 11797:  # original number of samples in `val.json`
        # the file has been restructured with proper indentation
        raise Exception('The file `val.json` has been split into `val.json` and `test.json` already!\nNo need to run this script again.')

    # fetch the last 5000 samples as test data
    test_data = data[-5000:]
    remaining_data = data[:-5000]

    with open('./datasets/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    with open('./datasets/val.json', 'w', encoding='utf-8') as f:
        json.dump(remaining_data, f, ensure_ascii=False, indent=4)

    # at the same time, restructure `train.json` with proper indentation
    with open('./datasets/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('./datasets/train.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print('Successfully split `val.json` into `val.json` and `test.json`, and restructured all files with indentation.')