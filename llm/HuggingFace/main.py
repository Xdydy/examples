from datasets import concatenate_datasets, load_dataset
bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

dataset = concatenate_datasets([bookcorpus, wiki])

d = dataset.train_test_split(test_size=0.1)

def dataset_to_text(dataset, output_filename="data.txt"):
    with open(output_filename, "w") as f:
        for t in dataset['text']:
            print(t, file=f)

dataset_to_text(d['train'], "train.txt")
dataset_to_text(d['test'], "test.txt")