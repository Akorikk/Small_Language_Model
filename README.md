# Small_Language_Model

For data scraping i am using bs4 and webBaseloader 


if not os.path.exists("train.bin"):
This line checks if the file train.bin already exists on your computer.

Purpose: Avoid re-processing the dataset if you’ve already done it. Saves time.

Think of it like: “Do I already have the processed numbers? If yes, skip everything below
Convert the entire dataset into numerical form (token IDs) ready for training.
df.map(...) → applies a function to every row of the dataset.

processing → the function you wrote earlier that converts text → token IDs + length.

remove_columns=['text'] → removes the original text from the new dataset, keeping only token IDs and lengths.

desc="tokenizing the splits" → shows a progress description.

num_proc=8 → uses 8 parallel processes to speed up tokenization.

for split, dset in tokenized.items():
Many datasets have splits like 'train', 'validation', 'test'.

This loop goes through each split and processes them separately.

Purpose: Handle each part of your dataset individually.

arr_len = np.sum(dset['len'], dtype=np.uint64)
dset['len'] contains the length of token IDs for each row.

np.sum() calculates total number of tokens in this split.

dtype=np.uint64 → ensures we can handle very large numbers.

Purpose: Figure out how big the final array should be to store all tokens.

filename = f'{split}.bin'
dtype = np.uint16
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
filename → name of the binary file, e.g., train.bin

dtype=np.uint16 → each token ID will be stored as a 16-bit number

np.memmap → creates a memory-mapped array on disk instead of loading everything into RAM.

Purpose: Efficiently store millions of token IDs without crashing your computer.

total_batches = 1024
idx = 0
total_batches → we’ll process the dataset in 1024 smaller chunks.

idx → keeps track of where to write in the big array.

Purpose: Avoid memory overload by writing data in small batches.

for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
    batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
    arr_batch = np.concatenate(batch['ids'])
    arr[idx : idx + len(arr_batch)] = arr_batch
    idx += len(arr_batch)
dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True) → get one small chunk of the dataset.

with_format('numpy') → make it NumPy arrays so we can write easily.

np.concatenate(batch['ids']) → flatten token IDs into one long array.

arr[idx : idx + len(arr_batch)] = arr_batch → write this batch into the memory-mapped file at the right position.

idx += len(arr_batch) → move the pointer to the next position.

Purpose: Efficiently write all token IDs to disk batch by batch, avoiding memory crashes.

arr.flush()
flush() ensures all data is written to the binary file.

Without this, some data might stay in memory and not be saved.

Summary / Why we need all this

Convert text → token IDs (numerical representation).

Store them in binary files (train.bin, val.bin) for fast loading.

Use memory mapping to handle huge datasets without crashing RAM.

Process in batches to save memory and time.

💡 Think of it like:

Your text is like a giant book.

Tokenization → turn words into numbers.

Memory mapping + batches → store the giant book page by page on your computer so you can read it quickly later.

If you want, I can also draw a diagram of this pipeline showing:
text → token IDs → batch → memory-mapped binary file → ready for LM training.