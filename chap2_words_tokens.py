import ssl
import urllib.request

import re

# to implement byte pair encoding
try:
    from importlib.metadata import version
    import tiktoken
    print("tiktoken version:", version("tiktoken"))
except Exception:
    tiktoken = None
    print("tiktoken not installed; skipping BPE features.")

try:
       import certifi
except Exception:  # pragma: no cover - certifi optional
       certifi = None

url = (
       "https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt"
)
file_path = "the-verdict.txt"

try:
       if certifi:
              ssl_context = ssl.create_default_context(cafile=certifi.where())
       else:
              ssl_context = ssl.create_default_context()

       with urllib.request.urlopen(url, context=ssl_context) as response, open(
              file_path, "wb"
       ) as handle:
              handle.write(response.read())
except ssl.SSLError as exc:
       raise RuntimeError(
              "SSL verification failed. If you're on macOS, install certificates via "
              "the 'Install Certificates.command' for your Python, or run: "
              "python -m pip install --upgrade certifi."
       ) from exc

# load file 
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
print(raw_text[:100])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
if "<UNK>" not in vocab:
       vocab["<UNK>"] = len(vocab)
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) 

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# now start the tokenizer class
tokenizer = SimpleTokenizerV1(vocab)
text = """It's the last he painted, you know," 
          Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print('Encoded IDs:')
print(ids)

# now see if we can turn it back to text
print('Decoded text:')
print(tokenizer.decode(ids))


# now try with some text that isnt included in the test file
#text = "Hello, do you like tea?"
#print('test text')
#print(tokenizer.encode(text))
#print('decoded text')
#print(tokenizer.decode(tokenizer.encode(text))) 

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# updated tokenizer class - deals with unknown words and adds <unk>
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))

# BPE toekenizer
tokenizer = tiktoken.get_encoding("gpt2")
print("BPE tokenizer vocab size:", tokenizer.n_vocab)

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# data loader for text file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print('Tokens read in from text file')
print(len(enc_text))

enc_sample = enc_text[50:]
print('Sample tokens:')

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")


# next word prediction
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

# convert token ids into text
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

