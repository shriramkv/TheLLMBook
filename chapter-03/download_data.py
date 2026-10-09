# Downloads the Tiny Shakespeare text (about 1.1 MB) used by mini_gpt.py
import urllib.request
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(URL, "input.txt")
print("Saved input.txt")
