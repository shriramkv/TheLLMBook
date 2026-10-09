# Lab 4.1 Part B: compare real tokenisers across languages (pip install tiktoken)
import tiktoken

messages = {
    "English": "Your parcel will be delivered to the warehouse tomorrow morning. Please keep your invoice ready.",
    "Hindi": "आपका पार्सल कल सुबह गोदाम पर पहुंचा दिया जाएगा। कृपया अपना चालान तैयार रखें।",
    "Tamil": "உங்கள் பார்சல் நாளை காலை கிடங்குக்கு வழங்கப்படும். தயவுசெய்து உங்கள் விலைப்பட்டியலை தயாராக வைத்திருங்கள்.",
}

for name in ("gpt2", "cl100k_base", "o200k_base"):
    enc = tiktoken.get_encoding(name)
    counts = {lang: len(enc.encode(text)) for lang, text in messages.items()}
    print(f"{name:12s}", counts)
# Our results (October 2026):
# gpt2         {'English': 17, 'Hindi': 114, 'Tamil': 291}
# cl100k_base  {'English': 17, 'Hindi': 82, 'Tamil': 145}
# o200k_base   {'English': 17, 'Hindi': 25, 'Tamil': 32}
