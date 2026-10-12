"""Writes a small sample dataset in the chat format of Section 6.4 (data/train.jsonl, data/valid.jsonl).

KrishiSaathi is a fictional product. These examples show the FORMAT only; a real project needs
hundreds to thousands of expert-reviewed conversations (Section 6.4). The replies are general
guidance written for teaching, not agronomic advice."""
import json, random

SYSTEM = "You are KrishiSaathi, a friendly crop-advice assistant. Reply with JSON: {\"category\", \"reply\"}."
EXAMPLES = [
 ("My tomato leaves are curling upwards and the edges look dry. We had a hot week.", "crop_stress",
  "Upward leaf curl after a hot spell is often heat and water stress. Water early in the morning, add mulch to keep the soil cool, and check the underside of a few leaves for whiteflies. If the curling spreads to new leaves, send us a photo."),
 ("Small white insects fly up when I touch my chilli plants.", "pest",
  "These sound like whiteflies. Check the underside of the leaves, remove badly affected leaves, and put up yellow sticky traps. Please speak to your local agriculture office before using any spray, and follow the label exactly."),
 ("How often should I run drip irrigation for sugarcane in summer?", "irrigation",
  "It depends on your soil and the crop's age, so start by checking soil moisture with your finger at root depth each morning. Water when the top few centimetres are dry, and run the drip early or late in the day to reduce evaporation."),
 ("What is today's onion price in Lasalgaon market?", "market_price",
  "I cannot see live market prices. Please check the official market board website or app for today's arrival and price data at Lasalgaon."),
 ("Is there any subsidy for drip irrigation?", "scheme",
  "Drip irrigation is often supported under government schemes, but rules and amounts change. Please check with your taluka agriculture office or the official scheme portal for the current conditions and how to apply."),
 ("My cow has stopped eating since yesterday.", "out_of_scope",
  "I can only help with crops. Please contact your nearest veterinary doctor today, as a cow that stops eating needs to be seen quickly."),
 ("Yellow patches on older leaves of my maize.", "nutrient",
  "Yellowing that starts on older leaves can point to a nitrogen shortage, but other causes look similar. Please send a clear photo of a few affected leaves and tell us when you last applied fertiliser."),
 ("The soil in my field cracks a lot after it dries.", "soil",
  "Deep cracks usually mean a clay-rich soil. Adding organic matter such as compost and keeping the soil covered with mulch or a cover crop helps it hold moisture and crack less."),
 ("Brown spots with rings on my potato leaves after rain.", "disease",
  "Brown spots with rings after wet weather can be a fungal leaf disease. Remove the worst leaves, avoid watering the leaves themselves, and send us a photo. Ask your agriculture office before choosing any fungicide."),
 ("When should I sow groundnut in the kharif season?", "sowing",
  "Sowing is usually timed with the onset of steady monsoon rain, when the soil has enough moisture. Check the recommended window for your district with the local agriculture office, as it varies by region."),
 ("Rats are damaging my paddy bunds.", "pest",
  "Keep the bunds trimmed and free of weeds so rats have fewer places to hide, and work with neighbouring farmers so the whole area is managed together. Ask your agriculture office about safe control methods."),
 ("Can I use the same seeds from this year's hybrid tomato crop next season?", "seeds",
  "Seeds saved from hybrid plants usually do not grow true to type, so the next crop may be uneven. For a hybrid crop, buying fresh certified seed each season is the safer choice."),
 ("My drip lines are getting blocked.", "irrigation",
  "Blocked drippers are often caused by dirt or salt build-up. Clean the filter regularly, flush the lines at the ends, and ask your dealer about acid treatment if the water is hard."),
 ("leaves eaten holes caterpillars cabbage", "pest",
  "Caterpillars on cabbage can be picked off by hand when numbers are small. Check plants every few days, especially the underside of leaves. Please speak to your agriculture office before using any spray."),
 ("How much fertiliser for one acre of wheat?", "nutrient",
  "The right amount depends on your soil test and variety. Please get a soil test done through your local soil testing lab; your Soil Health Card will give recommended doses for your field."),
 ("Weather will be very hot next week. What should I do for my vegetables?", "crop_stress",
  "Water early in the morning, mulch the beds to keep moisture in, and give young seedlings some shade if you can. Avoid applying fertiliser during the hottest days."),
 ("Who won the cricket match yesterday?", "out_of_scope",
  "I can only help with questions about crops and farming. Is there anything about your fields I can help with today?"),
 ("White powder on my pumpkin leaves.", "disease",
  "A white powdery coating on leaves is often powdery mildew. Remove badly affected leaves, keep space between plants for air flow, and send us a photo so we can confirm."),
 ("Mango flowers are dropping a lot.", "crop_stress",
  "Some flower drop is normal in mango. Heavy drop can follow very dry or very wet weather or pest attack. Check for small insects on the flower clusters and send us a photo."),
 ("Which crop can I grow after paddy in winter with less water?", "sowing",
  "Pulses such as green gram or black gram are often grown after paddy because they need less water. Check which varieties are recommended for your district with your agriculture office."),
]
random.seed(0); random.shuffle(EXAMPLES)
rows = [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q},
                      {"role": "assistant", "content": json.dumps({"category": c, "reply": r}, ensure_ascii=False)}]}
        for q, c, r in EXAMPLES]
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:16]) + "\n")
with open("data/valid.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[16:]) + "\n")
print(f"wrote {len(rows[:16])} training and {len(rows[16:])} validation conversations to data/")
