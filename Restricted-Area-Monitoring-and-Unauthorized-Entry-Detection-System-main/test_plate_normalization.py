import re

def normalize_indian_plate(raw_text):
    text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

    # Require the standard 10-character layout:
    # AA 00 AA 0000
    if len(text) != 10:
        return None

    state = text[:2]
    district = text[2:4].translate(str.maketrans({
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "Z": "2",
        "S": "5", "B": "8", "G": "6"
    }))
    series = text[4:6]
    number = text[6:10].translate(str.maketrans({
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "Z": "2",
        "S": "5", "B": "8", "G": "6"
    }))

    if (
        not state.isalpha()
        or len(state) != 2
        or not district.isdigit()
        or not series.isalpha()
        or not number.isdigit()
    ):
        return None

    return state + district + series + number


samples = [
    "KA2SAB1234",
    "KA25AB1234",
    "KA25A8123",
    "0KA2SAB1234",
]

for sample in samples:
    print(f"{sample!r} -> {normalize_indian_plate(sample)}")
