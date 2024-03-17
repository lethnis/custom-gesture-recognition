from pathlib import Path

text = Path("classes.txt").read_text()

classes = [i.strip() for i in text.split(",")]

print(classes)

dict_classes = {idx: val.strip() for idx, val in enumerate(text.split(","))}

print(dict_classes)
