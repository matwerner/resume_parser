import pandas as pd
import string
import sys

def ispunct(text: str) -> bool:
    return text in string.punctuation + '–'


def mask_text(text: str) -> str:
    masked_text = ''
    for c in text:
        masked_character = '?'
        if c.isdigit():
            masked_character = '0'
        elif c.islower():
            masked_character = 'x'
        elif c.isupper():
            masked_character = 'X'
        elif c.isspace() or ispunct(c):
            masked_character = c
        masked_text += masked_character
    return masked_text


def main():
    args = sys.argv
    if len(args) != 3:
        print('python3 mask_text.py <input> <output>')
    input_filepath = args[1]
    output_filepath = args[2]

    # Load
    df = pd.read_csv(input_filepath)

    # Mask
    df['text'] = df['text'].apply(mask_text)
    
    # Save
    df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    main()