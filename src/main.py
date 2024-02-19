import argparse
from utils import load_standard_phrases
from text_processor import process_text_file

def main():
    parser = argparse.ArgumentParser(description="Text Analysis Tool")
    parser.add_argument('--text', type=str, help="Text to analyze")
    parser.add_argument('--phrases', type=str, help="Path to file containing standard phrases")
    args = parser.parse_args()

    if args.phrases:
        standard_phrases = load_standard_phrases(args.phrases)
    else:
        print("No file provided for standard phrases.")
        return

    if args.text:
        output_file_path = 'data/revised_text'
        suggestions = process_text_file(args.text, output_file_path, standard_phrases)
        for original, phrase, score, revised in suggestions:
            print(f"Original: {original} | Phrase: {phrase} | Similarity: {score:.2f} | Revised sentence: {revised}\n")
    else:
        print("No text provided to analyze.")

    

if __name__ == "__main__":
    main()
