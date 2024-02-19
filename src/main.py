import argparse
from utils import load_standard_phrases, find_most_similar, generate_revised_sentence
from tqdm import tqdm

def analyze_text(file_path, standard_phrases, threshold=0.3):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    suggestions = []

    sentences = [sentence.strip() for sentence in text.split('.') if sentence]
    
    for sentence in tqdm(sentences, desc="Analyzing text"):
        most_similar_phrase, similarity = find_most_similar(sentence, standard_phrases)
        if similarity >= threshold:
            revised_sentence = generate_revised_sentence(sentence, most_similar_phrase)
            suggestions.append((sentence, most_similar_phrase, similarity, revised_sentence))
    
    return suggestions


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
        suggestions = analyze_text(args.text, standard_phrases)
        for original, phrase, score, revised in suggestions:
            print(f"Original: {original} | Phrase: {phrase} | Similarity: {score:.2f} | Revised sentencce: {revised}\n")
    else:
        print("No text provided to analyze.")

    

if __name__ == "__main__":
    main()
