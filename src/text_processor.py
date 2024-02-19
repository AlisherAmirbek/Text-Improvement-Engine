from utils import find_most_similar, generate_revised_sentence
from tqdm import tqdm

def analyze_text(file_path, standard_phrases, threshold=0.3):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    suggestions = []

    sentences = [sentence.strip() + '.' for sentence in text.split('.') if sentence.strip()]
    
    for sentence in tqdm(sentences, desc="Analyzing text"):
        most_similar_phrase, similarity = find_most_similar(sentence, standard_phrases)
        if similarity >= threshold:
            revised_sentence = generate_revised_sentence(sentence, most_similar_phrase)
            suggestions.append((sentence, most_similar_phrase, similarity, revised_sentence))
    
    return suggestions

def write_text_to_file(updated_text, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(updated_text)


def process_text_file(input_file_path, output_file_path, standard_phrases, threshold=0.3):
    suggestions = analyze_text(input_file_path, standard_phrases, threshold)

    with open(input_file_path, 'r', encoding='utf-8') as file:
        original_text = file.read()

    updated_text = original_text

    for original, _, _, revised in suggestions:
        updated_text = updated_text.replace(original, revised)

    write_text_to_file(updated_text, output_file_path)

    print(f"Updated text has been written to a new file: {output_file_path}")

    return suggestions
