# Text Improvement Tool

This tool analyzes text to identify sentences that could be improved and suggests revisions to enhance clarity, coherence, and overall impact. It leverages advanced natural language processing (NLP) techniques to generate suggestions that align closely with the context and intent of the original text.

## Setup Process

### Prerequisites

- Python 3.10 or later
- pip (Python package installer)

### Installation

1. **Clone the Repository**

```bash
git clone https://yourrepositorylink.git
cd Text-Improvement-Tool
```

2. **Create a Virtual Environment**

```bash
python -m venv .venv
```

Activate the virtual environment:

- On Windows: `.venv\Scripts\activate`
- On macOS and Linux: `source .venv/bin/activate`

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### Running the Tool
```bash
python src/main.py --input path/to/input.txt --output path/to/phrases.csv
```

Replace `path/to/input.txt` with the path to your text file and `path/to/phrases.csv` with the path for the suggested phrases.

## Technologies Used

- **Sentence-BERT (SBERT)**: Adjusts the original BERT model to better understand and compare whole sentences quickly and accurately.
- **Phi-2 Transformer**: A very advanced AI model with 2.7 billion settings (parameters) for understanding and creating text.

## Rationale Behind Design Decisions

- **Why SBERT?** SBERT was used because it's fast and good at figuring out what sentences mean, helping to spot where improvements are needed without getting slowed down.
- **Why Phi-2?** Phi-2's huge number of parameters means it's excellent at handling complex language tasks. It can rewrite sentences to include suggested improvements smoothly, making sure the new sentence makes sense and stays true to the original meaning.
- **Combining SBERT and Phi-2**: First, SBERT helps us find the text parts that need work. Then, Phi-2 steps in to rewrite these parts cleverly and coherently. It's a one-two punch for making text better.

## Conclusion

Using SBERT and Phi-2 gives us a powerful toolkit for enhancing text. SBERT points out where improvements are needed, and Phi-2 makes those improvements in a way that's smart and keeps the original vibe of the text.


