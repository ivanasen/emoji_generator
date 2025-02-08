import re
import torch
import nltk
import pickle
import pandas as pd
import string
from nltk.tokenize import WordPunctTokenizer

# from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt_tab")
stop_words = set(stopwords.words("english"))

UNK_TOKEN = "<UNK>"
WORD_TO_IDX_FILE = "word_to_idx"
MODEL_PATH = "skip_gram_2025-02-05_00_23_07"
THRESHOLD = 0.25
embeddings_dim = 300
context_window = 3


class SkipGramLanguageModel(torch.nn.Module):

    def __init__(self, word_to_idx, embedding_dim, context_size):
        super(SkipGramLanguageModel, self).__init__()
        self.word_to_idx = word_to_idx
        vocab_size = len(word_to_idx)
        self.idx_to_word = {v: k for k, v in word_to_idx.items()}
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        self.context_size = context_size

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        logits = self.linear(embeds)
        return logits

    def find_closest_embeddings(
        self, word: str | torch.Tensor, vocab_dict: dict[str, int], n_closest: int = 5
    ) -> list[tuple[str, float]]:

        if isinstance(word, str):
            try:
                word_idx = vocab_dict[word]
            except ValueError:
                raise ValueError(f"Word '{word}' not found in vocabulary")

            word_embedding = self.embeddings.weight[word_idx]
        else:
            word_embedding = word

        similarities = torch.nn.functional.cosine_similarity(
            word_embedding.unsqueeze(0), self.embeddings.weight, dim=1
        )

        top_indices = torch.argsort(similarities, descending=True)[1 : n_closest + 1]

        closest_words = [
            (self.idx_to_word[idx.item()], similarities[idx].item())
            for idx in top_indices
        ]

        return closest_words

    def sent_to_emojis(
        self, sent: str, emoji_embeddings: dict[str, torch.Tensor], stop_words: set[str]
    ) -> str:
        result = []
        tokenizer = WordPunctTokenizer()
        for original_token in tokenizer.tokenize(sent):
            token = original_token.lower()
            if (
                token in stop_words
                or token in string.punctuation
                or token not in self.word_to_idx
            ):
                result.append(original_token)
            else:
                word_embeddings = self.embeddings.weight[self.word_to_idx[token]]
                if emoji := self.find_closest_emoji(word_embeddings, emoji_embeddings):
                    result.append(emoji[0])
                else:
                    result.append(original_token)
        return " ".join(word for word in result).strip()

    def find_closest_emoji(
        self, word_embedding: torch.Tensor, emoji_embeddings: dict[str, torch.Tensor]
    ) -> str:
        emoji_to_idx = {
            i: emoji for i, (emoji, _) in enumerate(emoji_embeddings.items())
        }

        # most_similar = None
        similarities = torch.nn.functional.cosine_similarity(
            word_embedding.unsqueeze(0),
            torch.stack(list(emoji_embeddings.values())),
            dim=1,
        )

        closest_emojis = torch.argsort(similarities, descending=True)

        # print(f"{[(emoji_to_idx[idx.item()], similarities[idx.item()].item()) for idx in closest_emojis[:5]]}")

        if similarities[closest_emojis[0]].item() < THRESHOLD:
            return None

        return emoji_to_idx[closest_emojis[0].item()]

    def embed_phrase(
        self, phrase: str, stop_words: set[str] | None = None
    ) -> torch.Tensor | None:
        tokenizer = WordPunctTokenizer()
        # skip the stop words, they don't bring semantics
        meaningful_tokens = [
            word
            for word in tokenizer.tokenize(phrase.lower())
            if word not in string.punctuation and word in self.word_to_idx
        ]
        if stop_words:
            meaningful_tokens = [
                word for word in meaningful_tokens if word not in stop_words
            ]
        # If no meaningful words or a word is not in the dictionary
        if not meaningful_tokens:
            return None

        # We are not normalizing as in the cosine similarity the length doesn't matter
        return sum(
            self.embeddings.weight[self.word_to_idx[word]] for word in meaningful_tokens
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_to_idx = pickle.load(open(WORD_TO_IDX_FILE, "rb"))
emojis_dataset = pd.read_csv("./emojis.csv")

model = SkipGramLanguageModel(word_to_idx, embeddings_dim, context_window)
state_dict = torch.load(MODEL_PATH, device)
model.load_state_dict(state_dict)


class SkipGramModel:
    def __init__(self, model: SkipGramLanguageModel, emojis_dataset: pd.DataFrame):
        self.model = model
        self.stop_words = stop_words

        emojis = dict(zip(emojis_dataset["Name"], emojis_dataset["Representation"]))

        self.emoji_embeddings = {
            (emojis[name], name): model.embed_phrase(name)
            for name in emojis.keys()
            if model.embed_phrase(name) is not None
        }

        # print(emojis["jack-o-lantern"])

    def sent_to_emojis(self, sent: str) -> str:
        return self.model.sent_to_emojis(sent, self.emoji_embeddings, self.stop_words)

    __call__ = sent_to_emojis


# print(model.find_closest_embeddings("love", word_to_idx, n_closest=10))

# hot_beverage = model.embed_phrase("hot beverage", stop_words)

# print(model.find_closest_embeddings("hot", word_to_idx, n_closest=10))
# print()

# print(model.find_closest_embeddings("beverage", word_to_idx, n_closest=10))
# print()

# print(model.find_closest_embeddings("coffee", word_to_idx, n_closest=10))
# print()

# print(model.find_closest_embeddings(hot_beverage, word_to_idx, n_closest=10))
# print()

# # No embeddings for bouquet
# print(f"{word_to_idx["bouquet"]}")


# skip_gram_model = SkipGramModel(model, emojis_dataset)

# sents = [
#     "coffee",
#     "chicken lays eggs bouquet",
#     "Flexin' in a bikini on national television- Things I never would have imagined for 500, Alex -- Did…",
#     "This is a test sentence, grab a coffee with chicken",
#     "extra cheese crispy chicken strips",
#     "avocado",
# ]

# for sent in sents:
#     print(f"{sent} ->\n{skip_gram_model(sent)}\n")
