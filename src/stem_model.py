from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
import re

class StemModel(object):
    def __init__(self, emojis_dataset, stop_words):
        self.stop_words = stop_words
        self.tokenizer = WordPunctTokenizer()

        self.emojis_dataset = emojis_dataset[["Representation", "Name"]]
        self.emojis_dataset["Stemmed"] = [self.pre_process(name) for name in emojis_dataset.Name]        



    def pre_process(self, sentence: str) -> list[str]:
      tokens = self.tokenizer.tokenize(sentence)
      stemmer = PorterStemmer()
      stemmed_sent = [stemmer.stem(t) for t in tokens]
      tokens = [t for t in stemmed_sent if t not in self.stop_words]
      return tokens

    def call(self, t: str) -> str:
        tokens = self.tokenizer.tokenize(t)

        result = ""

        for token in tokens:
          processed = self.pre_process(token)

          if len(processed) == 0: # It is stop-word
            result += token + " "
            continue

          processed = processed[0]

          matches = []

          for _, emoji in self.emojis_dataset.iterrows():
            #for emoji_token in emoji.Stemmed:
            #  dist = edit_distance(processed, emoji_token)
            #  if dist < 2:
            #    matches.append((emoji.Representation, emoji.Stemmed))
            if processed in emoji.Stemmed:
              matches.append((emoji.Representation, emoji.Stemmed))

          if len(matches) > 0:
            # result += matches[random.randint(0, len(matches) - 1)][0]
            # result += choose_random_weighted(matches)[0]
            result += self.choose_shortest_len(matches)[0]
          else:
            result += token
          result += " "

        return result

    __call__ = call

    def choose_shortest_len(self, matches: list[tuple[str, list[str]]]) -> str:
      return sorted(matches, key=lambda x: len(x[1]))[0]
