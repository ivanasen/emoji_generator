
class StemModel(object):

    def __init__(self, emojis_dataset):
        self.emojis_dataset = emojis_dataset

    __call__ = call

    def call(self, t: str) -> str:
        tokens = word_tokenize(sent)

        result = ""

        for token in tokens:
          processed = pre_process(token)

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
            result += choose_shortest_len(matches)[0]
          else:
            result += token
          result += " "

        return result




def pre_process(sentence: str) -> list[str]:
  ps = PorterStemmer()
  clean_sent = re.sub(r'[^\w\s]', '', sentence)
  tokens = word_tokenize(clean_sent)
  stemmed_sent = [ps.stem(t) for t in tokens]
  tokens = [t for t in stemmed_sent if t not in stop_words]
  return tokens

def choose_shortest_len(matches: list[tuple[str, list[str]]]) -> str:
  return sorted(matches, key=lambda x: len(x[1]))[0]

def stem_model(sent: str) -> str:
  tokens = word_tokenize(sent)

  result = ""

  for token in tokens:
    processed = pre_process(token)

    if len(processed) == 0: # It is stop-word
      result += token + " "
      continue

    processed = processed[0]

    matches = []

    for _, emoji in emojis_dataset.iterrows():
      #for emoji_token in emoji.Stemmed:
      #  dist = edit_distance(processed, emoji_token)
      #  if dist < 2:
      #    matches.append((emoji.Representation, emoji.Stemmed))
      if processed in emoji.Stemmed:
        matches.append((emoji.Representation, emoji.Stemmed))

    if len(matches) > 0:
      # result += matches[random.randint(0, len(matches) - 1)][0]
      # result += choose_random_weighted(matches)[0]
      result += choose_shortest_len(matches)[0]
    else:
      result += token
    result += " "

  return result