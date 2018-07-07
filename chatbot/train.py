
def generate_batch(tokenized_from, tokenized_to, word2index, preprocessing_params, hparams):
    n_samples = len(tokenized_from)
    # must have same amount of comments/answers (from/to)
    assert n_samples == len(tokenized_to)
    num_batches = n_samples // hparams['batch_size']
    while True:
        for batch_i in range(0, num_batches):
            start = batch_i * hparams['batch_size']
            end = (batch_i + 1) * hparams['batch_size']
            yield create_model_input_output(
                tokenized_from[start: end], tokenized_to[start: end], word2index, preprocessing_params)

def create_model_input_output(tokenized_from, tokenized_to, word2index, preprocessing_params):
    # TODO throw away samples with length > max_seq_length
    unk_index, eos_index, bos_index = [word2index[w] for w in [preprocessing_params['unk'],
                                                               preprocessing_params['eos'],
                                                               preprocessing_params['bos']]]
    n_samples = len(tokenized_from)
    max_seq_length = preprocessing_params['max_seq_length']
    encoder_input = np.zeros((n_samples, max_seq_length), dtype=int)
    decoder_input = np.zeros((n_samples, max_seq_length), dtype=int)
    decoder_output = np.zeros((n_samples, max_seq_length), dtype=int)
    decoder_output_oh = np.zeros((n_samples, max_seq_length, preprocessing_params['full_vocab_size']), dtype=float)
    j = 0

    for i, sentence_from, sentence_to in zip(range(n_samples), tokenized_from, tokenized_to):
        if i % 10000 == 0:
            print(f"Creating input/output for model {i}/{n_samples} done.")
        j += 1
        if j == 10000:
            break
        encoder_input[i, :len(sentence_from)] = [
            word2index.get(w, unk_index) for w in sentence_from[:max_seq_length]]
        sentence_to_indexed = [word2index.get(w, unk_index) for w in sentence_to[:max_seq_length - 1]]
        decoder_input[i, :len(sentence_to) + 1] = [bos_index] + sentence_to_indexed
        decoder_output[i, :len(sentence_to) + 1] = sentence_to_indexed + [eos_index]
        decoder_output_oh[i, np.arange(max_seq_length), decoder_output[i]] = 1
    return [encoder_input, decoder_input], decoder_output_oh



if __name__ == '__main__':
    # read files with samples
    path = os.path.join(DATA_DIR, "tokenized_from.pickle")
    tokenized_from = pickle.load(open(path, "rb"))
    path = os.path.join(DATA_DIR, "tokenized_to.pickle")
    tokenized_to = pickle.load(open(path, "rb"))