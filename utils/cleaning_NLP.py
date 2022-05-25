import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text, stopwords = True):
    """_summary_

    Args:
        text (_type_): _description_
        stopwords (bool, optional): _description_. Defaults to True.
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '') 
    #text = re.sub(r'\W+', '', text)
    if stopwords:
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text


def tokenize(lst_of_strings, max_nb_words = 50000, verbose = True):
    """Takes a list of strings and creates a tokenizer object

    Args:
        lst_of_strings (lst): List of strings (usually all of your dataset)
        max_nb_words (int, optional): The maximum number of words to be used. (most frequent). Defaults to 50000.

    Returns:
        tokenizer: object that can further tokenize more examples
    """

    tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(lst_of_strings)
    if verbose:
        print('     Found %s unique tokens.' % len(tokenizer.word_index))
        
    return tokenizer


def tokenize_n_split(essay_lst, Y, max_nb_words, max_sequence_length, test_size = 0.2, 
                     random_state = 1):
    """
    Args:
        essay_lst (lst): List of strings with all of the data
        Y (array): target variables with c columns (where c is the number of classes)
        max_nb_words (int): Max number of words used
        max_sequence_length (int): Length of words used for features
        test_size (float, optional): Share of data devoted to test. Defaults to 0.2.
        random_state (int, optional): Random seed. Defaults to 1.

    Returns:
        arrays: Training and test sets (X and Y)
    """

    tokenizer = tokenize(essay_lst, max_nb_words = max_nb_words, verbose =False)
    X = tokenizer.texts_to_sequences(essay_lst)
    X = pad_sequences(X, maxlen=max_sequence_length)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = test_size, random_state = random_state)

    return X_train, X_test, Y_train, Y_test


def run_lstm(X_train, Y_train, max_nb_words, embedding_dim, 
             dropout, rec_dropout, epochs = 5, 
             batch_size = 64, lstm_units = 100):
    """Runs a Long Short Term Memory Recurrent Neural Network (LSTM RNN)

    Args:
        X_train (array): Feature array
        Y_train (array): Target array
        max_nb_words (int): The maximum number of words to be used. (most frequent). 
        embedding_dim (int): Number of embedding dimensions used
        dropout (float): Dropout rate
        rec_dropout (float): Recurrent dropout rate
        epochs (int, optional): Number of epochs the algorithm will run for. Defaults to 5.
        batch_size (int, optional): Number of samples used in each batch. Defaults to 64.

    Returns:
        Tuple: model, history of epoch metrics
    """
    model = Sequential()
    model.add(Embedding(max_nb_words, embedding_dim, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(dropout))
    model.add(LSTM(lstm_units, dropout=dropout, recurrent_dropout=rec_dropout))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=epochs, 
                        batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', 
                                                 patience=3, min_delta=0.0001)])

    return model, history