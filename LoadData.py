import numpy as np
def GetData():
    text = open('wiki.test.raw').read()
    text = str(text[0:2001])
    
    chars = sorted(list(set(text)))
    char_size = len(chars)
    char2id = dict((c, i) for i, c in enumerate(chars))
    id2char = dict((i, c) for i, c in enumerate(chars))
    
    Sequence_Length = 50
    skip = 2
    Sections = []
    Next_Char = []
    for i in range(0, len(text)-Sequence_Length, skip):
        Sections.append(text[i:i+Sequence_Length])
        Next_Char.append(text[i+Sequence_Length])
        
    X = np.zeros((len(Sections), Sequence_Length, char_size))
    Y = np.zeros((len(Sections), char_size))
    
    for i, section in enumerate(Sections):
        for j, char in enumerate(section):
            X[i, j, char2id[char]] = 1.0
        Y[i, char2id[Next_Char[i]]] = 1.0
            
    return Sequence_Length, char_size, char2id, id2char, X, Y