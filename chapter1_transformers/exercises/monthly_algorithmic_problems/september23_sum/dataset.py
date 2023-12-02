import itertools
import numpy as np
import torch as t
from torch.utils.data import Dataset

class Pairs :

    def __init__(self, digits=4 ):
        self.digits = digits
        self.cases  = [ [ [a,b] for a in range(10) for b in range(10) if (a+b<9) ],
                        [ [a,b] for a in range(10) for b in range(10) if (a+b==9)],
                        [ [a,b] for a in range(10) for b in range(10) if (a+b>9) ]  ]

        values = [ i for i in range(3)]
        p      = np.array( list(itertools.product(values, repeat=digits-1))) 
        zeros  = np.zeros( (p.shape[0],1), dtype=int)
        Pairs.p = np.concatenate((zeros,p), axis=1)

    def generate_one(self) :

        digits = lambda case : self.cases[case][np.random.randint(0,len(self.cases[case]))]
        pair   = lambda case : [ digits( self.p[case][d] ) for d in range(self.digits) ] 
        toint  = lambda a    : int(''.join(map(str, a)))   
        def to_intlist( a ) :
            padded_string = str(a).zfill(4)
            return [ int(digit) for digit in padded_string]

        case = np.random.randint(0,self.p.shape[0])

        p = pair(case)
        a, b = zip(*[(value[0], value[1]) for value in p])
        c = to_intlist( toint(a) + toint(b) )
        return a,b,c,case

    @staticmethod
    def is_carry( ps, digit) :
        c = Pairs.p[ ps ]
        if ( c[digit] == 2 ) :
            return 1
        elif ( digit+1 <= 3  and c[digit] == 1 and c[digit+1] == 2 ) :
            return 1
        elif ( digit+2 <= 3  and c[digit] == 1 and c[digit+1] == 1  and c[digit+2] == 2 ) :
            return 1
        elif ( digit+3 <= 3  and c[digit] == 1 and c[digit+1] == 1  and c[digit+2] == 1  and c[digit+3] == 2 ) :
            return 1
        return 0

    

class SumDataset(Dataset):

    def __init__(self, size: int, num_digits: int,seed=42):

        np.random.seed(seed)
        t.manual_seed(seed)

        self.pairs = Pairs(digits=num_digits)

        self.english = [ "ST", "L1000", "L100 ", "L10  ", "L1   ", "+", "R1000", "R100 ", "R10  ", "R1   ", "=", "S1000", "S100 ", "S10  ", "S1   "]

        self.vocab       = [str(i) for i in range(10)] + [" + ", " = ", "ST "]
        self.vocab_index = { self.vocab[i] : i for i in range(len(self.vocab)) }
        self.size = size
        self.num_digits = num_digits

        def generate_one() :
            a,b,c,p = self.pairs.generate_one()
            return  [self.vocab_index["ST "]] + list(a) + [self.vocab_index[" + "]] + list(b) + [self.vocab_index[" = "]] + list(c) , p

        toks, p = zip(*[generate_one() for _ in range(size)])

        self.toks = t.tensor(toks)
        self.p    = t.tensor(p)

    def __getitem__(self, index):
        return self.toks[index]

    def __len__(self):
        return self.size

    def to(self, device: str):
        self.toks = self.toks.to(device)
        return self

    def format(self , index) :
        toks = self.toks[index]
        return "".join( [self.vocab[tok] for tok in toks])
    
    def format_word(self, word) :
        return self.english[word]

