import gzip
from numpy import *
from sklearn.neighbors import KDTree
from sklearn.base import BaseEstimator, TransformerMixin


class pyMSSG(BaseEstimator, TransformerMixin):

    def __init__(self, embedding_file, window=5):
        self.Vw = []
        self.Vs = []
        self.window = window
        self.word2id = {}
        self.id2word = []
        self.nsense = []
        self.sense2word = []
        self.sense_start = []
        self.mu = []
        openf = gzip.open if embedding_file[-2:] == 'gz' else open
        with openf(embedding_file) as f:
            row = f.next().split()
            self.vocab_size = int(row[0])
            self.embedding_dim = int(row[1])
            self.maxout = int(row[3])

            sense_count = 0
            for i in xrange(self.vocab_size):
                row = f.next().split()
                if len(row)==1:
                    row = ['', row[0]]
                w = row[0]
                self.nsense.append(int(row[1]))
                self.word2id[w] = i
                self.id2word.append(w)

                # Reading global embeddings
                self.Vw.append(self._vector(f))

                # Reading sense embeddings
                self.sense_start.append(sense_count)
                for j in xrange(self.nsense[i]):
                    self.Vs.append(self._vector(f))
                    if self.maxout == 0:
                        self.mu.append(self._vector(f))

                    self.sense2word.append(w)
                    sense_count += 1

        self.Vw = asarray(self.Vw)
        self.Vw /= linalg.norm(self.Vw, axis=1).reshape(-1, 1)
        self.Vs = asarray(self.Vs)
        self.Vs /= linalg.norm(self.Vs, axis=1).reshape(-1, 1)
        if self.maxout == 0:
            self.mu = asarray(self.mu)
            self.mu /= linalg.norm(self.mu, axis=1).reshape(-1, 1)
        self._sense_tree = KDTree(self.Vs)

    def _vector(self, f):
        return [float(x) for x in f.next().split()]

    def sense_rep(self, sense_id, N=10):
        dist, idx = self._sense_tree.query(self.Vs[sense_id], k=N)
        return [self.sense2word[i] for i in idx[0][1:]]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, list)
        result = []
        for x in X:
            assert isinstance(x, list)
            senses = []
            for i in xrange(len(x)):
                context = []
                for w in xrange(-self.window, self.window + 1):
                    if w != 0 and 0 <= i + w < len(x):
                        if x[i + w] in self.word2id:
                            context.append(self.Vw[self.word2id[x[i + w]]])
                if len(context) == 0 or x[i] not in self.word2id:
                    sid = self.sense_start[self.word2id['unknown']]
                else:
                    context = asarray(context).mean(axis=0)
                    context /= linalg.norm(context)
                    sid, min_dist = None, None
                    wid = self.word2id[x[i]]
                    for j in xrange(self.nsense[wid]):
                        dist = linalg.norm(self.Vs[self.sense_start[wid] + j])
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            sid = self.sense_start[wid] + j
                senses.append(sid)
            result.append(senses)
        return result


if __name__ == '__main__':
    print 'Loading the embeddings...'
    m = pyMSSG('../../multi-sense-skipgram/vectors-NP-MSSG.gz')
    str = """
        ministers and vice ministers meets regularly to discuss policy issues
        governors of the one eight provinces are appointed by and serve at the
       pleasure of the president the constitutional law of one nine
    """
    print [[m.sense_rep(i) for i in l]
           for l in m.fit_transform([str.split()])]


