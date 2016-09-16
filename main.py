import collections

import pandas as pd

from nltk.corpus import wordnet as wn


class ExtSynset(object):
    def __init__(self, synset):
        self.synset = wn.synset(synset)

    @staticmethod
    def flatten(l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(
                    el, basestring):
                for sub in ExtSynset.flatten(el):
                    yield sub
            else:
                yield el

    @staticmethod
    def hyp(s):
        return s.hypernyms()

    def get_hypernyms(self):
        hyps = self.synset.tree(self.hyp)
        return list(set([x.name() for x in self.flatten(hyps)]))

    def get_meronyms(self):
        return [x.name() for x in self.synset.part_meronyms()]

    def get_holonyms(self):
        return [x.name() for x in self.synset.member_holonyms()]

    def get_features(self, types=['hypernyms', 'meronyms', 'holonyms']):
        features = []
        if 'hypernyms' in types:
            features += [(self.synset.name(), x, 'hypernym')
                         for x in self.get_hypernyms()]
        if 'meronyms' in types:
            features += [(self.synset.name(), x, 'meronym')
                         for x in self.get_meronyms()]
        if 'holonyms' in types:
            features += [(self.synset.name(), x, 'holonym')
                         for x in self.get_holonyms()]
        return pd.DataFrame(features, columns=['synset', 'feature', 'type'])

if __name__ == '__main__':
    ss = wn.all_synsets()
    ss_n = [x for x in ss if x.pos() == 'n']

    feature_list = pd.concat([s.get_features() for s in ss_n])
    feature_list.to_csv('wordnet_feature_list_nouns.csv')
