from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
import pickle
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
from nltk import sent_tokenize

class DocumentRetriever():
    def __init__(self, biencoder='msmarco-distilbert-base-v4', crossencoder='cross-encoder/ms-marco-MiniLM-L-6-v2', use_cuda=False):
        
        device = 'cuda' if use_cuda else 'cpu'
        
        self.biencoder = SentenceTransformer(biencoder, device=device)
        self.crossencoder = CrossEncoder(crossencoder, device=device)
    
    def load(self, folder):
        self.corpus_embeddings = np.load(folder + '/embed.npy')
        with open(folder + '/corpus.pkl', 'rb') as f:
            self.corpus = pickle.load(f)
        with open(folder + '/corpus_titles.pkl', 'rb') as f:
            self.corpus_titles = pickle.load(f)
        
    def predict_titles(self, queries, max_n_guesses=5):
        
        question_embeddings = self.biencoder.encode(queries)
        hits_for_all_questions = util.semantic_search(question_embeddings, self.corpus_embeddings, top_k=50)
        hits_per_query = []

        for query, hits in zip(queries, hits_for_all_questions):

            cross_inp = [[query, self.corpus[hit['corpus_id']]] for hit in hits]
            cross_scores = self.crossencoder.predict(cross_inp)

            # Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]

            hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
            hits_per_query.append(hits[:max_n_guesses])

        def search_results_to_doc_titles(results):
            return [(self.corpus_titles[result['corpus_id']], result['score']) for result in results]
    
        return [search_results_to_doc_titles(results) for results in hits_per_query]

    def predict(self, queries, max_n_guesses=5):
        
        question_embeddings = self.biencoder.encode(queries)
        hits_for_all_questions = util.semantic_search(question_embeddings, self.corpus_embeddings, top_k=50)
        hits_per_query = []

        for query, hits in zip(queries, hits_for_all_questions):

            cross_inp = [[query, self.corpus[hit['corpus_id']]] for hit in hits]
            cross_scores = self.crossencoder.predict(cross_inp)

            # Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]

            hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
            hits_per_query.append(hits[:max_n_guesses])

        def search_results_to_docs(results):
            return [(self.corpus[result['corpus_id']], result['score']) for result in results]
    
        return [search_results_to_docs(results) for results in hits_per_query]

if __name__ == '__main__':
    retriever = DocumentRetriever()
    retriever.load('../notebooks/wiki_full_msmarco_distilbert_base_v4')

    results = retriever.predict("This deal that ended the independent Lone Star Republic.")
    print(results)