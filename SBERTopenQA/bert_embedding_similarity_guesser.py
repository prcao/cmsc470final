from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
import pickle
from typing import List, Tuple
from functools import reduce
from lr_buzzer import LogisticRegressionBuzzer

class EmbeddingGuesser():
    
    def __init__(self, model_type='all-mpnet-base-v2', use_cuda=False):
        device = 'cuda' if use_cuda else 'cpu'
        self.model = SentenceTransformer(model_type, device=device)
        
    def save(self, folder):
        embed_matrix_fname = folder + '/embed.npy'
        answer_list_fname = folder + '/pages.pkl'
        
        np.save(embed_matrix_fname, self.embed_matrix)
        with open(answer_list_fname, 'wb') as f:
            pickle.dump(answer_list_fname, f)
            
    def load(self, folder):
        embed_matrix_fname = folder + '/embed.npy'
        answer_list_fname = folder + '/pages.pkl'
        
        self.embed_matrix = np.load(embed_matrix_fname)
        with open(answer_list_fname, 'rb') as f:
            self.answers = pickle.load(f)
        
    def train(self, path_to_train_data='../data/qanta_train.json'):
        
        with open(path_to_train_data, encoding='utf-8') as in_f:
            qs = json.load(in_f)['questions']
        
        questions = [x['text'] for x in qs]
        answers = [x['page'] for x in qs]
        
        self.embed_matrix = self.model.encode(questions)
        self.questions = questions
        self.answers = answers
    
    def guess(self, questions, max_n_guesses=10):
        question_embeddings = self.model.encode(questions)
        top_results = util.semantic_search(question_embeddings, self.embed_matrix, top_k=max_n_guesses)
        
        ret = []
        for question_results in top_results:
            ret.append([(self.answers[guess['corpus_id']], guess['score']) for guess in question_results])
        return ret

class QuizBowlModel:

    def __init__(self, use_cuda=False, prob_threshold=.7):
        self.model = EmbeddingGuesser(use_cuda=use_cuda)
        self.buzzer = LogisticRegressionBuzzer()
        self.model.load('./qanta_train_all_mpnet_base_v2')
        self.buzzer.load('./lr_buzzer.pkl')
        self.threshold = prob_threshold

    def guess_and_score(self, question_text: List[str]) -> List[Tuple[str, float]]:
        negatives = ["not", "n't", "never"]
        answers = []
        guesses_for_all_questions = self.model.guess(question_text)

        for i, guesses_for_question in enumerate(guesses_for_all_questions):
            question = question_text[i]

            q_len = len(question)
            only_guesses = [guess[0] for guess in guesses_for_question]
            num_negatives = reduce(lambda acc, x: acc + question.count(x), negatives, 0)

            all_guesses_features_for_q = []
            for guess, score in guesses_for_question:
                num_duplicates = only_guesses.count(guess)
                features = [q_len, score, num_negatives, num_duplicates]
                all_guesses_features_for_q.append(features)

            probs = self.buzzer.predict(all_guesses_features_for_q)
            best_guess = only_guesses[np.argmax(probs)]
            best_prob = probs[np.argmax(probs)]

            answers.append((best_guess, best_prob))  
            
        return answers

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        guess_and_scores = self.guess_and_score(question_text)
        return [(guess, score > self.threshold) for guess, score in guess_and_scores]

if __name__ == '__main__':
    from lr_buzzer import generate_runs
    model = QuizBowlModel(use_cuda=False, prob_threshold=.75)
    q = "Calculating a Racah W-coefficient requires knowledge of six parameters corresponding to this quantity. Another set of coefficients arising from this quantity relate reduced matrix elements to the spherical tensor. The individual components of the operator corresponding to this quantity commute with its square, but not with each other. That square of the operator corresponding to this quantity has as its eigenfunctions the spherical harmonics. This quantity's conservation follows by Noether's theorem from rotational invariance, and implies Kepler's second law. In the case of a particle undergoing uniform circular motion, this quantity's magnitude equals m times v times r. For 10 points, name this quantity, equal to moment of inertia times angular velocity"
    
    print(model.guess_and_score(generate_runs(q)))
