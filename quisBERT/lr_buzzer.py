
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from functools import reduce
import json

def generate_runs(text, run_length=200):
    ret = []
    for length in range(run_length, len(text), run_length):
        ret.append(text[:length])
    ret.append(text)
    return ret

def generate_training_data(guesser, train_data_path, out_data_path, run_length=200, top_n_guesses=3, limit=2000):

    negatives = ["not", "n't", "never"]
    progress = 0
    with open(train_data_path, encoding='utf-8') as train_f, open(out_data_path, 'w') as out_f:
        questions = json.load(train_f)['questions']
        out_f.write('run_length,score,num_negatives,num_duplicates,label\n')
        for q in questions[:limit]:

            if progress % 2 == 0:
                print(f'{progress} of {len(questions)}')

            answer = q['page']

            runs = generate_runs(q['text'], run_length=run_length)

            all_guesses = guesser.guess(runs)
                
            for i in range(len(runs)):
                run = runs[i]
                guesses = all_guesses[i]
                only_guesses = [guess[0] for guess in guesses]

                num_negatives = reduce(lambda acc, x: acc + run.count(x), negatives, 0)

                for guess, score in guesses[:top_n_guesses]:
                    label = 1 if answer == guess else 0
                    num_duplicates = only_guesses.count(guess)
                    out_f.write(f'{len(run)},{score},{num_negatives},{num_duplicates},{label}\n')

            progress += 1

class LogisticRegressionBuzzer():

    def train(self):
        df = pd.read_csv('../data/final_buzzer_train_large.csv')
        df.head()

        x,y = df.drop(columns=['label']), df['label']
        self.lr = LogisticRegression()
        self.lr.fit(x, y)

    def predict(self, X):
        return self.lr.predict_proba(X)[:,1]

    def save(self):
        with open('./lr_buzzer.pkl', 'wb') as f:
            pickle.dump(self.lr, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.lr = pickle.load(f)

if __name__ == '__main__':

    buzzer = LogisticRegressionBuzzer()
    # buzzer.train()
    # buzzer.save()
    buzzer.load('./lr_buzzer.pkl')
    probs = buzzer.predict([[200,1,.5,0],[600,.1,.5,4],[500,.05,.3,4],[500,1,.3,4]])

    import numpy as np
    print(probs)
    print(np.argmax(probs))
    print(probs[np.argmax(probs)])

