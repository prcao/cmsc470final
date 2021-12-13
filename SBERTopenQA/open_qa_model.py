from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
from document_retriever import DocumentRetriever

class OpenQAModel():

    def __init__(self, use_cuda=False):
        self.reader = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/transformer-qa.2021-02-11.tar.gz")
        self.document_retriever = DocumentRetriever(use_cuda=use_cuda)

    def load(self, document_retriever_folder):
        self.document_retriever.load(document_retriever_folder)

    def read(self, question, contexts):
        if type(contexts) is str:
            contexts = [contexts]
        def read_one(context):
            results = self.reader.predict(passage=context, question=question)
            return (results['best_span_str'], results['best_span_probs'])
        return [read_one(context) for context in contexts]
    
    def predict(self, questions):
        all_doc_ids_to_read = self.document_retriever.predict(questions)
        ret = []
        for sent, docs_to_read in zip(questions, all_doc_ids_to_read):
            docs_to_read = [doc[0] for doc in docs_to_read]
            ret.append(self.read(sent, docs_to_read))
        return ret

if __name__ == '__main__':
    model = OpenQAModel(use_cuda=True)
    model.load('./model/wiki_full_msmarco_distilbert_base_v4')
    # results = model.predict(["What is the tallest mountain?", "who is the president?", "what is the color of the sky?", "who am I?"])
    # print(results)

    while True:
        question = input("Enter query: ")
        results = model.predict([question])
        print(results)