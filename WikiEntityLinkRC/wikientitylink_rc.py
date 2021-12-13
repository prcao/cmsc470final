from typing import Any, List
# from entity_linking.REL import rel_wrapper
import codecs
import wikipediaapi
import time
import requests
import logging

wiki = wikipediaapi.Wikipedia('en')

API_URL = "https://rel.cs.ru.nl/api"


def entity_link(text):

    if text is None:
        return []

    res = requests.post(API_URL, json={
            "text": text,
            "spans": []
        }).json()

    res = [(x[3]) for x in res]
    return res

def get_wiki_text(wiki_title):
    page = wiki.page(wiki_title)

    if page.exists():
        return page.text
    else:
        return None

# Reading Comprehension Model
class AllenNLPRoBERTa():
    def __init__(self):
        from allennlp.predictors.predictor import Predictor
        import allennlp_models.rc
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/transformer-qa.2021-02-11.tar.gz")
    
    def predict(self, question:str, context:str) -> str:
        try:
            results = self.predictor.predict(question, context)
            if results['best_span'][0] == -1:
                return None
            # return results
            return results['best_span_str'], results['best_span_probs'], results['best_span_scores']
        except:
            return None

# Wikipedia text splitter
filtered_sections = set(['References', 'Sources', 'Further reading', 'External links', 'See also'])

def get_text_by_section(section, out:List[str]) -> None:
    if section.title in filtered_sections:
        return

    if len(section.sections) == 0:
        out.append(section.text)
    else:
        for s in section.sections:
            get_text_by_section(s, out)

def flatten(t:List[List[Any]]) -> List[Any]:
    return [item for sublist in t for item in sublist]
    
def split_article(wiki_title:str, max_tokens:int=-1) -> List[str]:
    page = wiki.page(wiki_title)

    # print(page.sections)
    all_text = []
    get_text_by_section(page, all_text)
    
    # This does not actually guarantee text is split into less than max_tokens lol
    if max_tokens > 0:
        def split_long_texts(text):
            if len(text.split()) > max_tokens:
                return text.split('\n')
            return [text]
        all_text = flatten([split_long_texts(text) for text in all_text])

    return all_text

if __name__ == '__main__':

    logging.getLogger('allennlp.common.params').disabled = True 
    logging.getLogger('allennlp.nn.initializers').disabled = True 
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO) 
    logging.getLogger('urllib3.connectionpool').disabled = True 

    prompt = """
    Robert Walker argued that failing to take this action would lead to an overflow of Northern insane asylums and British intervention.
    Juan Almonte resigned his diplomatic post in indignation over this event. 
    Isaac Van Zandt began discussing this plan with Abel Upshur before Upshur died. 
    Anson Jones proposed this plan, which reduced his own power but reserved the weaker side the right to split
    into five parts in the future. Five years after it, a payment of 10 million dollars helped the area at issue repay debts. 
    The Regulator-Moderator war was calmed just before this deal was struck. 
    The question of whether the Nueces River became a southern border in this transaction led to war later in James K Polk's presidency.
    For 10 points, name this deal that ended the independent Lone Star Republic.
    """


    sentences = prompt.split(".")
    model = AllenNLPRoBERTa()

    for sent in sentences:
        entities = entity_link(sent)
        print(entities)
        for e in entities:
            sections = split_article(e)
            for section in sections:
                prediction = model.predict(sent, section)
                if prediction:
                    data = {
                        # 'label': question['page'],
                        'model_output': prediction[0],
                        'model_output_probs': prediction[1],
                        'model_output_score': prediction[2],
                    }
                    print(data)
