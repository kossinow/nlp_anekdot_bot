import pandas as pd
import pymorphy2
import re
import telebot

from gensim import corpora, models, similarities
from nltk.corpus import stopwords

print('import ok')

# initialize data
anekdot_df = pd.read_csv('anekdot.csv')
morph = pymorphy2.MorphAnalyzer()
stops = set(stopwords.words("russian"))

print('data ok')

def review_to_wordlist(review):
    # 1) удаляем символы кроме букв
    review_text = re.sub("[^а-яА-Яa-zA-Z]", " ", review)
    # 2) переводим в нижний регистр
    words = review_text.lower().split()
    # 3) убираем стоп слова
    words = [w for w in words if not w in stops]
    # 4) лемматизируем
    words = [morph.parse(w)[0].normal_form for w in words]
    return(words)

print('fun review_to_wordlist ok')
# create corpora, vectors(bow)
texts = anekdot_df.anekdot.values
texts = [review_to_wordlist(text) for text in texts]
dictionary = corpora.Dictionary(texts)
feature_cnt = len(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus) 

print('create corpora ok')

# function which process request phrase, then makes similaruty vector and gives most corresponding phrase from corpora
def give_anekdot(keyword):
    kw_vector = dictionary.doc2bow(review_to_wordlist(keyword))
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    sim = index[tfidf[kw_vector]]

    requested_top = pd.DataFrame(sim, columns=['similarity'])
    sorted_top = requested_top.sort_values('similarity', ascending=False).index.values
    return anekdot_df.anekdot[sorted_top[0]]


print('fun give_anekdot ok')

bot = telebot.TeleBot('bot_token')

print('bot ok')

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "Привет":
        bot.send_message(message.from_user.id,
                         "Привет, чем я могу тебе помочь?")
    else:
        bot.send_message(message.from_user.id, give_anekdot(message.text))
        
print('message ok')

bot.polling(none_stop=True, interval=0)
