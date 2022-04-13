import re

import pandas as pd
import jpype as jp
from tqdm import tqdm
import gensim.parsing.preprocessing as gsp
import string
import nltk
import ssl


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

from typing import List

from pandas import ExcelWriter
from pandas import ExcelFile
corpus_df = pd.read_excel("/Volumes/ExtremeSSD/bitirmep/corpus.xlsx", usecols=['kategori', 'icerik'])



### START ZEMBEREK
#
# Requirements: zemberek-full.jar, lm.2gram.slm and normalization folder contains ascii-map, lookup-from-graph and split files
# visit https://www.kaggle.com/egebasturk1/yemeksepeti-sentiment-analysis/data

ZEMBEREK_PATH = r'/Users/oytunakdeniz/PycharmProjects/process0/Zemberek-Python-Examples/bin/zemberek-full.jar'

jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))

TurkishMorphology = jp.JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()


#TurkishSpellChecker = jp.JClass('zemberek.normalization.TurkishSpellChecker')
TurkishSentenceNormalizer = jp.JClass('zemberek.normalization.TurkishSentenceNormalizer')
Paths = jp.JClass('java.nio.file.Paths')

lookupRoot = Paths.get(r'/Users/oytunakdeniz/PycharmProjects/process0/Zemberek-Python-Examples/data/normalization')
lmPath = Paths.get(r'/Users/oytunakdeniz/PycharmProjects/process0/Zemberek-Python-Examples/data/lm/lm.2gram.slm')
morphology = TurkishMorphology.createWithDefaults()
normalizer = TurkishSentenceNormalizer(morphology, lookupRoot, lmPath)


tqdm.pandas()

def normalization(tweet):
    return " " if len(tweet) == 0 or tweet.isspace() else str(normalizer.normalize(jp.JString(tweet)))

corpus_df['processed1'] = corpus_df['icerik'].progress_apply(normalization)
corpus_df['processed1'].to_csv('processed1normalized.csv', header=False, index=False)



corpus_df['processed2'] = corpus_df['processed1'].apply(lambda x: x.translate(str.maketrans(dict.fromkeys(string.punctuation, ' '))))
corpus_df['processed2'].to_csv('processed2withoutpunc.csv', header=False, index=False)



corpus_df['processed3'] = corpus_df['processed2'].apply(gsp.strip_multiple_whitespaces)
corpus_df['processed3'].to_csv('processed3withoutspaces.csv', header=False, index=False)





corpus_df['processed4'] = corpus_df['processed3'].apply(gsp.strip_numeric)
corpus_df['processed4'].to_csv('processed4withoutnums.csv', header=False, index=False)




def correct_old_characters(self):
    self = re.sub(r"Â", "A", self)
    self = re.sub(r"Î", "I", self)
    self = re.sub(r"î", "ı", self)
    self = re.sub(r"â", "a", self)
    self = re.sub(r"û", "u", self)
    self = re.sub(r"Û", "U", self) # for the rest use default lower
    return self

corpus_df['processed5'] = corpus_df['processed4'].apply(lambda x: correct_old_characters(x))
corpus_df['processed5'].to_csv('processed5withoutaccented.csv', header=False, index=False)


def lower(self):
    self = re.sub(r"İ", "i", self)
    self = re.sub(r"I", "ı", self)
    self = re.sub(r"Ç", "ç", self)
    self = re.sub(r"Ş", "ş", self)
    self = re.sub(r"Ü", "ü", self)
    self = re.sub(r"Ğ", "ğ", self)
    self = self.lower() # for the rest use default lower
    return self

corpus_df['processed6'] = corpus_df['processed5'].apply(lambda x: lower(x))
corpus_df['processed6'].to_csv('processed6lower.csv', header=False, index=False)


whitelist = set('abcçdefgğhıijklmnoöpqrsştuüvwxyz ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ')

corpus_df['processed7'] = corpus_df['processed6'].apply(lambda x: ''.join(filter(whitelist.__contains__, x)))
corpus_df['processed7'].to_csv('processed7withoutnonturkish.csv', header=False, index=False)


corpus_df['processed8'] = corpus_df['processed7'].apply(gsp.strip_short)
corpus_df['processed8'].to_csv('processed8withoutshorts.csv', header=False, index=False)





def remove_stopwords(text):
    stop_words = ['çok', 'at', 'in', 'im', 'acaba','acep','allah','adamakıllı','adeta','ait','altmýþ',
                  'altmış','altý',
                  'altı','ama',
                  'amma','anca','ancak','arada','artýk','aslında','aynen','ayrıca','az','açıkça','açıkçası',
                  'bana','bari','bazen','bazý','bazı','başkası','baţka','belki','ben','benden','beni','benim',
                  'beri','beriki','beþ','beş','beţ','bilcümle','bile','bin','binaen','binaenaleyh','bir','biraz',
                  'birazdan','birbiri','birden','birdenbire','biri','birice','birileri','birisi','birkaç',
                  'birkaçı','birkez','birlikte','birçok','birçoğu','birþey','birþeyi','birşey','birşeyi','birţey',
                  'bitevi','biteviye','bittabi','biz','bizatihi','bizce','bizcileyin','bizden','bize','bizi','bizim',
                  'bizimki','bizzat','boşuna','bu','buna','bunda','bundan','bunlar','bunları','bunların','bunu','bunun',
                  'buracıkta','burada','buradan','burası','böyle','böylece','böylecene','böylelikle','böylemesine',
                  'böylesine','büsbütün','bütün','cuk','cümlesi','da','daha','dahi','dahil','dahilen','daima','dair',
                  'dayanarak','de','defa','dek','demin','demincek','deminden','denli','derakap','derhal','derken',
                  'deđil','değil','değin','diye','diđer','diğer','diğeri','doksan','dokuz','dolayı','dolayısıyla',
                  'doğru','dört','edecek','eden','ederek','edilecek','ediliyor','edilmesi','ediyor','elbet',
                  'elbette','elli','emme','en','enikonu','epey','epeyce','epeyi','esasen','esnasında','etmesi',
                  'etraflı','etraflıca','etti','ettiği','ettiğini','evleviyetle','evvel','evvela','evvelce',
                  'evvelden','evvelemirde','evveli','eđer','eğer','fakat','filanca','gah','gayet','gayetle','gayri',
                  'gayrı','gelgelelim','gene','gerek','gerçi','geçende','geçenlerde','gibi','gibilerden','gibisinden',
                  'gine','göre','gırla','hakeza','halbuki','halen','halihazırda','haliyle','handiyse','hangi','hangisi',
                  'hani','hariç','hasebiyle','hasılı','hatta','hele','hem','henüz','hep','hepsi','her',
                  'herhangi','herkes','herkesin','hiç','hiçbir','hiçbiri','hoş','hulasaten','iken','iki',
                  'ila','ile','ilen','ilgili','ilk','illa','illaki','imdi','indinde','inen','insermi','ise',
                  'ister','itibaren','itibariyle','itibarıyla','iyi','iyice','iyicene','için','iş','işte',
                  'iţte','kadar','kaffesi','kah','kala','kanýmca','karşın','katrilyon','kaynak','kaçı','kelli',
                  'kendi','kendilerine','kendini','kendisi','kendisine','kendisini','kere','kez','keza','kezalik',
                  'keşke','keţke','ki','kim','kimden','kime','kimi','kimisi','kimse','kimsecik','kimsecikler',
                  'külliyen','kýrk','kýsaca','kırk','kısaca','lakin','leh','lütfen','maada','madem','mademki',
                  'mamafih','mebni','međer','meğer','meğerki','meğerse','milyar','milyon','mu','mü','mý','mı',
                  'nasýl','nasıl','nasılsa','nazaran','naşi','ne','neden','nedeniyle','nedenle','nedense',
                  'nerde','nerden','nerdeyse','nere','nerede','nereden','neredeyse','neresi','nereye',
                  'netekim','neye','neyi','neyse','nice','nihayet','nihayetinde','nitekim','niye','niçin',
                  'o','olan','olarak','oldu','olduklarını','oldukça','olduğu','olduğunu','olmadı',
                  'olmadığı','olmak','olması','olmayan','olmaz','olsa','olsun','olup','olur','olursa','oluyor',
                  'on','ona','onca','onculayın','onda','ondan','onlar','onlardan','onlari','onlarýn','onları',
                  'onların','onu','onun','oracık','oracıkta','orada','oradan','oranca','oranla','oraya','otuz',
                  'oysa','oysaki','pek','pekala','peki','pekçe','peyderpey','rağmen','sadece','sahi','sahiden',
                  'sana','sanki','sekiz','seksen','sen','senden','seni','senin','siz','sizden','sizi','sizin',
                  'sonra','sonradan','sonraları','sonunda','tabii','tam','tamam','tamamen','tamamıyla','tarafından',
                  'tek','trilyon','tüm','var','vardı','vasıtasıyla','ve','velev','velhasıl','velhasılıkelam','veya',
                  'veyahut','ya','yahut','yakinen','yakında','yakından','yakınlarda','yalnız','yalnızca','yani',
                  'yapacak','yapmak','yaptı','yaptıkları','yaptığı','yaptığını','yapılan','yapılması','yapıyor',
                  'yedi','yeniden','yenilerde','yerine','yetmiþ','yetmiş','yetmiţ','yine','yirmi','yok','yoksa',
                  'yoluyla','yüz','yüzünden','zarfında','zaten','zati','zira','çabuk','çabukça','çeşitli',
                  'çok','çokları','çoklarınca','çokluk','çoklukla','çokça','çoğu','çoğun','çoğunca','çoğunlukla',
                  'çünkü','öbür','öbürkü','öbürü','önce','önceden','önceleri','öncelikle','öteki','ötekisi','öyle',
                  'öylece','öylelikle','öylemesine','öz','üzere','üç','þey','þeyden','þeyi','þeyler','þu','þuna',
                  'þunda','þundan','þunu','şayet','şey','şeyden','şeyi','şeyler','şu','şuna','şuncacık','şunda',
                  'şundan','şunlar','şunları','şunu','şunun','şura','şuracık','şuracıkta','şurası','şöyle',
                  'ţayet','ţimdi','ţu','ţöyle', 'hala', 'yer', 'güzel', 'büyük']
    stop_words = ['a','acaba','altı','altmış','ama','ancak','arada','artık','asla','aslında','aslında','ayrıca',
                  'az','bana','bazen','bazı','bazıları','belki','ben','benden','beni','benim','beri','beş',
                  'bile','bilhassa','bin','bir','biraz','birçoğu','birçok','biri','birisi','birkaç','birşey',
                  'biz','bizden','bize','bizi','bizim','böyle','böylece','bu','buna','bunda','bundan','bunlar',
                  'bunları','bunların','bunu','bunun','burada','bütün','çoğu','çoğunu','çok','çünkü','da',
                  'daha','dahi','dan','de','defa','değil','diğer','diğeri','diğerleri','diye','doksan','dokuz',
                  'dolayı','dolayısıyla','dört','e','edecek','eden','ederek','edilecek','ediliyor','edilmesi',
                  'ediyor','eğer','elbette','elli','en','etmesi','etti','ettiği','ettiğini','fakat','falan',
                  'filan','gene','gereği','gerek','gibi','göre','hala','halde','halen','hangi','hangisi',
                  'hani','hatta','hem','henüz','hep','hepsi','her','herhangi','herkes','herkese','herkesi',
                  'herkesin','hiç','hiçbir','hiçbiri','i','ı','için','içinde','iki','ile','ilgili','ise',
                  'işte','itibaren','itibariyle','kaç','kadar','karşın','kendi','kendilerine','kendine',
                  'kendini','kendisi','kendisine','kendisini','kez','ki','kim','kime','kimi','kimin',
                  'kimisi','kimse','kırk','madem','mi','mı','milyar','milyon','mu','mü','nasıl','ne',
                  'neden','nedenle','nerde','nerede','nereye','neyse','niçin','nin','nın','niye','nun',
                  'nün','o','öbür','olan','olarak','oldu','olduğu','olduğunu','olduklarını','olmadı',
                  'olmadığı','olmak','olması','olmayan','olmaz','olsa','olsun','olup','olur','olur','olursa',
                  'oluyor','on','ön','ona','önce','ondan','onlar','onlara','onlardan','onları','onların',
                  'onu','onun','orada','öte','ötürü','otuz','öyle','oysa','pek','rağmen','sana','sanki',
                  'sanki','şayet','şekilde','sekiz','seksen','sen','senden','seni','senin','şey','şeyden',
                  'şeye','şeyi','şeyler','şimdi','siz','siz','sizden','sizden','size','sizi','sizi',
                  'sizin','sizin','sonra','şöyle','şu','şuna','şunları','şunu','ta','tabii','tam',
                  'tamam','tamamen','tarafından','trilyon','tüm','tümü','u','ü','üç','un','ün','üzere',
                  'var','vardı','ve','veya','ya','yani','yapacak','yapılan','yapılması','yapıyor','yapmak',
                  'yaptı','yaptığı','yaptığını','yaptıkları','ye','yedi','yerine','yetmiş','yi','yı','yine',
                  'yirmi','yoksa','yu','yüz','zaten','zira','zxtest']

    stop_words = ['acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'bir', 'birkaç', 'birşey', 'biz',
                  'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'den', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem',
                  'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü',
                  'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey',
                  'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani', 'dan']
    word_tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

corpus_df['processed9'] = corpus_df['processed8'].apply(remove_stopwords)
corpus_df['processed9'].to_csv('processed9withoutstops.csv', header=False, index=False)


import re
import json
import pickle
import sys
import nltk

def check(root, suffix, guess, action):
    if action == "unsuz yumusamasi":
        return len(suffix)>0 and suffix[0] in ["a","e","ı","i","o","ö","u","ü"] and checkSuffixValidation(suffix)[0]
    if action == "unlu daralmasi":
        if guess=="demek" and checkSuffixValidation(suffix)[0]:
            return True
        if guess=="yemek" and checkSuffixValidation(suffix)[0]:
            return True

        if suffix.startswith("yor"):
            lastVowel = ""
            for letter in reversed(guess[:-3]):
                if letter in ["a","e","ı","i","o","ö","u","ü"]:
                    lastVowel = letter
                    break
            if lastVowel in ["a","e"] and checkSuffixValidation(suffix)[0]:
                return True
        return False
    if action == "fiil" or action == "olumsuzluk eki":
        return checkSuffixValidation(suffix)[0] and not ((root.endswith("la") or (root.endswith("le"))) and suffix.startswith("r"))
    if action == "unlu dusmesi":
        count = 0
        for letter in guess:
            if letter in ["a","e","ı","i","o","ö","u","ü"]:
                count+=1
                lastVowel = letter
        if checkSuffixValidation(suffix)[0] and count==2 and (lastVowel in ["ı","i","u","ü"]) and (len(suffix)>0 and suffix[0] in ["a","e","ı","i","o","ö","u","ü"]):
            if lastVowel == "ı":
                return suffix[0] in ["a","ı"]
            elif lastVowel == "i":
                return suffix[0] in ["e","i"]
            elif lastVowel == "u":
                return suffix[0] in ["a","u"]
            elif lastVowel == "ü":
                return suffix[0] in ["e","ü"]
        return False
    return True

def findPos(kelime,revisedDict):
    l = []
    if "'" in kelime:
        l.append([kelime[:kelime.index("'")]+"_1","tirnaksiz",kelime])
    mid = []
    for i in range(len(kelime)):
        guess = kelime[:len(kelime)-i]
        suffix = kelime[len(kelime)-i:]
        ct = 1

        while guess+"_"+str(ct) in revisedDict:
            if check(guess, suffix, revisedDict[guess+"_"+str(ct)][1], revisedDict[guess+"_"+str(ct)][0]):
                guessList = (revisedDict[guess+"_"+str(ct)])
                while guessList[0] not in ["kok","fiil","olumsuzluk"]:
                    guessList = revisedDict[guessList[1]]
                mid.append([guessList[1], revisedDict[guess+"_"+str(ct)][0],guess+"_"+str(ct)])
            ct = ct+1

    temp = []
    for kel in mid:
        kelime_kok = kel[0][:kel[0].index("_")]
        kelime_len = len(kelime_kok)
        if kelime_kok.endswith("mak") or kelime_kok.endswith("mek"):
            kelime_len -= 3
        not_inserted = True
        for index in range(len(temp)):
            temp_kelime = temp[index]
            temp_kelime_kok = temp_kelime[0][:temp_kelime[0].index("_")]
            temp_len = len(temp_kelime_kok)
            if temp_kelime_kok.endswith("mak") or temp_kelime_kok.endswith("mek"):
                temp_len -= 3
            if(kelime_len>temp_len):
                temp.insert(index,kel)
                not_inserted = False
        if not_inserted:
            temp.append(kel)
    output = l+temp
    if len(output)==0:
        output.append([kelime+"_1","çaresiz",kelime+"_1",])
    return output

def checkSuffixValidation(suff):
    suffixList = ["","a", "abil", "acağ", "acak", "alım", "ama", "an", "ar", "arak", "asın", "asınız", "ayım", "da", "dan", "de", "den", "dı", "dığ", "dık", "dıkça", "dır", "di", "diğ", "dik", "dikçe", "dir", "du", "duğ", "duk", "dukça", "dur", "dü", "düğ", "dük", "dükçe", "dür", "e", "ebil", "eceğ", "ecek", "elim", "eme", "en", "er", "erek", "esin", "esiniz", "eyim", "ı", "ıl", "ım", "ımız", "ın", "ınca", "ınız", "ıp", "ır", "ıyor", "ız", "i", "il", "im", "imiz", "in", "ince", "iniz", "ip", "ir", "iyor", "iz", "k", "ken", "la", "lar", "ları", "ların", "le", "ler", "leri", "lerin", "m", "ma", "madan", "mak", "maksızın", "makta", "maktansa", "malı", "maz", "me", "meden", "mek", "meksizin", "mekte", "mektense", "meli", "mez", "mı", "mış", "mız", "mi", "miş", "miz", "mu", "muş", "mü", "muz", "müş", "müz", "n", "nın", "nız", "nin", "niz", "nun", "nuz", "nün", "nüz", "r", "sa", "se", "sı", "sın", "sınız", "sınlar", "si", "sin", "siniz", "sinler", "su", "sun", "sunlar", "sunuz", "sü", "sün", "sünler", "sünüz", "ta", "tan", "te", "ten", "tı", "tığ", "tık", "tıkça", "tır", "ti", "tiğ", "tik", "tikçe", "tir", "tu", "tuğ", "tuk", "tukça", "tur", "tü", "tüğ", "tük", "tükçe", "tür", "u", "ul", "um", "umuz", "un", "unca", "unuz", "up", "ur", "uyor", "uz", "ü", "ül", "ün", "üm", "ümüz", "ünce", "ünüz", "üp", "ür", "üyor", "üz", "ya", "yabil", "yacağ", "yacak", "yalım", "yama", "yan", "yarak", "yasın", "yasınız", "yayım", "ydı", "ydi", "ydu", "ydü", "ye", "yebil", "yeceğ", "yecek", "yelim", "yeme", "yen", "yerek", "yesin", "yesiniz", "yeyim", "yı", "yım", "yın", "yınca", "yınız", "yıp", "yız", "yi", "yim", "yin", "yince", "yiniz", "yip", "yiz", "yken", "yla", "yle", "ymış", "ymiş", "ymuş", "ymüş", "yor", "ysa", "yse", "yu", "yum", "yun", "yunca", "yunuz", "yup", "yü", "yuz", "yüm", "yün", "yünce", "yünüz", "yüp", "yüz"]
    validList = []
    if suff in suffixList:
        validList.append(suff)
    for ind in range(1,len(suff)):
        if(suff[:ind] in suffixList):
            cont, contList = checkSuffixValidation(suff[ind:])
            if cont:
                contList = [suff[:ind]+"+"+l for l in contList]
                validList = validList+contList
    return len(validList)>0,validList

try:
    with open('revisedDict.pkl', 'rb') as f:
        revisedDict = pickle.load(f)
except IOError:
    print("Please run trainLexicon.py to generate revisedDict.pkl file")

def lemma(text):
    words = nltk.word_tokenize(text)
    roots = []
    for word in words:
        #print("Possible lemmas for",word,"in ranked order:")
        root = findPos(word, revisedDict)[0][0][:-2]
        #print(root)
        roots.append(root)
    return ' '.join(roots)

corpus_df['processed10'] = corpus_df['processed9'].apply(lemma)
corpus_df['processed10'].to_csv('processed10stemmed.csv', header=False, index=False)

from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)

split_it = corpus_df['processed10'].str.split()
c = Counter(i for x in split_it for i in x)#common_words

def remove_commonwords(text):
    word_tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in common_words]
    return ' '.join(filtered_text)

common_words = ['ben', 'ol', 'şiir', 'var', 'yer', 'sen', 'kendi', 'bütün', 'gel', 'gün', 'gözle',
                'kadın', 'deniz', 'insan', 'sonra', 'iş', 'kadar', 'git', 'güzel', 'değil', 'zaman',
                'yüz', 'gece', 'kal', 'yan', 'kuş', 'dünya', 'akşam', 'yalnız', 'bak', 'yok', 'geç',
                'ses', 'çocuk', 'için', 'elle', 'iki', 'ölüm', 'on', 'gör', 'yaz', 'seni', 'kitap',
                'adam', 'al', 'düş', 'kız', 'eski', 'bun', 'çocukla', 'yeni', 'ilk', 'üstün', 'aşk',
                'son', 'söyle', 'baş', 'artık', 'şey', 'sokak', 'onlar', 'öyle', 'şimdi', 'gül', 'biz',
                'saat', 'dağla', 'yap', 'ev', 'd', 'güneş', 'uzak', 'iyi', 'göz', 'bırak', 'ölü', 'sev',
                'uzun', 'ban', 'gök', 'kapı', 'sabah', 'başka', 'ağız', 'başla', 'yaşam', 'el', 'bil',
                'l', 'böyle', 'saç', 'yol', 'altın', 'unut', 'şair', 'çıkar', 'biraz', 'san', 'beyaz',
                'şehir', 'yapı', 'bile', 'de', 'öl', 'ışık', 'büyük', 'söz', 'rüzgar', 'hayat', 'ali',
                'istanbul', 'önce', 'kent', 'günle', 'oda', 'kara', 'olur', 'iç', 'içim', 'gider', 'yine',
                'bili', 'gidi', 'şarkı', 'ağla', 'et'
                ]
cleaned_corpus = corpus_df['processed10'].dropna().apply(remove_commonwords)

#corpus_beshec = pd.read_excel("beshececiler.xlsx")
#corpus_yedimes = pd.read_excel("yedimesaleciler.xlsx")
#corpus_garip = pd.read_excel("garip.xlsx")
#corpus_ikinciyeni = pd.read_excel("ikinciyeni.xlsx")
#corpus_islami = pd.read_excel("islami.xlsx")
corpus_toplumcugercekci = pd.read_excel("toplumcugercekci.xlsx")





#words_of_cleaned_corpus = [str(word).split(' ') for word in cleaned_corpus]
#words_of_cleaned_corpus = [str(word).split(' ') for word in corpus_beshec]
#words_of_cleaned_corpus = [str(word).split(' ') for word in corpus_yedimes]
#words_of_cleaned_corpus = [str(word).split(' ') for word in corpus_garip]
#words_of_cleaned_corpus = [str(word).split(' ') for word in corpus_ikinciyeni]
#words_of_cleaned_corpus = [str(word).split(' ') for word in corpus_islami]
words_of_cleaned_corpus = [str(word).split(' ') for word in corpus_toplumcugercekci]





text_dict = Dictionary(words_of_cleaned_corpus)

#text_dict.filter_extremes(no_below=10, no_above=0.2)

corpus_bow = [text_dict.doc2bow(words_of_corpus) for words_of_corpus in words_of_cleaned_corpus]



k = 1
corpus_lda = LdaModel(corpus_bow,
                      num_topics = k,
                      id2word = text_dict)

'''
for i in  corpus_lda.show_topics():
    print (i[1], i[1])

#corpus_lda.show_topics()
'''

for i,topic in corpus_lda.show_topics(formatted=True, num_topics=1, num_words=25):
    print(str(i)+": "+ topic)
    print()


