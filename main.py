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
    self = re.sub(r"??", "A", self)
    self = re.sub(r"??", "I", self)
    self = re.sub(r"??", "??", self)
    self = re.sub(r"??", "a", self)
    self = re.sub(r"??", "u", self)
    self = re.sub(r"??", "U", self) # for the rest use default lower
    return self

corpus_df['processed5'] = corpus_df['processed4'].apply(lambda x: correct_old_characters(x))
corpus_df['processed5'].to_csv('processed5withoutaccented.csv', header=False, index=False)


def lower(self):
    self = re.sub(r"??", "i", self)
    self = re.sub(r"I", "??", self)
    self = re.sub(r"??", "??", self)
    self = re.sub(r"??", "??", self)
    self = re.sub(r"??", "??", self)
    self = re.sub(r"??", "??", self)
    self = self.lower() # for the rest use default lower
    return self

corpus_df['processed6'] = corpus_df['processed5'].apply(lambda x: lower(x))
corpus_df['processed6'].to_csv('processed6lower.csv', header=False, index=False)


whitelist = set('abc??defg??h??ijklmno??pqrs??tu??vwxyz ABC??DEFG??HI??JKLMNO??PQRS??TU??VWXYZ')

corpus_df['processed7'] = corpus_df['processed6'].apply(lambda x: ''.join(filter(whitelist.__contains__, x)))
corpus_df['processed7'].to_csv('processed7withoutnonturkish.csv', header=False, index=False)


corpus_df['processed8'] = corpus_df['processed7'].apply(gsp.strip_short)
corpus_df['processed8'].to_csv('processed8withoutshorts.csv', header=False, index=False)





def remove_stopwords(text):
    stop_words = ['??ok', 'at', 'in', 'im', 'acaba','acep','allah','adamak??ll??','adeta','ait','altm????',
                  'altm????','alt??',
                  'alt??','ama',
                  'amma','anca','ancak','arada','art??k','asl??nda','aynen','ayr??ca','az','a????k??a','a????k??as??',
                  'bana','bari','bazen','baz??','baz??','ba??kas??','ba??ka','belki','ben','benden','beni','benim',
                  'beri','beriki','be??','be??','be??','bilc??mle','bile','bin','binaen','binaenaleyh','bir','biraz',
                  'birazdan','birbiri','birden','birdenbire','biri','birice','birileri','birisi','birka??',
                  'birka????','birkez','birlikte','bir??ok','bir??o??u','bir??ey','bir??eyi','bir??ey','bir??eyi','bir??ey',
                  'bitevi','biteviye','bittabi','biz','bizatihi','bizce','bizcileyin','bizden','bize','bizi','bizim',
                  'bizimki','bizzat','bo??una','bu','buna','bunda','bundan','bunlar','bunlar??','bunlar??n','bunu','bunun',
                  'burac??kta','burada','buradan','buras??','b??yle','b??ylece','b??ylecene','b??ylelikle','b??ylemesine',
                  'b??ylesine','b??sb??t??n','b??t??n','cuk','c??mlesi','da','daha','dahi','dahil','dahilen','daima','dair',
                  'dayanarak','de','defa','dek','demin','demincek','deminden','denli','derakap','derhal','derken',
                  'de??il','de??il','de??in','diye','di??er','di??er','di??eri','doksan','dokuz','dolay??','dolay??s??yla',
                  'do??ru','d??rt','edecek','eden','ederek','edilecek','ediliyor','edilmesi','ediyor','elbet',
                  'elbette','elli','emme','en','enikonu','epey','epeyce','epeyi','esasen','esnas??nda','etmesi',
                  'etrafl??','etrafl??ca','etti','etti??i','etti??ini','evleviyetle','evvel','evvela','evvelce',
                  'evvelden','evvelemirde','evveli','e??er','e??er','fakat','filanca','gah','gayet','gayetle','gayri',
                  'gayr??','gelgelelim','gene','gerek','ger??i','ge??ende','ge??enlerde','gibi','gibilerden','gibisinden',
                  'gine','g??re','g??rla','hakeza','halbuki','halen','halihaz??rda','haliyle','handiyse','hangi','hangisi',
                  'hani','hari??','hasebiyle','has??l??','hatta','hele','hem','hen??z','hep','hepsi','her',
                  'herhangi','herkes','herkesin','hi??','hi??bir','hi??biri','ho??','hulasaten','iken','iki',
                  'ila','ile','ilen','ilgili','ilk','illa','illaki','imdi','indinde','inen','insermi','ise',
                  'ister','itibaren','itibariyle','itibar??yla','iyi','iyice','iyicene','i??in','i??','i??te',
                  'i??te','kadar','kaffesi','kah','kala','kan??mca','kar????n','katrilyon','kaynak','ka????','kelli',
                  'kendi','kendilerine','kendini','kendisi','kendisine','kendisini','kere','kez','keza','kezalik',
                  'ke??ke','ke??ke','ki','kim','kimden','kime','kimi','kimisi','kimse','kimsecik','kimsecikler',
                  'k??lliyen','k??rk','k??saca','k??rk','k??saca','lakin','leh','l??tfen','maada','madem','mademki',
                  'mamafih','mebni','me??er','me??er','me??erki','me??erse','milyar','milyon','mu','m??','m??','m??',
                  'nas??l','nas??l','nas??lsa','nazaran','na??i','ne','neden','nedeniyle','nedenle','nedense',
                  'nerde','nerden','nerdeyse','nere','nerede','nereden','neredeyse','neresi','nereye',
                  'netekim','neye','neyi','neyse','nice','nihayet','nihayetinde','nitekim','niye','ni??in',
                  'o','olan','olarak','oldu','olduklar??n??','olduk??a','oldu??u','oldu??unu','olmad??',
                  'olmad??????','olmak','olmas??','olmayan','olmaz','olsa','olsun','olup','olur','olursa','oluyor',
                  'on','ona','onca','onculay??n','onda','ondan','onlar','onlardan','onlari','onlar??n','onlar??',
                  'onlar??n','onu','onun','orac??k','orac??kta','orada','oradan','oranca','oranla','oraya','otuz',
                  'oysa','oysaki','pek','pekala','peki','pek??e','peyderpey','ra??men','sadece','sahi','sahiden',
                  'sana','sanki','sekiz','seksen','sen','senden','seni','senin','siz','sizden','sizi','sizin',
                  'sonra','sonradan','sonralar??','sonunda','tabii','tam','tamam','tamamen','tamam??yla','taraf??ndan',
                  'tek','trilyon','t??m','var','vard??','vas??tas??yla','ve','velev','velhas??l','velhas??l??kelam','veya',
                  'veyahut','ya','yahut','yakinen','yak??nda','yak??ndan','yak??nlarda','yaln??z','yaln??zca','yani',
                  'yapacak','yapmak','yapt??','yapt??klar??','yapt??????','yapt??????n??','yap??lan','yap??lmas??','yap??yor',
                  'yedi','yeniden','yenilerde','yerine','yetmi??','yetmi??','yetmi??','yine','yirmi','yok','yoksa',
                  'yoluyla','y??z','y??z??nden','zarf??nda','zaten','zati','zira','??abuk','??abuk??a','??e??itli',
                  '??ok','??oklar??','??oklar??nca','??okluk','??oklukla','??ok??a','??o??u','??o??un','??o??unca','??o??unlukla',
                  '????nk??','??b??r','??b??rk??','??b??r??','??nce','??nceden','??nceleri','??ncelikle','??teki','??tekisi','??yle',
                  '??ylece','??ylelikle','??ylemesine','??z','??zere','????','??ey','??eyden','??eyi','??eyler','??u','??una',
                  '??unda','??undan','??unu','??ayet','??ey','??eyden','??eyi','??eyler','??u','??una','??uncac??k','??unda',
                  '??undan','??unlar','??unlar??','??unu','??unun','??ura','??urac??k','??urac??kta','??uras??','????yle',
                  '??ayet','??imdi','??u','????yle', 'hala', 'yer', 'g??zel', 'b??y??k']
    stop_words = ['a','acaba','alt??','altm????','ama','ancak','arada','art??k','asla','asl??nda','asl??nda','ayr??ca',
                  'az','bana','bazen','baz??','baz??lar??','belki','ben','benden','beni','benim','beri','be??',
                  'bile','bilhassa','bin','bir','biraz','bir??o??u','bir??ok','biri','birisi','birka??','bir??ey',
                  'biz','bizden','bize','bizi','bizim','b??yle','b??ylece','bu','buna','bunda','bundan','bunlar',
                  'bunlar??','bunlar??n','bunu','bunun','burada','b??t??n','??o??u','??o??unu','??ok','????nk??','da',
                  'daha','dahi','dan','de','defa','de??il','di??er','di??eri','di??erleri','diye','doksan','dokuz',
                  'dolay??','dolay??s??yla','d??rt','e','edecek','eden','ederek','edilecek','ediliyor','edilmesi',
                  'ediyor','e??er','elbette','elli','en','etmesi','etti','etti??i','etti??ini','fakat','falan',
                  'filan','gene','gere??i','gerek','gibi','g??re','hala','halde','halen','hangi','hangisi',
                  'hani','hatta','hem','hen??z','hep','hepsi','her','herhangi','herkes','herkese','herkesi',
                  'herkesin','hi??','hi??bir','hi??biri','i','??','i??in','i??inde','iki','ile','ilgili','ise',
                  'i??te','itibaren','itibariyle','ka??','kadar','kar????n','kendi','kendilerine','kendine',
                  'kendini','kendisi','kendisine','kendisini','kez','ki','kim','kime','kimi','kimin',
                  'kimisi','kimse','k??rk','madem','mi','m??','milyar','milyon','mu','m??','nas??l','ne',
                  'neden','nedenle','nerde','nerede','nereye','neyse','ni??in','nin','n??n','niye','nun',
                  'n??n','o','??b??r','olan','olarak','oldu','oldu??u','oldu??unu','olduklar??n??','olmad??',
                  'olmad??????','olmak','olmas??','olmayan','olmaz','olsa','olsun','olup','olur','olur','olursa',
                  'oluyor','on','??n','ona','??nce','ondan','onlar','onlara','onlardan','onlar??','onlar??n',
                  'onu','onun','orada','??te','??t??r??','otuz','??yle','oysa','pek','ra??men','sana','sanki',
                  'sanki','??ayet','??ekilde','sekiz','seksen','sen','senden','seni','senin','??ey','??eyden',
                  '??eye','??eyi','??eyler','??imdi','siz','siz','sizden','sizden','size','sizi','sizi',
                  'sizin','sizin','sonra','????yle','??u','??una','??unlar??','??unu','ta','tabii','tam',
                  'tamam','tamamen','taraf??ndan','trilyon','t??m','t??m??','u','??','????','un','??n','??zere',
                  'var','vard??','ve','veya','ya','yani','yapacak','yap??lan','yap??lmas??','yap??yor','yapmak',
                  'yapt??','yapt??????','yapt??????n??','yapt??klar??','ye','yedi','yerine','yetmi??','yi','y??','yine',
                  'yirmi','yoksa','yu','y??z','zaten','zira','zxtest']

    stop_words = ['acaba', 'ama', 'asl??nda', 'az', 'baz??', 'belki', 'biri', 'bir', 'birka??', 'bir??ey', 'biz',
                  'bu', '??ok', '????nk??', 'da', 'daha', 'de', 'den', 'defa', 'diye', 'e??er', 'en', 'gibi', 'hem',
                  'hep', 'hepsi', 'her', 'hi??', 'i??in', 'ile', 'ise', 'kez', 'ki', 'kim', 'm??', 'mu', 'm??',
                  'nas??l', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'ni??in', 'niye', 'o', 'sanki', '??ey',
                  'siz', '??u', 't??m', 've', 'veya', 'ya', 'yani', 'dan']
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
        return len(suffix)>0 and suffix[0] in ["a","e","??","i","o","??","u","??"] and checkSuffixValidation(suffix)[0]
    if action == "unlu daralmasi":
        if guess=="demek" and checkSuffixValidation(suffix)[0]:
            return True
        if guess=="yemek" and checkSuffixValidation(suffix)[0]:
            return True

        if suffix.startswith("yor"):
            lastVowel = ""
            for letter in reversed(guess[:-3]):
                if letter in ["a","e","??","i","o","??","u","??"]:
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
            if letter in ["a","e","??","i","o","??","u","??"]:
                count+=1
                lastVowel = letter
        if checkSuffixValidation(suffix)[0] and count==2 and (lastVowel in ["??","i","u","??"]) and (len(suffix)>0 and suffix[0] in ["a","e","??","i","o","??","u","??"]):
            if lastVowel == "??":
                return suffix[0] in ["a","??"]
            elif lastVowel == "i":
                return suffix[0] in ["e","i"]
            elif lastVowel == "u":
                return suffix[0] in ["a","u"]
            elif lastVowel == "??":
                return suffix[0] in ["e","??"]
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
        output.append([kelime+"_1","??aresiz",kelime+"_1",])
    return output

def checkSuffixValidation(suff):
    suffixList = ["","a", "abil", "aca??", "acak", "al??m", "ama", "an", "ar", "arak", "as??n", "as??n??z", "ay??m", "da", "dan", "de", "den", "d??", "d????", "d??k", "d??k??a", "d??r", "di", "di??", "dik", "dik??e", "dir", "du", "du??", "duk", "duk??a", "dur", "d??", "d????", "d??k", "d??k??e", "d??r", "e", "ebil", "ece??", "ecek", "elim", "eme", "en", "er", "erek", "esin", "esiniz", "eyim", "??", "??l", "??m", "??m??z", "??n", "??nca", "??n??z", "??p", "??r", "??yor", "??z", "i", "il", "im", "imiz", "in", "ince", "iniz", "ip", "ir", "iyor", "iz", "k", "ken", "la", "lar", "lar??", "lar??n", "le", "ler", "leri", "lerin", "m", "ma", "madan", "mak", "maks??z??n", "makta", "maktansa", "mal??", "maz", "me", "meden", "mek", "meksizin", "mekte", "mektense", "meli", "mez", "m??", "m????", "m??z", "mi", "mi??", "miz", "mu", "mu??", "m??", "muz", "m????", "m??z", "n", "n??n", "n??z", "nin", "niz", "nun", "nuz", "n??n", "n??z", "r", "sa", "se", "s??", "s??n", "s??n??z", "s??nlar", "si", "sin", "siniz", "sinler", "su", "sun", "sunlar", "sunuz", "s??", "s??n", "s??nler", "s??n??z", "ta", "tan", "te", "ten", "t??", "t????", "t??k", "t??k??a", "t??r", "ti", "ti??", "tik", "tik??e", "tir", "tu", "tu??", "tuk", "tuk??a", "tur", "t??", "t????", "t??k", "t??k??e", "t??r", "u", "ul", "um", "umuz", "un", "unca", "unuz", "up", "ur", "uyor", "uz", "??", "??l", "??n", "??m", "??m??z", "??nce", "??n??z", "??p", "??r", "??yor", "??z", "ya", "yabil", "yaca??", "yacak", "yal??m", "yama", "yan", "yarak", "yas??n", "yas??n??z", "yay??m", "yd??", "ydi", "ydu", "yd??", "ye", "yebil", "yece??", "yecek", "yelim", "yeme", "yen", "yerek", "yesin", "yesiniz", "yeyim", "y??", "y??m", "y??n", "y??nca", "y??n??z", "y??p", "y??z", "yi", "yim", "yin", "yince", "yiniz", "yip", "yiz", "yken", "yla", "yle", "ym????", "ymi??", "ymu??", "ym????", "yor", "ysa", "yse", "yu", "yum", "yun", "yunca", "yunuz", "yup", "y??", "yuz", "y??m", "y??n", "y??nce", "y??n??z", "y??p", "y??z"]
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

common_words = ['ben', 'ol', '??iir', 'var', 'yer', 'sen', 'kendi', 'b??t??n', 'gel', 'g??n', 'g??zle',
                'kad??n', 'deniz', 'insan', 'sonra', 'i??', 'kadar', 'git', 'g??zel', 'de??il', 'zaman',
                'y??z', 'gece', 'kal', 'yan', 'ku??', 'd??nya', 'ak??am', 'yaln??z', 'bak', 'yok', 'ge??',
                'ses', '??ocuk', 'i??in', 'elle', 'iki', '??l??m', 'on', 'g??r', 'yaz', 'seni', 'kitap',
                'adam', 'al', 'd????', 'k??z', 'eski', 'bun', '??ocukla', 'yeni', 'ilk', '??st??n', 'a??k',
                'son', 's??yle', 'ba??', 'art??k', '??ey', 'sokak', 'onlar', '??yle', '??imdi', 'g??l', 'biz',
                'saat', 'da??la', 'yap', 'ev', 'd', 'g??ne??', 'uzak', 'iyi', 'g??z', 'b??rak', '??l??', 'sev',
                'uzun', 'ban', 'g??k', 'kap??', 'sabah', 'ba??ka', 'a????z', 'ba??la', 'ya??am', 'el', 'bil',
                'l', 'b??yle', 'sa??', 'yol', 'alt??n', 'unut', '??air', '????kar', 'biraz', 'san', 'beyaz',
                '??ehir', 'yap??', 'bile', 'de', '??l', '??????k', 'b??y??k', 's??z', 'r??zgar', 'hayat', 'ali',
                'istanbul', '??nce', 'kent', 'g??nle', 'oda', 'kara', 'olur', 'i??', 'i??im', 'gider', 'yine',
                'bili', 'gidi', '??ark??', 'a??la', 'et'
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


