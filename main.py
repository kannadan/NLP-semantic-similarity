from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import wordnet_ic
from gensim.models import Word2Vec
from scipy import spatial
import csv
import pickle
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew

word_list = []
synonyms = {}
wordnet_results = {}
word2vec_results = {}
adVerb_result = {}
popularity_results = {}
antonyms = {}
antonym_results = {}
antonym_results_w2v = {}
antonym_pop_results = {}
antonym_wn_sorted = []
antonym_w2v_sorted = []
antonym_glove_sorted = []
antonym_pop_glove_results = {}

duos = {}
duo_results = {}
duo_results_w2v = {}
duo_pop_results = {}
duo_wn_sorted = []
duo_w2v_sorted = []

embeddings_dict = {}

def Read_Synonyms():
    with open('synonyms.txt') as f:
        for line in f.readlines():
            words = line.split("-")
            if len(words) < 2:
                continue
            # Get origin word
            words[0] = words[0].replace(" ", "").lower()
            word_list.append(words[0])
            # Get related synonyms
            synoms = words[1].split(",")
            synoms = list(map(lambda x: x.replace(" ", "").lower(), synoms))
            synoms[len(synoms) - 1] = synoms[len(synoms) - 1].strip()

            synonyms[words[0]] = synoms
    return

def Read_Antonyms():
    with open('antonyms.txt') as f:
        for line in f.readlines():
            words = line.split("-")
            if len(words) < 2:
                continue
            # Get origin word
            words[0] = words[0].replace(" ", "").lower()
            # Get related antonym
            words[1] = words[1].replace(" ", "").lower().strip()
            words[1] = words[1].split(",")[0]
            antonyms[words[0]] = words[1]
    return

def Read_Glove():
    with open("glove.42B.300d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

def Read_Duos():
    with open('Duos.txt') as f:
        for line in f.readlines():
            words = line.split("-")
            if len(words) < 2:
                continue
            # Get origin word
            words[0] = words[0].replace(" ", "").lower()
            # Get related antonym
            words[1] = words[1].replace(" ", "").lower().strip()
            words[1] = words[1].split(",")[0]
            duos[words[0]] = words[1]
    return

def Calculate_Wordnet_Similarity():
    for word in word_list:
        wordnet_results[word] = []
        main_word = get_synset(word)
        # print(f'Word: {word}, synset: {main_word}')
        if type(main_word) is int:
            continue
        similar = synonyms[word]
        for synonym in similar:
            word_type = main_word.name().split(".")[1]
            comparison = get_synset(synonym, word_type)
            if type(comparison) is int:
                continue
            value = main_word.wup_similarity(comparison)
            wordnet_results[word].append({
                'word': synonym,
                'result': value
            })

def Calculate_Wordnet_Similarity_Antonyms():
    for word in antonyms.keys():
        counter = antonyms[word]
        antonym_results[word] = {
            'word': counter,
            'result': None
        }
        main_word = get_synset(word)
        # print(f'Word: {word}, synset: {main_word}')
        if type(main_word) is int:
            continue
        word_type = main_word.name().split(".")[1]
        comparison = get_synset(counter, word_type)
        if type(comparison) is int:
            continue
        value = main_word.wup_similarity(comparison)
        antonym_results[word] = {
            'word': counter,
            'result': value
        }

def Calculate_Wordnet_Similarity_Duos():
    for word in duos.keys():
        counter = duos[word]
        duo_results[word] = {
            'word': counter,
            'result': None
        }
        main_word = get_synset(word)
        # print(f'Word: {word}, synset: {main_word}')
        if type(main_word) is int:
            continue
        word_type = main_word.name().split(".")[1]
        comparison = get_synset(counter, word_type)
        if type(comparison) is int:
            continue
        value = main_word.wup_similarity(comparison)
        duo_results[word] = {
            'word': counter,
            'result': value
        }

def Word2Vec_Similarity():
    # print("Start w2v")
    # model = Word2Vec(brown.sents(), min_count=1)
    # print("w2v model done")
    for word in word_list:
        word2vec_results[word] = []
        # print(f'Word: {word}, synset: {main_word}')

        similar = synonyms[word]
        model = Word2Vec([[word], similar], min_count=1)
        word_Vec = model.wv[word]
        for synonym in similar:
            # print(model.wv.similarity(word, synonym))
            res = spatial.distance.cosine(word_Vec, model.wv[synonym])
            word2vec_results[word].append({
                'word': synonym,
                'result': res
            })

def Word2Vec_Similarity_Antonyms():
    word_set = []
    for word in antonyms.keys():
        word_set.append([word])
        word_set.append([antonyms[word]])
    model = Word2Vec(word_set, min_count=1)
    for word in antonyms.keys():
        antonym = antonyms[word]
        # model = Word2Vec([[word], [antonym]], min_count=1)
        word_Vec = model.wv[word]
        res = spatial.distance.cosine(word_Vec, model.wv[antonym])
        antonym_results_w2v[word] = {
            'word': antonym,
            'result': res
        }

def Word2Vec_Similarity_Duos():
    word_set = []
    for word in duos.keys():
        word_set.append([word])
        word_set.append([duos[word]])
    model = Word2Vec(word_set, min_count=1)
    for word in duos.keys():
        second = duos[word]
        word_Vec = model.wv[word]
        res = spatial.distance.cosine(word_Vec, model.wv[second])
        duo_results_w2v[word] = {
            'word': second,
            'result': res
        }

def Glove_Similarity_Antonym():
    glove = []
    for word in antonyms.keys():
        second = antonyms[word]
        word_Vec = embeddings_dict[word] 
        res = spatial.distance.cosine(word_Vec, embeddings_dict[second])
        print(f'{word}:{second} - {res}')
        vec = {
            'word': second,
            'result': res
        }
        pop = antonym_pop_results[word]
        temp = vec
        temp["score_main"] = pop["score"]
        temp["score"] = pop["antonym"]["score"]
        temp["main"] = word
        temp["abs"] = pop["antonym"]["abs"]
        if temp["result"] is not None:
            glove.append(temp)    
    glove = sorted(glove, key=lambda a: a["abs"])    
    global antonym_glove_sorted    
    antonym_glove_sorted = glove
    x = []
    y = []
    for item in glove:
        x.append(item["result"])
        y.append(item["abs"])
    if len(x) == len(y) and len(y) > 1:
        x = np.array(x)
        y = np.array(y)
        r = np.corrcoef(x,y)        
        antonym_pop_glove_results["correlation_glove"] = r[0,1]
    else:
        antonym_pop_glove_results["correlation_glove"] = 0


def Antonym_Results_Processing():
    wordnet = []
    w2v = []
    for word in antonyms.keys():
        wordnet_res = antonym_results[word]
        w2v_res = antonym_results_w2v[word]
        w2v.append(w2v_res["result"])
        if wordnet_res["result"] is None:
            continue
        wordnet.append(wordnet_res["result"])
    antonym_results["average"] = np.average(wordnet)
    antonym_results_w2v["average"] = np.average(w2v)
    antonym_results["std"] = np.std(wordnet)
    antonym_results_w2v["std"] = np.std(w2v)
    antonym_results["skew"] = skew(wordnet)
    antonym_results_w2v["skew"] = skew(w2v)
    antonym_results["kurtosis"] = kurtosis(wordnet)
    antonym_results_w2v["kurtosis"] = kurtosis(w2v)

def Duos_Results_Processing():
    wordnet = []
    w2v = []
    for word in duos.keys():
        wordnet_res = duo_results[word]
        w2v_res = duo_results_w2v[word]
        w2v.append(w2v_res["result"])
        if wordnet_res["result"] is None:
            continue
        wordnet.append(wordnet_res["result"])
    duo_results["average"] = np.average(wordnet)
    duo_results_w2v["average"] = np.average(w2v)
    duo_results["std"] = np.std(wordnet)
    duo_results_w2v["std"] = np.std(w2v)
    duo_results["skew"] = skew(wordnet)
    duo_results_w2v["skew"] = skew(w2v)
    duo_results["kurtosis"] = kurtosis(wordnet)
    duo_results_w2v["kurtosis"] = kurtosis(w2v)


def get_synset(word, word_type=None):
    if word_type != None:
        synset = wn.synsets(word, word_type)
        if(len(synset) != 0):
            return synset[0]
        else:
            return 0
    synsetNoun = wn.synsets(word, 'n')
    if(len(synsetNoun) != 0):
        # print(synsetNoun)
        return synsetNoun[0]
    synsetVerb = wn.synsets(word, 'v')
    if(len(synsetVerb) != 0):
        # print(synsetVerb)
        return synsetVerb[0]
    return 0

def get_Derivative(synset):
    lemmas = synset.lemmas()
    derivative = None
    for i in range(len(lemmas)):
        derivatives = lemmas[i].derivationally_related_forms()
        if len(derivatives) != 0:
            derivative = get_verb_and_noun(derivatives)
            if derivative[0] is None and derivative[1] is None:
                derivative = None
                continue
            break
    return derivative

def get_verb_and_noun(lemmas):
    noun = None
    verb = None
    for lemma in lemmas:
        letter = lemma.synset().name().split(".")[1]
        if letter == "n" and noun is None:
            noun = lemma
        if letter == "v" and verb is None:
            noun = lemma
    return [noun, verb]

def create_table(name, rows):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)

def get_results_list(result_dict):
    table1_headers = ["Word", "Synonyms", "Average score", "StD"]
    table1 = [table1_headers]
    for word in word_list:
        results = result_dict[word]
        if len(results) == 0:
            continue
        row = [word]
        synonym_list = ""
        result_list = []
        for synonym in results:
            synonym_list += f'{synonym["word"]} ({synonym["result"]}), '
            result_list.append(synonym["result"])
        row.append(synonym_list)
        average = sum(result_list) / len(result_list)
        row.append(average)
        std = np.std(result_list)
        row.append(std)
        table1.append(row)
    return table1

def get_adverb_table(result_dict):
    table1_headers = ["Word", "Synonyms_wup", "Synonyms_lin", "Synonyms_w2v", "Average score wup", "Average score lin", "Average score wv", "StD wup", "StD lin", "StD w2v"]
    table1 = [table1_headers]
    for word in result_dict.keys():
        results = result_dict[word]
        if len(results) == 0:
            continue
        row = [f'({word}, {results[0]["derivative"].split(".")[0]})']
        synonym_list1 = ""
        synonym_list2 = ""
        synonym_list3 = ""
        result_list1 = []
        result_list2 = []
        result_list3 = []
        for synonym in results:
            deri = "-"
            if synonym["derSyn"]:
                deri = synonym["derSyn"].name().split(".")[0]
            synonym_list1 += f'({synonym["word"]}, {deri}) ({synonym["result_wup"]}), '
            synonym_list2 += f'({synonym["word"]}, {deri}) ({synonym["result_lin"]}), '
            synonym_list3 += f'({synonym["word"]}, {deri}) ({synonym["result_w2v"]}), '
            result_list1.append(synonym["result_wup"])
            result_list2.append(synonym["result_lin"])
            result_list3.append(synonym["result_w2v"])
        row.append(synonym_list1)
        row.append(synonym_list2)
        row.append(synonym_list3)
        average1 = sum(result_list1) / len(result_list1)
        average2 = sum(result_list2) / len(result_list2)
        average3 = sum(result_list3) / len(result_list3)
        row.append(average1)
        row.append(average2)
        row.append(average3)
        std1 = np.std(result_list1)
        std2 = np.std(result_list2)
        std3 = np.std(result_list3)
        row.append(std1)
        row.append(std2)
        row.append(std3)
        table1.append(row)
    return table1

def Adverb_Similarity():
    missed_main = 0
    missed_synonym = 0
    ic = wordnet_ic.ic('ic-semcor.dat')
    for word in word_list:
        main_word = get_synset(word)

        if type(main_word) is int:
            adVerb_result[word] = []
            # print(word)
            main_word = wn.synsets(word)
            derivative = get_Derivative(main_word[0])
            if derivative is None:
                missed_main += 1
                continue
            derivative = list(filter(lambda a: a is not None, derivative))
            if len(derivative) == 2:
                print("DOUBLE")
                #never happens, Meaning no derivate list has both noun and verb version
            main_word = derivative[0].synset()
            # print(f'Word: {word}, synset: {main_word}')
            similar = synonyms[word]
            derivativeSynonyms = []
            word_type = main_word.name().split(".")[1]
            for synonym in similar:
                comparison = get_synset(synonym, word_type)
                if type(comparison) is int:
                    derSyn = get_Derivative(wn.synsets(synonym)[0])
                    if derSyn is None:
                        missed_synonym += 1
                        continue
                    derSyn = list(filter(lambda a: a is not None, derivative))[0].synset()
                    name = derSyn.name().split(".")[0]
                    if(derivativeSynonyms.count(name) == 0):
                        derivativeSynonyms.append(derSyn.name().split(".")[0])
            if(derivativeSynonyms.count(main_word.name().split(".")[0]) == 0):
                derivativeSynonyms.append(main_word.name().split(".")[0])
            model = Word2Vec([[word], similar, derivativeSynonyms], min_count=1)
            word_Vec = model.wv[main_word.name().split(".")[0]]
            for synonym in similar:
                comparison = get_synset(synonym, word_type)
                derSyn = False
                if type(comparison) is int:
                    derSyn = get_Derivative(wn.synsets(synonym)[0])
                    if derSyn is None:
                        missed_synonym += 1
                        continue
                    derSyn = list(filter(lambda a: a is not None, derivative))[0].synset()
                    comparison = derSyn
                value = main_word.wup_similarity(comparison)
                lin = main_word.lin_similarity(comparison, ic)
                w2v = None
                if derSyn:
                    w2v = spatial.distance.cosine(word_Vec, model.wv[comparison.name().split(".")[0]])
                else:
                    w2v = spatial.distance.cosine(word_Vec, model.wv[synonym])
                adVerb_result[word].append({
                    "derivative": main_word.name(),
                    "derSyn": derSyn,
                    'word': synonym,
                    'result_wup': value,
                    'result_lin': lin,
                    "result_w2v": w2v
                })
    print(f'Missed main: {missed_main}, synonyms: {missed_synonym}')
    return

def get_popularity(corpus, word):
    score = corpus.count(word)
    return score

def Popularity_Similarity():
    #uses wup scoring for similarity
    words = brown.words()
    save_data = True
    try:
        a_file = open("popularity_data.pkl", "rb")
        output = pickle.load(a_file)
        global popularity_results
        popularity_results = output
        save_data = False
    except:
        print("no data found")

    if save_data:
        for word in wordnet_results.keys():
            popularity_results[word] = {"score": 0, "synonyms": []}
            popularity_results[word]["score"] = get_popularity(words, word)
            for synonyms in wordnet_results[word]:
                synonyms["score"] = get_popularity(words, synonyms["word"])
                synonyms["abs"] = abs(synonyms["score"] - popularity_results[word]["score"])
                popularity_results[word]["synonyms"].append(synonyms)
                print(popularity_results[word]["score"])
                print(synonyms)
        # delete pkl to recount scores
        a_file = open("popularity_data.pkl", "wb")
        pickle.dump(popularity_results, a_file)
        a_file.close()
    for word in popularity_results.keys():
        score = popularity_results[word]["score"]
        popularity_results[word]["synonyms"] = sorted(popularity_results[word]["synonyms"], key=lambda a: a["abs"])
        x = []
        y = []
        for syn in popularity_results[word]["synonyms"]:
            x.append(syn["result"])
            y.append(syn["abs"])
        if len(x) == len(y) and len(y) > 1:
            x = np.array(x)
            y = np.array(y)
            r = np.corrcoef(x,y)
            # print(r)
            popularity_results[word]["correlation"] = r[0,1]
        else:
            popularity_results[word]["correlation"] = 0
    return

def Antonym_Popularity():
    words = brown.words()
    save_data = True
    try:
        a_file = open("Antonym_popularity_data.pkl", "rb")
        output = pickle.load(a_file)
        global antonym_pop_results
        antonym_pop_results = output
        save_data = False
    except:
        print("no data found")
    if save_data:
        for word in antonyms.keys():
            antonym_pop_results[word] = {"score": 0, "antonym": {"score": 0, "antonym": antonyms[word], "abs": None}}
            antonym_pop_results[word]["score"] = get_popularity(words, word)
            antonym_pop_results[word]["antonym"]["score"] = get_popularity(words, antonyms[word])
            antonym_pop_results[word]["antonym"]["abs"] = abs(antonym_pop_results[word]["antonym"]["score"] - antonym_pop_results[word]["score"])
            print(antonym_pop_results[word]["score"])
            print(antonym_pop_results[word]["antonym"])
        # delete pkl to recount scores
        a_file = open("Antonym_popularity_data.pkl", "wb")
        pickle.dump(antonym_pop_results, a_file)
        a_file.close()

def Duo_Popularity():
    words = brown.words()
    save_data = True
    try:
        a_file = open("Duo_popularity_data.pkl", "rb")
        output = pickle.load(a_file)
        global duo_pop_results
        duo_pop_results = output
        save_data = False
    except:
        print("no data found")
    if save_data:
        for word in duos.keys():
            duo_pop_results[word] = {"score": 0, "partner": {"score": 0, "partner": duos[word], "abs": None}}
            duo_pop_results[word]["score"] = get_popularity(words, word)
            duo_pop_results[word]["partner"]["score"] = get_popularity(words, duos[word])
            duo_pop_results[word]["partner"]["abs"] = abs(duo_pop_results[word]["partner"]["score"] - duo_pop_results[word]["score"])
            print(duo_pop_results[word]["score"])
            print(duo_pop_results[word]["partner"])
        # delete pkl to recount scores
        a_file = open("Duo_popularity_data.pkl", "wb")
        pickle.dump(duo_pop_results, a_file)
        a_file.close()

def Antonym_Popular_Processed():
    w2v = []
    wordnet = []
    for word in antonyms.keys():
        wn = antonym_results[word]
        vec = antonym_results_w2v[word]
        pop = antonym_pop_results[word]
        temp = wn
        temp["score_main"] = pop["score"]
        temp["score"] = pop["antonym"]["score"]
        temp["main"] = word
        temp["abs"] = pop["antonym"]["abs"]
        if temp["result"] is not None:
            wordnet.append(temp)
        temp = vec
        temp["score_main"] = pop["score"]
        temp["score"] = pop["antonym"]["score"]
        temp["main"] = word
        temp["abs"] = pop["antonym"]["abs"]
        if temp["result"] is not None:
            w2v.append(temp)
    wordnet = sorted(wordnet, key=lambda a: a["abs"])
    w2v = sorted(w2v, key=lambda a: a["abs"])
    global antonym_wn_sorted
    global antonym_w2v_sorted
    antonym_wn_sorted = wordnet
    antonym_w2v_sorted = w2v
    x = []
    y = []
    for item in wordnet:
        x.append(item["result"])
        y.append(item["abs"])
    if len(x) == len(y) and len(y) > 1:
        x = np.array(x)
        y = np.array(y)
        r = np.corrcoef(x,y)
        # print(r)
        antonym_pop_results["correlation_wn"] = r[0,1]
    else:
        antonym_pop_results["correlation_wn"] = 0
    x = []
    y = []
    for item in w2v:
        x.append(item["result"])
        y.append(item["abs"])
    if len(x) == len(y) and len(y) > 1:
        x = np.array(x)
        y = np.array(y)
        r = np.corrcoef(x,y)
        # print(r)
        antonym_pop_results["correlation_w2v"] = r[0,1]
    else:
        antonym_pop_results["correlation_w2v"] = 0
    
def Duo_Popular_Processed():
    w2v = []
    wordnet = []
    for word in duos.keys():
        wn = duo_results[word]
        vec = duo_results_w2v[word]
        pop = duo_pop_results[word]
        temp = wn
        temp["score_main"] = pop["score"]
        temp["score"] = pop["partner"]["score"]
        temp["main"] = word
        temp["abs"] = pop["partner"]["abs"]
        if temp["result"] is not None:
            wordnet.append(temp)
        temp = vec
        temp["score_main"] = pop["score"]
        temp["score"] = pop["partner"]["score"]
        temp["main"] = word
        temp["abs"] = pop["partner"]["abs"]
        if temp["result"] is not None:
            w2v.append(temp)
    wordnet = sorted(wordnet, key=lambda a: a["abs"])
    w2v = sorted(w2v, key=lambda a: a["abs"])
    global duo_wn_sorted
    global duo_w2v_sorted
    duo_wn_sorted = wordnet
    duo_w2v_sorted = w2v
    x = []
    y = []
    for item in wordnet:
        x.append(item["result"])
        y.append(item["abs"])
    if len(x) == len(y) and len(y) > 1:
        x = np.array(x)
        y = np.array(y)
        r = np.corrcoef(x,y)
        # print(r)
        duo_pop_results["correlation_wn"] = r[0,1]
    else:
        duo_pop_results["correlation_wn"] = 0
    x = []
    y = []
    for item in w2v:
        x.append(item["result"])
        y.append(item["abs"])
    if len(x) == len(y) and len(y) > 1:
        x = np.array(x)
        y = np.array(y)
        r = np.corrcoef(x,y)
        # print(r)
        duo_pop_results["correlation_w2v"] = r[0,1]
    else:
        duo_pop_results["correlation_w2v"] = 0
    

def get_popularity_table():
    table1_headers = ["Word", "Synonyms", "Average score", "StD", "Pearson correlation"] 
    table1 = [table1_headers]
    for word in popularity_results.keys():
        results = popularity_results[word]
        if len(results["synonyms"]) == 0:
            continue
        row = [f'({word} - {results["score"]}']
        synonym_list = ""
        result_list = []
        for synonym in results["synonyms"]:
            synonym_list += f'{synonym["word"]} ({synonym["result"]} - {synonym["score"]}), '
            result_list.append(synonym["result"])
        row.append(synonym_list)
        average = sum(result_list) / len(result_list)
        row.append(average)
        std = np.std(result_list)
        row.append(std)
        row.append(results["correlation"])
        table1.append(row)
    return table1

def Get_Antonym_table():
    table1 = [["Average wup", antonym_results["average"], "Std wup", antonym_results["std"], "Skew wup", antonym_results["skew"], "Kurtosis wup", antonym_results["kurtosis"]]]
    table1.append(["Average w2v", antonym_results_w2v["average"], "Std w2v", antonym_results_w2v["std"], "Skew w2v", antonym_results_w2v["skew"], "Kurtosis w2v", antonym_results_w2v["kurtosis"]])
    table1_headers = ["Word", "Antonym", "Similarity wup", "Similarity w2v"] 
    table1.append(table1_headers)
    for word in antonyms.keys():
        results1 = antonym_results[word]["result"]
        results2 = antonym_results_w2v[word]["result"]
        antonym = antonyms[word]
        row = [word]
        row.append(antonym)
        row.append(results1)
        row.append(results2)
        table1.append(row)
    return table1

def Get_Duo_table():
    table1 = [["Average wup", duo_results["average"], "Std wup", duo_results["std"], "Skew wup", duo_results["skew"], "Kurtosis wup", duo_results["kurtosis"]]]
    table1.append(["Average w2v", duo_results_w2v["average"], "Std w2v", duo_results_w2v["std"], "Skew w2v", duo_results_w2v["skew"], "Kurtosis w2v", duo_results_w2v["kurtosis"]])
    table1_headers = ["Word", "duo", "Similarity wup", "Similarity w2v"] 
    table1.append(table1_headers)
    for word in duos.keys():
        results1 = duo_results[word]["result"]
        results2 = duo_results_w2v[word]["result"]
        partner = duos[word]
        row = [word]
        row.append(partner)
        row.append(results1)
        row.append(results2)
        table1.append(row)
    return table1

def Antonym_Pop_Table():
    table1 = [["Pearson correlation wup", antonym_pop_results["correlation_wn"]]]
    table1_headers = ["Word", "Popularity", "Antonym", "Popularity", "Abs", "Similarity wup"] 
    table1.append(table1_headers)
    for item in antonym_wn_sorted:
        row = [item["main"]]
        row.append(item["score_main"])
        row.append(item["word"])
        row.append (item["score"])
        row.append(item["abs"])
        row.append(item["result"])
        table1.append(row)
    table2 = [["Pearson correlation w2v", antonym_pop_results["correlation_w2v"]]]
    table2_headers = ["Word", "Popularity", "Antonym", "Popularity", "Abs", "Similarity w2v"] 
    table2.append(table1_headers)
    for item in antonym_w2v_sorted:
        row = [item["main"]]
        row.append(item["score_main"])
        row.append(item["word"])
        row.append (item["score"])
        row.append(item["abs"])
        row.append(item["result"])
        # print(item["main"])
        table2.append(row)
    return table1, table2

def Antonym_Pop_Glove_Table():
    table1 = [["Pearson correlation glove", antonym_pop_glove_results["correlation_glove"]]]
    table1_headers = ["Word", "Popularity", "Antonym", "Popularity", "Abs", "Similarity glove"] 
    table1.append(table1_headers)
    for item in antonym_glove_sorted:
        row = [item["main"]]
        row.append(item["score_main"])
        row.append(item["word"])
        row.append (item["score"])
        row.append(item["abs"])
        row.append(item["result"])
        table1.append(row)
    return table1

def Duo_Pop_Table():
    table1 = [["Pearson correlation wup", duo_pop_results["correlation_wn"]]]
    table1_headers = ["Word", "Popularity", "Duo", "Popularity", "Abs", "Similarity wup"] 
    table1.append(table1_headers)
    for item in duo_wn_sorted:
        row = [item["main"]]
        row.append(item["score_main"])
        row.append(item["word"])
        row.append (item["score"])
        row.append(item["abs"])
        row.append(item["result"])
        table1.append(row)
    table2 = [["Pearson correlation w2v", duo_pop_results["correlation_w2v"]]]
    table2_headers = ["Word", "Popularity", "Duo", "Popularity", "Abs", "Similarity w2v"] 
    table2.append(table1_headers)
    for item in duo_w2v_sorted:
        row = [item["main"]]
        row.append(item["score_main"])
        row.append(item["word"])
        row.append (item["score"])
        row.append(item["abs"])
        row.append(item["result"])
        # print(item["main"])
        table2.append(row)
    return table1, table2

if __name__ == "__main__":
    Read_Synonyms()
    Calculate_Wordnet_Similarity()

    # task1 results
    table1 = get_results_list(wordnet_results)
    create_table("wup_similarity_task1.csv", table1)

    # Task 2
    Word2Vec_Similarity()
    table2 = get_results_list(word2vec_results)
    create_table("word2vec_similarity_task2.csv", table2)

    # Task 3
    Adverb_Similarity()
    table3 = get_adverb_table(adVerb_result)
    create_table("adverbs_task3.csv", table3)

    # Task 4
    Popularity_Similarity()
    table4 = get_popularity_table()
    create_table("popularity_table_task4.csv", table4)

    # Task 5
    Read_Antonyms()
    Calculate_Wordnet_Similarity_Antonyms()
    Word2Vec_Similarity_Antonyms()
    Antonym_Results_Processing()
    table5 = Get_Antonym_table()
    create_table("antonym_table_task5.csv", table5)
    
    # Task 6
    Antonym_Popularity()
    Antonym_Popular_Processed()
    (table6, table7) = Antonym_Pop_Table()
    create_table("antonym_pop_wn_task6.csv", table6)
    create_table("antonym_pop_w2v_task6.csv", table7)

    #Task 7
    Read_Glove()
    print("READ GLOVE")
    Glove_Similarity_Antonym()
    table8 = Antonym_Pop_Glove_Table()
    create_table("antonym_pop_glove_task7.csv", table8)

    # Task 8
    Read_Duos()
    Calculate_Wordnet_Similarity_Duos()
    Word2Vec_Similarity_Duos()
    Duos_Results_Processing()
    table9 = Get_Duo_table()
    create_table("Duos_table_task8.csv", table9)

    Duo_Popularity()
    Duo_Popular_Processed()
    (table10, table11) = Duo_Pop_Table()
    create_table("duo_pop_wn_task8.csv", table10)
    create_table("duo_pop_w2v_task8.csv", table11)