from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import wordnet_ic
from gensim.models import Word2Vec
from scipy import spatial
import csv
import numpy

word_list = []
synonyms = {}
wordnet_results = {}
word2vec_results = {}
adVerb_result = {}

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
    with open(name, 'wt') as file:
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
        std = numpy.std(result_list)
        row.append(std)
        table1.append(row)
    return table1

def get_adverb_table(result_dict):
    table1_headers = ["Word", "Synonyms_wup", "Synonyms_lin", "Synonyms_w2v" "Average score wup", "Average score lin", "Average score wv", "StD wup", "StD lin", "StD w2v"]
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
        std1 = numpy.std(result_list1)
        std2 = numpy.std(result_list2)
        std3 = numpy.std(result_list3)
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


if __name__ == "__main__":
    Read_Synonyms()
    Calculate_Wordnet_Similarity()

    # task1 results
    table1 = get_results_list(wordnet_results)
    create_table("wup_similarity.csv", table1)

    # Task 2
    Word2Vec_Similarity()
    table2 = get_results_list(word2vec_results)
    create_table("word2vec_similarity.csv", table2)

    # Task 3
    Adverb_Similarity()
    table3 = get_adverb_table(adVerb_result)
    create_table("adverbs_task3.csv", table3)

