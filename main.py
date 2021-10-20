from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from gensim.models import Word2Vec
from scipy import spatial
import csv
import numpy

word_list = []
synonyms = {}
wordnet_results = {}
word2vec_results = {}

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
