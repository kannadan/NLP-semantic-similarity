
word_list = []
synonyms = {}

def Read_Synonyms():
    with open('synonyms.txt') as f:        
        for line in f.readlines():
            words = line.split("-")
            if len(words) < 2:
                continue
            # Get origin word
            words[0] = words[0].replace(" ", "")
            word_list.append(words[0])
            # Get related synonyms
            synoms = words[1].split(",")
            synoms = list(map(lambda x: x.replace(" ", ""), synoms))
            synoms[len(synoms) - 1] = synoms[len(synoms) - 1].strip()            
            
            synonyms[words[0]] = synoms
    return




if __name__ == "__main__":
    Read_Synonyms()