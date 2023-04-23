"""
● We want to research the processes and challenges that will be associated with
translating the ASL sentences into English once we have identified the individual words
● Ideally we will make a prototype for this translation that can handle some simple
sentences, but this will depend on how many libraries already exist and how helpful they
will turn out to be.


NLTK (python library) seems to be an interesting option for this can tokenize sentences


list of tags and their meaning https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
Examples of tags https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk

Assuming we can kinda identify the POS of the words in the sentence, how do we reconstruct the sentence to match what we want
I imagine the steps of this process are going to be
1. Reorder the words in the sentence from TOPIC-COMMENT -> SUBJECT-VERB-OBJECT without adding words (Tomorrow library I go -> I go library tomorrow)
2. Identify the tense of the sentence (find time words, look for non-word queues) and conjugate the verb (I go library tomorrow -> I will go library tomorrow)
3. Add supporting words to make the sentence make sense (I will go library tomorrow -> I will go to the library tomorrow) (I bet a lot of these will be homebrew/vibes)

ideas for easy/distinct words
school, I, you, we, tomorrow, name, my, have, yesterday, what, go, hello, car
"""
import nltk
import sys


#TEST_SENTENCES = ["yesterday school I go", "my name Mat yesterday", "your name what", "yesterday car I have", "yesterday my school I go"]
TimeWords = ["tomorrow", "yesterday", "today"]

knownVerbs = [["have", "will have", "had"], ["go", "will go", "went"], ["is", "will be", "was"]]



foundObject = False
Question=False
Tense = ""


def printTaggedSentence(tagged):
    #loop through the array and add the words to the sentence
    sentence = ""
    for token in tagged:
        sentence+=token[0]
        sentence+=" "
    print(sentence)
def returnTaggedSentence(tagged):
    sentence = ""
    for token in tagged:
        sentence+=token[0]
        sentence+=" "
    return sentence
#now that we have figured out each word's place in the sentence, reorder the words
def sortWords(token):
    global foundObject
    POS=token[1]
    if token[0] in TimeWords:
        #we want to put our time words last in the sentence
        global Tense
        if(token[0] == "tomorrow"):
            Tense = "future"
        if(token[0]=="yesterday"):
            Tense = "past"
        return 4
    if "VB" in POS:
        #verbs go after the object
        return 2
    if "NN" in POS or "PRP" in POS:
        #the object typically comes first in the sentence before the subject
        if foundObject:
            return 1
        else:
            foundObject = True
            return 3
    if "W" in POS:
        Question=True
        return 1
    else :
        return 999

def combinePhrases(tagged):
    i = 0
    for token in tagged:
        if token[0]=="my" or token[0]=="your":#if we have a possession word, we must ensure that it stays next to whatever word it is possessing
            nextToken = tagged[i+1]
            x=list(token)
            x[0]+= " "+nextToken[0]
            x[1]=nextToken[1]
            tagged[i]=tuple(x)
            del tagged[i+1]
        i+=1
    #print(tagged)



#Conjugate the verb based on tense
def findVerbPlace(tagged):
    i = 0
    for token in tagged:
        if "VB" in token[1]:
            return i
        else:
            i+=1
    return 99
def findVerb(tagged):
    verbPlace = findVerbPlace(tagged)#find the verb in the sentence
    if(verbPlace==99):
        return knownVerbs[2]
    for verb in knownVerbs:#for each known verb
        for conjugated in verb:#and for each conjugation
            if tagged[verbPlace][0]==conjugated:#check 
                return verb
def conjugateSentence(tagged):
    verbPlace = findVerbPlace(tagged)
    foundVerb = []
    if(verbPlace==99):#if no verb is found in the sentence
        tagged.insert(1, ("is", "VBP"))#we assume in this case that the verb is 'is'
        foundVerb = knownVerbs[2]#the verb 'is'
        verbPlace=1
    else:
        foundVerb = findVerb(tagged)
    if(Tense=="future"):
        tagged[verbPlace]=(foundVerb[1], "VB")
    if(Tense=="past"):
        tagged[verbPlace]=(foundVerb[2], "VB")
def applyheuristics(tagged):
    verbPlace = findVerbPlace(tagged)
    verb = findVerb(tagged)
    if verb[0] =="go":
        tagged.insert(verbPlace+1, ("to", "TO"))
def main(sentence):
    global foundObject
    global Question
    global Tense
    foundObject = False
    Question=False
    Tense = ""
    tokens = nltk.word_tokenize(sentence)
    tagged=nltk.pos_tag(tokens)
    #print(tagged)
    combinePhrases(tagged)
    tagged.sort(key=sortWords)
    conjugateSentence(tagged)
    applyheuristics(tagged)
    return returnTaggedSentence(tagged)
if __name__ == "__main__":
    main(sys.argv[1])

    


