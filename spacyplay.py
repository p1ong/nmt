# A simple play area for experiemnting with Spacy 3.0
# The code below is derived from SpaCy 101 tutorial

import spacy

def testspacy():

    print("Spacy Play start ...\n")

    en_nlp = spacy.load("en_core_web_sm")
    en_text = "The cat sat on the mat and the cow jumped over the moon."
    doc = en_nlp(en_text)

    print("Pharsing: " + en_text + "\n")
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.dep_,token.shape_)



    print("\n... Spacy Play end")

    return