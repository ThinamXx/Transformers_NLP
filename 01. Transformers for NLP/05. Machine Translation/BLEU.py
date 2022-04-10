#@ BILINGUAL EVALUATION UNDERSTUDY:
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

#@ EXAMPLE 1:
reference = [['the', 'cat', 'likes', 'milk'], ['cat', 'likes' 'milk']]
candidate = ['the', 'cat', 'likes', 'milk']
score = sentence_bleu(reference, candidate)
print("Example 1", score)

#@ EXAMPLE 2:
reference = [['the', 'cat', 'likes', 'milk']]
candidate = ['the', 'cat', 'likes', 'milk']
score = sentence_bleu(reference, candidate)
print("Example 2", score)

#@ EXAMPLE 3:
reference = [['the', 'cat', 'likes', 'milk']]
candidate = ['the', 'cat', 'enjoys', 'milk']
score = sentence_bleu(reference, candidate)
print("Example 3", score)

#@ CHENCHERRY SMOOTHING:
reference = [['je','vous','invite', 'a', 'vous', 'lever','pour', 'cette', 'minute', 'de', 'silence']]
candidate = ['levez','vous','svp','pour', 'cette', 'minute', 'de', 'silence']
score = sentence_bleu(reference, candidate)
print("Without smoothing score", score)

#@ CHENCHERRY SMOOTHING: 
chencherry = SmoothingFunction()
r1 = list("je vous invite a vous lever pour cette minute de silence")
candidate = list('levez vous svp pour cette minute de silence')
print("With smoothing score", sentence_bleu([r1], candidate, 
                                            smoothing_function=chencherry.method1))