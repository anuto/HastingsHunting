import PyPDF2 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

STRING1 = "The Imperviousness of Literature to War."
STRING2 = "Readers and Writers."

#write a for-loop to open many files -- leave a comment if you'd #like to learn how
filename = 'newage18.pdf' 
#open allows you to read the file
pdfFileObj = open(filename,'rb')
#The pdfReader variable is a readable object that will be parsed
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
#discerning the number of pages will allow us to parse through all #the pages
num_pages = pdfReader.numPages
count = 0
text = ""
#The while loop will read each page
while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()

f = open(STRING1 + ".txt", "w")
text = text.replace("\n", "")
firstIndex = text.index(STRING1)
try:
	firstIndex = text[firstIndex + 1:].index(STRING1)
except ValueError:
	print("only one occurrence of string1")

secondIndex = text.index(STRING2)
try:
	secondIndex = text[secondIndex + 1:].index(STRING2)
except ValueError:
	print("only one occurrence of string2")

mamText = text[firstIndex : secondIndex]
f.write(mamText)

