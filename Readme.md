# Eksamensbeskrivelse
## Spørgsmål
Baseret på demokraternes afsluttende udsagn, hvordan skiller Bernie Sanders sig ud fra de resterende demokratiske kandidater?
## Oversigt
I projektet vil vi benytte os af Natural Language Processing for at analysere de afsluttende udsagn fra de forskellige kandidater for det demokratiske parti i USA.

Vi vil gøre brug af web scraping og data cleaning til at hente og klargøre data til analyse.
Herefter vil vi analysere sproget og visualisere det med word clouds 
Til sidst lave en Sentiment Analysis for at finde ud af, om talerne er objektive eller subjektive og om budskabet er positivt eller negativt ladet.

## Teknologier
- Pandas
- (Web scraping)
- Word cloud
- Numpy
- Matplotlib
- Sklearn
- Nltk
- Pickle
- RegEx


## Installation:
- conda install nltk
- conda install -c conda-forge wordcloud
- conda install -c conda-forge spacy
- python -m spacy download en (skal køres fra en admin promt, brug evt. anaconda promt)
- pip install scattertext
- pip install pyLDavis
- pip install textblob

## How to use
In the jupyter notebook ‘Runner.ipynb’, there’s four blocks of code. 
Run each of them and the program will make magic.

## Udfordringer
De største problemer var rengøring af dataen der blev webscrapet samt brugen af sklearn. Andre problemer lå i konverteringen af test.ipynb til enkelte python filer.