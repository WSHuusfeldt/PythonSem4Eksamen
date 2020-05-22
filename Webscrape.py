import requests as requests
import bs4
import pickle
from pathlib import Path

urls = ["https://www.nytimes.com/interactive/2020/01/31/us/politics/pete-buttigieg-campaign-speech.html","https://www.nytimes.com/interactive/2020/01/31/us/politics/joe-biden-campaign-speech.html","https://www.nytimes.com/interactive/2020/01/31/us/politics/amy-klobuchar-campaign-speech.html","https://www.nytimes.com/interactive/2020/01/31/us/politics/bernie-sanders-campaign-speech.html","https://www.nytimes.com/interactive/2020/01/31/us/politics/elizabeth-warren-campaign-speech.html","https://www.nytimes.com/interactive/2020/01/31/us/politics/andrew-yang-campaign-speech.html"]

democrats = ["Pete Buttigieg", "Joe Biden", "Amy Klobuchar", "Bernie Sanders", "Elizabeth Warren", "Andrew Yang"]

def url_to_transcript_nytimess(url):
    """
        Returns: Text webscraped from URL.
    """
    r =  requests.get(url)
    soup = bs4.BeautifulSoup(r.text, "html.parser")
    content = soup.find_all('p', attrs={'class': 'g-body'})
    text = ""
    for txt in content:
        text += txt.text
    return text

def pickle_transcripts(urls):
    """
        Pickles the transcripts
    """
    transcripts = [url_to_transcript_nytimess(url) for url in urls]
    for i, c in enumerate(democrats):
        Path("transcripts").mkdir(parents=True, exist_ok=True)
        with open("transcripts/" + c + ".txt", "wb") as file:
            pickle.dump(transcripts[i], file)


def run():        
    pickle_transcripts(urls)