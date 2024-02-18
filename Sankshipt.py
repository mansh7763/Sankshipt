pip install streamlit
pip install requests
pip install nltk
pip install spacy
pip install transformers
pip install rake-nltk
pip install emojis
pip install emojis
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordne

import streamlit as st
import requests
import nltk
import spacy
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
from autocorrect import Speller
from bs4 import BeautifulSoup
from transformers import BartTokenizer, BartForConditionalGeneration
from rake_nltk import Rake
from transformers import pipeline





filtered_articles = []
unsafe_allow_html=True
no_of_words=150
final_keywords = []
# Create a list to store the urls and titles
urls=[]
titles=[]
fin=[]
prev_links = []

# Download the required NLTK resources

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def summarize_text(text, no_of_words):
    summarizer = pipeline("summarization")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    max_length=no_of_words+10;
    min_length=no_of_words-10;
    inputs = tokenizer([text], max_length=max_length, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
    # inputs = tokenizer([text], max_length=max_length, return_tensors='pt', truncation=True)
    # summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # return summary



def summ_and_scrape(urls, titles, headers,lang):

    # Scrape only the first URL from the urls list
    url = urls[0]

    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # This will get the text within <p> tags
        if "timesofindia.indiatimes.com" in url:
            p_tags = soup.find('div', class_='_s30J clearfix')
        else:
            p_tags = soup.find_all('p')

        # print(f"URL: {url}")
        article_text = ""
        word_count = 0
        for tag in p_tags:
            paragraph_text = tag.text.strip()
            words = paragraph_text.split()
            # Calculate the number of words in the paragraph
            word_count += len(words)
            if word_count <= 1000:
                # Add the text of each <p> tag to the article_text string
                article_text += paragraph_text + " "
            else:
                # Stop collecting text once the limit of 1000 words is reached
                break

        # Generate a summary for the current article using BERT-large
        summary_text = summarize_text(article_text,no_of_words)
        # def extract_keywords_rake(text, top_n=3):
        #     doc = nlp(text)
        #     if no_of_words <=60:
        #         top_n = 2
        #     # Extract noun chunks (keyphrases)
        #     keyphrases = [chunk.text for chunk in doc.noun_chunks]
            
        #     # Get the top N keyphrases
        #     extracted_keyphrases = keyphrases[:top_n]
            
        #     return extracted_keyphrases
        def extract_keywords_rake(text, top_n=2):
            nlp = spacy.load("en_core_web_sm")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            doc = nlp(text)            
            stopwords = set(STOP_WORDS)
            custom_stopwords = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday","The decision"}  # Add any additional words you want to exclude
            keyphrases = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() not in stopwords and chunk.text not in custom_stopwords]
            extracted_keyphrases = keyphrases[:top_n]
            return extracted_keyphrases
        from colorama import init, Fore, Style
        init(autoreset=True)  # Initialize colorama
        def highlight_keywords_rake(highlighted_paragraph, keywords):

            for keyword in keywords:
                replacement = f":red[{keyword}]"
                final_keywords.append(keyword)
                highlighted_paragraph = highlighted_paragraph.replace(keyword, replacement,1)
            return highlighted_paragraph

        # Extract and highlight only the top 3 keywords using RAKE
        rake_keywords = extract_keywords_rake(summary_text, top_n=2)
        # Highlight RAKE-based keywords in blue
        highlighted_paragraph_rake = highlight_keywords_rake(summary_text, rake_keywords)
        # language of displaying the summary:
        languages_dict = {  'english': 'en',
                            'hindi': 'hi',
                            'bengali': 'bn',
                            'telugu': 'te',
                            'marathi': 'mr',
                            'tamil': 'ta',
                            'urdu': 'ur',
                            'gujarati': 'gu',
                            'kannada': 'kn',
                            'odia': 'or',
                            'punjabi': 'pa',
                        }                                       

        
        from translate import Translator

        def translate_english_to_hindi(text,language_name):
            # Convert the user input to lowercase for case-insensitivity
            language_key = language_name.lower()
            # Check if the entered language is in the dictionary
            if language_key in languages_dict:
                language_key = language_key
                language_name = languages_dict[language_key]

            translator = Translator(to_lang=language_name)
            highlighted_paragraph_rake = translator.translate(text)
            return highlighted_paragraph_rake


        st.subheader("Summarized points:")

        st.markdown(translate_english_to_hindi(highlighted_paragraph_rake,lang))
        st.write("\n---\n")
    except requests.exceptions.Timeout:
        pass
        # st.write(f"Skipping {url} due to timeout\n---\n")

    # Displaying urls
    for i in range(2):  
        st.write(titles[i])
        st.write(urls[i])
        st.write("\n---\n")
#Preprocessing the prompt
def preprocess_prompt(prompt):
    prompt = prompt.lower()
    # Spell correction
    spell = Speller()
    prompt = spell(prompt)   
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in word_tokenize(prompt) if token not in stop_words]
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # Return the list of tokens
    return " ".join(tokens)

# Create a list to store the filtered article

def fetch_and_filter_articles(api_key, prompt, valid_links,prev_links, language='en'):
    base_url = f'https://newsapi.org/v2/everything?qInTitle={prompt}&apiKey={api_key}'
    # Set the parameters for the API request
    params = {
        'apiKey': api_key,
        'q': prompt,
        'language': language,
    }
    # Make the API request
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract relevant information from the response
        total_results = data.get('totalResults', 0)
        articles = data.get('articles', [])
        # Print the total number of articles before filtering
        st.write(f'Total Related Articles Found: {total_results}')
        # Filter articles based on the list of valid links
        for index, article in enumerate(articles, start=1):
            url = article.get('url', '')
            title = article.get('title', '')
            for i in valid_links:
                if i in url:
                  cnt = 0
                  for k in prev_links:
                    if k != url:
                      cnt+=1
                  if 1 or cnt == len(prev_links):
                    fin.append(url)
                    titles.append(title)
                    valid_links.remove(i)
            # Check if any of the valid links is a substring of the article link
            if any(link in url for link in valid_links):
                filtered_articles.append(article)

        # Print the number of articles after filtering
        # num_articles_after_filtering = len(filtered_articles)
        # print(f'Number of Articles After Filtering: {num_articles_after_filtering}')

        # Print or store the information for the filtered articles
        for index, article in enumerate(filtered_articles, start=1):
            if index > 10:
                break
            url = article.get('url', '')
            title = article.get('title', '')
            if not url:
                continue
            urls.append(url)
            titles.append(title)
            # title = article.get('title', '')
            # description = article.get('description', '')
            # url = article.get('url', '')
            # source_name = article.get('source', {}).get('name', '')

            # Process the title, description, and URL as needed

            # Print or store the information
            # print(f'Article {index}:')
            # print(f'Title: {title}')
            # print(f'Description: {description}')
            # print(f'URL: {url}')
            # print(f'Websites: {source_name}')
            # print('---')
            # print('\n \n')
    else:
        st.write(f'Error: {response.status_code}, {response.text}')
        return 0
    
def fetch_and_filter_articles_title(api_key, prompt, valid_links,prev_links, language='en'):
    base_url = f'https://newsapi.org/v2/everything?qInTitle={prompt}&apiKey={api_key}'
    # Set the parameters for the API request
    params = {
        'apiKey': api_key,
        'q': prompt,
        'language': language,
    }
    # Make the API request
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract relevant information from the response
        total_results = data.get('totalResults', 0)
        articles = data.get('articles', [])
        # Print the total number of articles before filtering
        # st.write(f'Total Articles Found before filtering: {total_results}')
        # Filter articles based on the list of valid links
        for index, article in enumerate(articles, start=1):
            url = article.get('url', '')
            title = article.get('title', '')
            for i in valid_links:
                if i in url:
                    fin.append(url)
                    titles.append(title)
                    valid_links.remove(i)                   
            # Check if any of the valid links is a substring of the article link
            if any(link in url for link in valid_links):
                filtered_articles.append(article)

        # Print the number of articles after filtering
        # num_articles_after_filtering = len(filtered_articles)
        # print(f'Number of Articles After Filtering: {num_articles_after_filtering}')

        # Print or store the information for the filtered articles
        for index, article in enumerate(filtered_articles, start=1):
            if index > 10:
                break
            url = article.get('url', '')
            title = article.get('title', '')
            if not url:
                continue
            urls.append(url)
            titles.append(title)
            # title = article.get('title', '')
            # description = article.get('description', '')
            # url = article.get('url', '')
            # source_name = article.get('source', {}).get('name', '')

            # Process the title, description, and URL as needed

            # Print or store the information
            # print(f'Article {index}:')
            # print(f'Title: {title}')
            # print(f'Description: {description}')
            # print(f'URL: {url}')
            # print(f'Websites: {source_name}')
            # print('---')
            # print('\n \n')
        st.write(title)
        st.write(url)
    else:
        st.write(f'Error: {response.status_code}, {response.text}')
        return 0
# Replace 'your_api_key' with your actual NewsAPI key
api_key = '<Your_api_key>'

# List of valid website links
website_links = [
    'ft.com', 'wsj.com', 'bloomberg.com', 'asia.nikkei.com', 'economist.com', 'variety.com',
    'hollywoodreporter.com', 'deadline.com', 'webmd.com', 'mayoclinic.org', 'nih.gov', 'who.int',
    'thelancet.com', 'bbc.com', 'edition.cnn.com', 'aljazeera.com', 'nytimes.com', 'theguardian.com',
    'sciencenews.org', 'nature.com', 'sciencemag.org', 'newscientist.com', 'scientificamerican.com',
    'espn.com', 'thehindu.com', 'indianexpress.com', 'timesofindia.indiatimes.com',
    'telegraphindia.com', 'economictimes.indiatimes.com', 'livemint.com'
]

# Take input from user about prompt
import emojis
st.markdown("# <span style='font-size:70px;'>WELCOME TO</span>", unsafe_allow_html=True)
st.image("logo.png", width=400)
st.write("\n---\n")
prompt = st.text_input(f"Enter what you want to search for: {emojis.encode(':sparkles:')}", placeholder=f"Enter the prompt")

no_of_words=st.select_slider("Enter the number of words you want to summarize",options=[30,40,50,60,70,80])

lang=st.selectbox("Select the language: ",("English","Hindi","Bengali","Telugu","Marathi","Tamil","Urdu","Gujarati","Kannada","Odia","Punjabi"),index=0)


# Create a form to take user input


if st.button("Summarize"):
    # This will print first three articles
    if(len(prompt)>0):
        prompt = preprocess_prompt(prompt)
        fetch_and_filter_articles(api_key, prompt, website_links,prev_links)
        summ_and_scrape(fin, titles, headers,lang)
        # st.write("\n---\n")
        # st.write("Articles related to the highlighted keywords:")
        st.subheader("Articles related to the highlighted keywords:")
        st.write("\n---\n")
        for i in final_keywords:
            p = preprocess_prompt(i)
            print(p)
            fetch_and_filter_articles_title(api_key, p, website_links,prev_links)
            st.write("\n---\n")






# Take input from user about prompt





# This part is all about scrapping the text and summarizing it

# Define the BART-large tokenizer and model
# paragraph = "This is a sample paragraph with a keyword."
# keyword = "with a keyword"

# Highlight the keyword in the paragraph using a custom CSS class
# highlighted_paragraph = paragraph.replace(keyword, f'<span class="highlight">{keyword}</span>', 1)

# Write the highlighted paragraph with custom CSS
# st.markdown(
#     f"<p>{highlighted_paragraph}</p>",
#     unsafe_allow_html=True
# )

# Add custom CSS to the app
