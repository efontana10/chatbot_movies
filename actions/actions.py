# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.parsing.preprocessing import preprocess_string

# Load ML model
model = Doc2Vec.load(R'C:\Users\e.fontana\Desktop\movies\movies_doc2vec')
# Load dataset to get movie titles
df = pd.read_csv(R'C:\Users\e.fontana\Desktop\movies\wiki_movie_plots_deduped.csv', sep=',', usecols = ['Release Year', 'Title', 'Plot'])
df = df[df['Release Year'] >= 2000]

class ActionMovieSearch(Action):

	def name(self) -> Text:
		return "action_movie_search"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
		# use model to find the movie
		new_doc = preprocess_string(userMessage)
		test_doc_vector = model.infer_vector(new_doc)
		sims = model.dv.most_similar(positive = [test_doc_vector])		
		# Get first 5 matches
		movies = [df['Title'].iloc[s[0]] for s in sims[:5]]

		botResponse = f"I found the following movies: {movies}.".replace('[','').replace(']','')
		dispatcher.utter_message(text=botResponse)

		return []
