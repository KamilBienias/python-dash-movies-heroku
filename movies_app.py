import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import pandas as pd
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# trenowanie modelu
from sklearn.datasets import load_files

print("Załadowuje pliki z katalogu movie_reviews")
raw_movie = load_files('movie_reviews')
movie = raw_movie.copy()
movie.keys()

print()
print("Dwie ostatnie recenzje:")
print(movie['data'][-2:])

print()
print("0 to recenzja negatywna a 1 to pozytywna")
print(movie['target'][:10])

print()
print("Nazwy targetow")
print(movie['target_names'])

print()
print("Nazwy dwoch pierwszych plikow z filmami")
print(movie['filenames'][:2])

print()
print("Wszystkie recenzje biore do treningu")
X = movie['data']
print("Rozmiar X =", len(X))
y = movie['target']
print("Rozmiar y =", len(y))


# buduje macierz o wymiarze max 3000. Jeśli w dokumencie będzie 5000 różnych słów,
# to ucina do 3000 najczęściej występujących. Redukcja wymiarowości
tfidf = TfidfVectorizer(max_features=3000)
# dopasowuje i przetwarza zbiór treningowy
X = tfidf.fit_transform(X)
print()
print(f'X shape: {X.shape}')

# print()
# print("Kazda recenzja to macierz rzadka. Na przyklad pierwsza:")
# print(X[0])

# print()
# print("Wyswietla jako numpy")
# print(X[0].toarray())

from sklearn.naive_bayes import MultinomialNB

print("Naiwny klasyfikator bayesowski")
classifier = MultinomialNB()
classifier.fit(X, y)
print()
print("Wynik score na zbiorze treningowym")
print(classifier.score(X, y))


img_robot = 'robot.jpeg'
encoded_img_robot = base64.b64encode(open(img_robot, 'rb').read())

app.layout = html.Div([

    html.Div([
        html.H4('Model Uczenia Maszynowego - wnioskowanie bayesowskie.'),
        html.H6('Na podstawie bazy ' + str(len(movie["data"])) + ' recenzji filmowych model będzie wnioskował czy użytkownik ocenił film pozytywnie czy negatywnie.'),
        html.H6('Co myślisz o filmach "Krótkie spięcie" z 1986 i 1988 roku? Opinię wpisz po angielsku.')
    ], style={'textAlign': 'center'}),

    html.Hr(),

    html.Div([
        html.Img(src=f'data:image/png;base64,{encoded_img_robot.decode()}')
    ], style={'width': '200px', 'margin': 'auto'}
    ),

    html.Hr(),

    html.H4("Wynik powyżej 50% oznacza ocenę pozytywną",
            style={'textAlign': 'center'}),

    # sekcja wynikowa
    html.Div([
            html.Div(id='output-1'),
            html.Hr()
    ], style={'margin': '0 auto', 'textAlign': 'center'}),

    # druga sekcja wynikowa
    html.Div([
            html.Div(id='output-2'),
            html.Hr()
    ], style={'margin': '0 auto', 'textAlign': 'center'}),

    dcc.Textarea(
        id='input-1',
        placeholder='Twoja opinia po angielsku...',
        style={'width': '50%'}
    ),
    # wczesniej bylo Przeslij, ale nie zapisywal nowej recenzji
    html.Button("Wyświetl podsumowanie", id='button-1', n_clicks=0,
                style={'color': 'blue',
                       'background-color': 'yellow'}),


], style={'background-color': 'grey', 'color': 'white'})


@app.callback(
    Output('output-1', 'children'),
    [Input('input-1', 'value')]
)
def predict_sentiment(new_review):

    if new_review:
        new_review_as_list = [new_review]
        new_review_tfidf = tfidf.transform(new_review_as_list)
        new_review_prob = classifier.predict_proba(new_review_tfidf)
        new_review_prob_positive = new_review_prob[0][1]
        # print("new_review_prob")
        # print(new_review_prob[0][1])
        # print(type(new_review_pred))
        if new_review_prob_positive > 0.5:
            return html.Div([
                html.H4(f'Pozytywna ocena: {round(new_review_prob_positive * 100, 2)}%')
                ], style={'background-color': 'green',
                          'width': '60%',
                          'margin': '0 auto',
                          'color': 'white'})
        elif new_review_prob_positive <= 0.5:
            return html.Div([
                html.H4(f'Negatywna ocena: {round(new_review_prob_positive * 100, 2)}%')
                ], style={'background-color': 'red',
                          'width': '60%',
                          'margin': '0 auto',
                          'color': 'white'})


@app.callback(
    Output('output-2', 'children'),
    [Input('button-1', 'n_clicks'),
     State('input-1', 'value')]
)
def save_recension_adn_display_summary(n_clicks, new_review):

    # niestety na heroku nie zapisuje nowych recenzji do folderu,
    # ale na linuxie zapisuje
    if int(n_clicks) > 0:

        new_review_as_list = [new_review]
        new_review_tfidf = tfidf.transform(new_review_as_list)
        new_review_prob = classifier.predict_proba(new_review_tfidf)
        new_review_prob_positive = new_review_prob[0][1]

        if new_review_prob_positive > 0.5:
            # usunalem bo nie da sie zapisywac na heroku
            # numer to ilosc filmow plus ilosc klikniec
            # file_name = "recenzja_" + str(len(movie['data']) + int(n_clicks)) + ".txt"
            # positive_path = os.path.join("movie_reviews", "pos", file_name)
            # text_file = open(positive_path, "w")
            # text_file.write(str(new_review))
            # text_file.close()

            positive_folder = os.path.join("movie_reviews", "pos")
            # lista slow z nowej recenzji
            words_in_new_review = list(new_review.split())
            print("words_in_new_review:")
            print(words_in_new_review)

            # pusty slownik wystapien slow
            dict_of_words_with_counts = {}
            # dla kazdego slowa z nowej recenzji
            for word in words_in_new_review:
                # zamienia wszystkie litery na male
                word_lower_case = word.lower()
                # inicjuje licznik wystapien danego slowa w calym folderze
                word_counter = 0
                # otwiera kazdy plik w folderze movie_reviews/pos
                for filename in os.listdir(positive_folder):
                    content = open(os.path.join(positive_folder, filename), "r")
                    # bierze wiersze z pliku
                    for line in content:
                        # bierze kazde slowo z linii
                        for word_in_text_file in line.split():
                            if word_lower_case == word_in_text_file:
                                word_counter += 1
                print(word, word_counter)
                dict_of_words_with_counts[word] = word_counter

            return html.Div([
                html.H6("Rezenzja pozytywna o treści:"),
                html.H6(new_review),
                html.H6("Ilość powtórzeń każdego słowa w bazie recenzji pozytywnych:"),
                html.Div(str(dict_of_words_with_counts))
            ], style={"color": "green",
                      "background-color": "white"})

            # tego nie uzywam na heroku bo nie zapisuje nowych recenzji
            # return html.Div([
            #     html.H6("Rezenzja pozytywna z numerem " +
            #         str(len(movie['data']) + int(n_clicks))),
            #     html.H6("będzie w zbiorze treningowym przy następnym uruchomieniu. Jej treść:"),
            #     html.H6(new_review)
            # ], style={"color": "green"})

        elif new_review_prob_positive <= 0.5:
            # usunalem bo nie da sie zapisywac na heroku
            # numer to ilosc filmow plus ilosc klikniec
            # file_name = "recenzja_" + str(len(movie['data']) + int(n_clicks)) + ".txt"
            # negative_path = os.path.join("movie_reviews", "neg", file_name)
            # text_file = open(negative_path, "w")
            # text_file.write(str(new_review))
            # text_file.close()

            negative_folder = os.path.join("movie_reviews", "neg")
            # lista slow z nowej recenzji
            words_in_new_review = list(new_review.split())
            print("words_in_new_review:")
            print(words_in_new_review)

            # pusty slownik wystapien slow
            dict_of_words_with_counts = {}
            # dla kazdego slowa z nowej recenzji
            for word in words_in_new_review:
                # zamienia wszystkie litery na male
                word_lower_case = word.lower()
                # inicjuje licznik wystapien danego slowa w calym folderze
                word_counter = 0
                # otwiera kazdy plik w folderze movie_reviews/neg
                for filename in os.listdir(negative_folder):
                    content = open(os.path.join(negative_folder, filename), "r")
                    # bierze wiersze z pliku
                    for line in content:
                        # bierze kazde slowo z linii
                        for word_in_text_file in line.split():
                            if word_lower_case == word_in_text_file:
                                word_counter += 1
                print(word, word_counter)
                dict_of_words_with_counts[word] = word_counter

            return html.Div([
                html.H6("Rezenzja negatywna o treści:"),
                html.H6(new_review),
                html.H6("Ilość powtórzeń każdego słowa w bazie recenzji negatywnych:"),
                html.Div(str(dict_of_words_with_counts))
            ], style={"color": "red",
                      "background-color": "white"})

            # tego nie uzywam na heroku bo nie zapisuje nowych recenzji
            # return html.Div([
            #     html.H6("Rezenzja negatywna z numerem " +
            #             str(len(movie['data']) + int(n_clicks))),
            #     html.H6("będzie w zbiorze treningowym przy następnym uruchomieniu. Jej treść:"),
            #     html.H6(new_review)
            # ], style={"color": "red"})


# if __name__ == '__main__':
#     app.run_server(debug=True)
