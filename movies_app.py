import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn

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
print("Dwie pierwsze recenzje:")
print(movie['data'][:2])

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
print("Wszystkie 2000 recenzji biore do treningu")
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

print()
print("Kazda recenzja to macierz rzadka. Na przyklad pierwsza:")
print(X[0])

print()
print("Wyswietla jako numpy")
print(X[0].toarray())

from sklearn.naive_bayes import MultinomialNB

print("Naiwny klasyfikator bayesowski")
classifier = MultinomialNB()
classifier.fit(X, y)
print()
print("Wynik score na zbiorze treningowym")
print(classifier.score(X, y))


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

img_robot = 'robot.jpeg'
encoded_img_robot = base64.b64encode(open(img_robot, 'rb').read())

app.layout = html.Div([

    html.Div([
        html.H4('Model Uczenia Maszynowego - wnioskowanie bayesowskie.'),
        html.H6('Na podstawie bazy 2000 recencji filmowych model będzie wnioskował'),
        html.H6('czy użytkownik ocenił film pozytywnie czy negatywnie.'),
        html.H6('Co myślisz o filmach "Krótkie spięcie" z 1986 i 1988 roku?'),
        html.H6("Opinię należy wpisać po angielsku.")
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

    dcc.Textarea(
        id='input-1',
        placeholder='Twoja opinia po angielsku...',
        style={'width': '100%'},
    ),


], style={'background-color': 'grey', 'color': 'white'})


@app.callback(
    Output('output-1', 'children'),
    [Input('input-1', 'value')]
)
def predict_sentiment(new_review):

    if new_review:
        new_review = [new_review]
        # to juz wczesniej utworzone
        # tfidf = TfidfVectorizer(max_features=3000)
        new_review_tfidf = tfidf.transform(new_review)
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


# if __name__ == '__main__':
#     app.run_server(debug=True)
