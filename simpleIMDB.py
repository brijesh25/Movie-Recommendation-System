import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
from textwrap import wrap
from ast import literal_eval

metadata = pd.read_csv('./the-movies-dataset/movies_metadata-very-small.csv', low_memory=False)

# Calculation based on the ImetadataB formula
# Weighted Rating (WR) = (v/(v+m) * R) + (m/(m+v) * C)
# where,

# v is the number of votes for the movie;
# m is the minimum votes required to be listed in the chart;
# R is the average rating of the movie; And
# C is the mean vote across the whole report

C = metadata['vote_average'].mean()

m = metadata['vote_count'].quantile(0.95)

valid_movies = metadata.copy().loc[metadata['vote_count'] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

valid_movies['score'] = valid_movies.apply(weighted_rating, axis=1)
valid_movies = valid_movies.sort_values('score', ascending=False)
#print ("Serial No         Title        Vote Count       Vote Average     Score")

print(valid_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))

#Code for Bar Graph

titles = np.array(valid_movies['title'])[0:15]
score = np.array(valid_movies['score'].head(15))


#print(titles)
# print(score.shape)

titles = [ '\n'.join(wrap(l, 15)) for l in titles ]

from pylab import rcParams
rcParams['figure.figsize'] = 40,20

ind = np.arange(1,len(score)+1)
# print(ind)
plt.title("Bar Graph of Simple ImetadataB")
plt.ylabel("Score")
plt.xlabel("Movie Title")
plt.bar(ind, score,tick_label=titles,align = 'center')
plt.legend(['Score'])
plt.show()
plt.savefig('Simple-ImetadataB.png', bbox_inches='tight')