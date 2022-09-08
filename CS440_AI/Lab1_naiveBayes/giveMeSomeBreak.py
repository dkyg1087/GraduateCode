import reader
import naive_bayes as nb
from sklearn.naive_bayes import MultinomialNB

def main():
    train_set, train_labels, dev_set, dev_labels = nb.load_data("data/movie_reviews/train",'data/movie_reviews/dev')
    clf = MultinomialNB()
    clf.fit(train_set, train_labels)
    
if __name__ == '__main__':
    main()
    
