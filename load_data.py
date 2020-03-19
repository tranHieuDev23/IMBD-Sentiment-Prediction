from bs4 import BeautifulSoup
import csv


def strip_html(text):
    result = BeautifulSoup(text, 'lxml').get_text()
    return result


def load_dataset(filename, data_size=5000):
    comments = []
    sentiments = []
    with open(filename, 'r', encoding='utf-8') as input_file:
        csv_input = csv.reader(input_file)
        line_count = 0
        for row in csv_input:
            line_count += 1
            if (line_count == 1):
                continue
            if (line_count > data_size + 1):
                break
            comment = strip_html(row[0])
            sentiment = row[1]
            comments.append(comment)
            sentiments.append(sentiment)
    return (comments, sentiments)
