import numpy as np 
import pandas as pd
from datetime import datetime
import smtplib, ssl
import time
import os
from selenium import webdriver

# For prediction purposes 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVR
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


def getStocks(n): 

    driver = webdriver.Chrome('/Users/nicoletraboulsi/Desktop/stock-predictor1/stock-predictor-files/chromedriver')
    url = "https://finance.yahoo.com/screener/predefined/growth_technology_stocks"
    driver.get(url)

    # A list to hold the ticker values for the stocks
    stock_list = []
    n += 1
    # Iterates through the ticker names on the stock screener list and stores the value of the ticker in stock_list
    for i in range(1, n):
        ticker = driver.find_element_by_xpath(
            '//*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(i) + ']/td[1]/a')
        stock_list.append(ticker.text)
    driver.quit()
    # Predict the future price of a stock at a specified amount of days
    number = 0
    for i in stock_list:
        print("Number: " + str(number))
        try:
            predictData(i, 5)
        except Exception as e: 
            print(e)
            print("Stock: " + i + " was not predicted")    
        number += 1

def sendMessage(output):

    smtp_server = "smtp.gmail.com"
    port = 587 # for starttls
    username = "nmtraboulsi@gmail.com"
    password = input("Type your password here: ")
    att = "7149066773@txt.att.net" 
    message = output 
    msg = """From: %s To: %s %s""" % (username, att, message)

    # Create a secure SSL context
    context = ssl.create_default_context()
    
    # Try to log in to server and email and send a text message
    server = smtplib.SMTP(smtp_server, port)
    server.ehlo()
    server.starttls(context=context)
    server.login(username, password)
    server.sendmail(username, att, msg)
    server.quit()
    print('sent')

def predictData(stock, days):
    print(stock)

    # Outputting the Historical data into a .csv for later use
    start = datetime(2018, 1, 1)
    end = datetime.now()
    
    df = yf.download(stock, start=start, end=end,output_format='pandas')
    csv_name = ('Exports/' + stock + '_Export.csv')
    df.to_csv(csv_name)
    df['prediction'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    forecast_time = int(days)

    # Predicting the Stock price in the future
    X = np.array(df.drop(['prediction'], 1))
    Y = np.array(df['prediction'])
    X = preprocessing.scale(X)
    X_prediction = X[-forecast_time:]
   
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
    #printing the train and test data
    print("Train Data:", X_train.shape, Y_train.shape)
    print("Test Data:", X_test.shape, Y_test.shape)

    # Performing the Regression on the training data
    #clf = LinearRegression()
    clf_lin = svm.SVR(kernel='linear', C=1).fit(X_train, Y_train)
    clf_poly = svm.SVR(kernel='poly', C=1, degree=2).fit(X_train, Y_train)
    clf_rbf = svm.SVR(kernel='rbf', C=1, gamma=0.1).fit(X_train, Y_train)
    #clf.fit(X_train, Y_train)
    prediction_lin = (clf_lin.predict(X_prediction))
    prediction_poly = (clf_poly.predict(X_prediction))
    prediction_rbf = (clf_rbf.predict(X_prediction))

    style.use('seaborn')
    file_path = '/Users/nicoletraboulsi/Desktop/stock-predictor1/stock-predictor-files/Exports/AYX_Export.csv'
    df2 = pd.read_csv(file_path, parse_dates=True, index_col=0)

    # printing the first 5 predictions for each stock
    print("First five predicted prices:", prediction_lin[0:5])
    print("First five predicted prices:", prediction_poly[0:5])
    print("First five predicted prices:", prediction_rbf[0:5])

    # Training on the test data 
    print("Scores for Linear on test data:", clf_lin.score(X_test, Y_test)) 
    print("Scores for Polynomial on test data:", clf_poly.score(X_test, Y_test))
    print("Scores for RBF on test data:", clf_rbf.score(X_test, Y_test))

    # Training on the train data (significantly better because it's already been trained)
    print("Scores for Linear on train data::", clf_lin.score(X_train, Y_train)) 
    print("Scores for Polynomial on train data:", clf_poly.score(X_train, Y_train))
    print("Scores for RBF on train data:", clf_rbf.score(X_train, Y_train))

    # Cross Validation- is using our training data in order to get estimates about how well our model will work on data we haven't seen before
    scores_lin = cross_val_score(clf_lin, X_train, Y_train, cv=5) 
    print("Cross Validation Scores for Linear:", scores_lin) 
    print("Accuracy for linear: %0.2f (+/- %0.2f)" % (scores_lin.mean(), scores_lin.std() * 2))

    scores_poly = cross_val_score(clf_poly, X_train, Y_train, cv=5)
    print("Cross Validation Scores for Polynomial:", scores_poly)  
    print("Accuracy for polynomial: %0.2f (+/- %0.2f)" % (scores_poly.mean(), scores_poly.std() * 2))

    scores_rbf = cross_val_score(clf_rbf, X_train, Y_train, cv=5)
    print("Cross Validation Scores for RBF:", scores_rbf)  
    print("Accuracy for rbf: %0.2f (+/- %0.2f)" % (scores_rbf.mean(), scores_rbf.std() * 2))
    
    last_row = df.tail(1)
    #print(last_row)

    # Sending the text message if the predicted price of the stock is at least 1 greater than the previous close price
    if (float(prediction_lin[4]) > (float(last_row['Close']))):
        output = ("\n\nStock:" + str(stock) + "\nPrior Close:\n" + str(last_row['Close']) + "\n\nPrediction in 1 day: " + str(prediction_lin[0]) + "\nPrediction in 5 days: " + str(prediction_lin[4]))
        sendMessage(output)

if __name__=='__main__':
    getStocks(10)
    
    