import numpy as np
import pandas
import statsmodels.api as sm
import seaborn as sns

"""
In this question, you need to:
1) implement the linear_regression() procedure
2) Select features (in the predictions procedure) and make predictions.

"""

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.

    This can be the same code as in the lesson #3 exercise.
    """

    x = sm.add_constant(features)
    model = sm.OLS(values, x)
    result = model.fit()
    weights = result.params
    print weights
    intercept = weights[0]
    params = weights[1:]
    return intercept, params

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.

    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe.
    We recommend that you don't use the EXITSn_hourly feature as an input to the
    linear model because we cannot use it as a predictor: we cannot use exits
    counts as a way to predict entry counts.

    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    8 using gradient descent, or limit your number of features to 10 or so, since ordinary
    least squares can be very slow for a large number of features.

    If you receive a "server has encountered an error" message, that means you are
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features.
    '''
    ################################ MODIFY THIS SECTION #####################################
    # Select features. You should modify this section to try different features!             #
    # We've selected rain, precipi, Hour, meantempi, and UNIT (as a dummy) to start you off. #
    # See this page for more info about dummy variables:                                     #
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html          #
    ##########################################################################################
    print dataframe.columns
    dataframe['rush_hour'] = ((dataframe['Hour'] >= 10) & (dataframe['Hour'] <= 22)).astype(int)
    dataframe['dayofweek'] = pandas.to_datetime(dataframe['DATEn'], format='%d-%m-%y')
    dataframe['dayofweek'] = dataframe['dayofweek'].dt.dayofweek
    # print np.unique(dataframe['dayofweek'].dt.dayofweek)
    dataframe['weekday'] = ((dataframe['dayofweek'] < 5)).astype(int)
    features = dataframe[['rain',  'precipi',  'meantempi', 'maxpressurei', 'minpressurei', 'maxdewpti', 'fog','meanwindspdi', 'weekday', 'Hour', 'rush_hour']]
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)

    # Values
    values = dataframe['ENTRIESn_hourly']

    # Perform linear regression
    intercept, params = linear_regression(features, values)

    predictions = intercept + np.dot(features, params)
    import matplotlib.pylab as plt
    plt.plot(dataframe['ENTRIESn_hourly'].values, '-', alpha=0.5)
    plt.plot(predictions, '-', alpha=0.5)
    plt.xlabel('# Entry')
    plt.ylabel("ENTRIESn_hourly")
    plt.legend(["Measurements", "Predictions"])
    plt.title("Prediction-Measurement Comparison")
    plt.show()

    diff = abs(predictions- dataframe["ENTRIESn_hourly"].values)
    plt.plot(diff, '-', alpha=0.6)
    plt.fill_between(range(0, len(diff)), 0, diff)
    plt.xlabel('# Entry')
    plt.ylabel("ENTRIESn_hourly")
    plt.legend([r"$|$Predictions - Measurements$|$"])
    plt.title("Prediction-Measurement Comparison")
    plt.show()

    diff = predictions - dataframe["ENTRIESn_hourly"].values
    plt.hist(diff)
    plt.xlabel("Difference")
    plt.show()
    print "R^2 = %.3f" % compute_r_squared(dataframe["ENTRIESn_hourly"].values, predictions)
    return predictions


def compute_r_squared(data, predictions):
    '''
    In exercise 5, we calculated the R^2 value for you. But why don't you try and
    and calculate the R^2 value yourself.

    Given a list of original data points, and also a list of predicted data points,
    write a function that will compute and return the coefficient of determination (R^2)
    for this data.  numpy.mean() and numpy.sum() might both be useful here, but
    not necessary.

    Documentation about numpy.mean() and numpy.sum() below:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''

    SST = ((data - predictions)**2.0).sum()
    SSreg = ((data - np.mean(data))**2.0).sum()
    r_squared = 1 - SST / SSreg
    return r_squared

if __name__ == "__main__":
    df = pandas.read_csv("./data/turnstile_data_master_with_weather.csv", sep=",")
    # df = pandas.read_csv("./data/turnstile_weather_v2.csv", sep=",")
    predictions(df)
