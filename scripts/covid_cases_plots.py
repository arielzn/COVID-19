#!/usr/bin/env python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

CONFCSV = "../csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
# RECOCSV = "../csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
DEADCSV = "../csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

STATE = 'Belgium'
#STATE = 'Italy'
#STATE = 'Spain'
#STATE = 'France'
#STATE = 'Germany'

CASES_ZERO_DAY = 50


def df_state_filter(df):
    df.drop(df[df['Province/State'].notna()].index, inplace=True)   # keep full countries only, remove provinces/regions
    df.set_index('Country/Region', inplace=True)
    df.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
    df = df.loc[STATE]
    return df


def create_CRD_df(ccsv, rcsv, dcsv):
    cdf = pd.read_csv(ccsv)
    rdf = pd.read_csv(rcsv)
    ddf = pd.read_csv(dcsv)

    cdf = df_state_filter(cdf)
    rdf = df_state_filter(rdf)
    ddf = df_state_filter(ddf)

    df = pd.DataFrame({'Date': cdf.index, 'Confirmed': cdf.values,
                      'Recovered': rdf.values, 'Deaths': ddf.values})

    return df


def create_CD_df(ccsv, dcsv):
    cdf = pd.read_csv(ccsv)
    ddf = pd.read_csv(dcsv)

    cdf = df_state_filter(cdf)
    ddf = df_state_filter(ddf)

    df = pd.DataFrame({'Date': cdf.index, 'Confirmed': cdf.values,
                      'Deaths': ddf.values})

    df['dConf'] = df['Confirmed'] - df['Confirmed'].shift(1)
    df['dDeath'] = df['Deaths'] - df['Deaths'].shift(1)
    return df


def exp_fit(x, y):
    """ fit x, y input to a y=a*exp(b*x) function
    return a,b parameters and x, y values for the function on the fitted range"""

    fit_param = np.polyfit(x, np.log(y), 1)
    # fit_par = np.polyfit(x, np.log(y), 1,  w=np.sqrt(y))
    # sp.optimize.curve_fit(lambda t,a,b: a*np.exp(b*x), x, y )

    a = np.exp(fit_param[1])
    b = fit_param[0]
    fit_x = np.linspace(0, x_values[-1], 400)
    fit_y = a*np.exp(b*fit_x)

    return a, b, fit_x, fit_y


if __name__ == "__main__":

    dfcd = create_CD_df(CONFCSV, DEADCSV)

    #filter tables before ZERO_DAY
    dfcd = dfcd[dfcd['Confirmed'] >= CASES_ZERO_DAY]
    dfcd.index = range(len(dfcd))

    print("{0} cases after more than {1}\n".format(STATE, CASES_ZERO_DAY))
    print(dfcd)

    # fit confirmed curve
    x_values = dfcd.index
    y_valuesC = dfcd['Confirmed']
    y_valuesD = dfcd['Deaths']

    aC, bC, fit_x, fit_yC = exp_fit(x_values, y_valuesC)

    # create plots
    fdpi=96
    fig, axs = plt.subplots(3)
    fig.set_figwidth(400/fdpi)
    fig.set_figheight(1000/fdpi)
    x_label="Days since {0} confirmed".format(CASES_ZERO_DAY)

    axs[0].set_title('COVID cases evolution in {}'.format(STATE))
    # plot data scatters
    axs[0].scatter(x_values, y_valuesC, label="Confirmed", color='C0')
    axs[0].scatter(x_values, y_valuesD, label="Deaths", color='C1')
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel("Total cases")

    # plot fitted curve
    fit_function="{0:.3f}*exp({1:.3f}*x)".format(aC, bC)
    axs[0].plot(fit_x, fit_yC, label=fit_function, color='C0')
    axs[0].legend()

    # plot daily bar charts
    axs[1].bar(x_values, dfcd['dConf'], color='C0')
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel("Daily Confirmed")
    axs[2].bar(x_values, dfcd['dDeath'], color='C1')
    axs[2].set_xlabel(x_label)
    axs[2].set_ylabel("Daily Deaths")

    plt.tight_layout()
    figfile='covid_evolution_{0}.png'.format(STATE)
    plt.savefig(figfile, dpi=fdpi*1.5)
    plt.show()
