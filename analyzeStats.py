#!/usr/bin/env python3


def loadFile(fName):
    import os
    if not os.path.isfile(fName):
        return None
    with open(fName) as f:
        arr = []
        for l in f:
            if 'Total Duration' in l: continue
            if 'ucjf-zoom' in l: continue
            l = [x.strip() for x in l.split(',')]
            arr.append( [l[0], l[1], int(l[2]) ] )

        return arr
    return None

def loadDay(day):
    sess = ['01_Higgs',
            '02_Neutrino',
            '03_BSM',
            '04_TopEW',
            '05_QuarkLepton',
            '06_StrongHadron',
            '07_HeavyIons',
            '08_AstroParticle',
            '09_DarkMatter',
            '10_Theory',
            '11_Accelerrators',
            '12_PresentDetectors',
            '13_FutureDetectors',
            '14_Computing',
            '15_EducationOutreach',
            '16_DiversityInclusion',
            '17_TechnologyIndustry']


    allDays = {}
    for s in sess:
        d = loadFile(day+'/participants_'+s+'_'+day+'.csv')
        if d != None:
            allDays[s] = d
    return allDays


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadPlenary():
    dfAll = pd.DataFrame(columns = ['name', 'mail', 'mins', 'day', 'session'])
    for i,d in  enumerate(['Monday', 'Tuesday']):
        data = loadFile('Plenaries/ICHEP2020_Plenary'+str(i+1)+'.csv')
        df = pd.DataFrame(data, columns = ['name', 'mail', 'mins']) 
        df['day'] = d
        df['session'] = 'Plenary'
        dfAll = dfAll.append(df)

    return dfAll

def loadAll():

    dfAll = pd.DataFrame(columns = ['name', 'mail', 'mins', 'day', 'session'])

    for day in ['Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        data = loadDay(day)

        for s in data:
            df = pd.DataFrame(data[s], columns = ['name', 'mail', 'mins']) 
            df['day'] = day
            df['session'] = s
            dfAll = dfAll.append(df)
            #print(df)

    dfAll['mins'] = np.clip(dfAll['mins'], a_min=0, a_max=60*6)

    return dfAll

def plotDaysTotal(df, dfPlen):
    days = []
    vals = []
    for d in df['day'].unique():
        days.append(d)
        val = len(df[(df['day'] == d) & (df['mins'] >= 15)]['name'].unique())
        vals.append(val)

    for d in ['Monday', 'Tuesday']:
        valPlen = len(dfPlen[(dfPlen['mins'] >= 15) & (dfPlen['day'] == d)]['name'].unique())
        days.append(d)
        vals.append(valPlen)

    
    days[-1] = ' '+days[-1]+' '
    plt.bar(days, vals)
    plt.ylabel('#participants')
    #plt.show()
    plt.savefig('plots/DaysTotal.png')
    plt.close()


def plotSessionsTotal(df):
    sess = df['session'].unique()
    sess = np.sort(sess)[::-1]
    vals = []
    for s in sess:
        #sess.append(s)
        val = len(df[(df['session'] == s) & (df['mins'] >= 15)]['name'].unique())
        vals.append(val)


    plt.barh(sess, vals)
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    plt.xlabel('#participants')
    #plt.show()
    plt.savefig('plots/SessionsTotal.png')
    plt.close()


def plotDurations(df):
    df['hours'] = df['mins']/60.
    res = df.groupby('name')['hours'].agg(np.sum)

    plt.hist(res, bins=np.arange(0,25, 0.25)  )
    #res.hist()

    plt.xlabel('Attended [hours]')
    plt.ylabel('#participants')

    #plt.show()
    plt.savefig('plots/timeHist.png')
    plt.close()


def printStats(df):
    df = df[df['mins'] >= 15]
    df['hours'] = 1./60.* df['mins']
    res = df.groupby('name')['hours'].agg(np.sum)

    nPart = len(res)
    totTime = res.sum()

    print('nPart: ',nPart)
    print('totTime: ',int(round(totTime,0)), 'PersonHours')
    print('timeAvg: ', round(totTime/nPart,1),'hours')
    print('timeMed: ', round(res.median(),1),'hours')
    
    #plt.hist(res, bins=np.arange(0,25, 0.25)  )
    ##res.hist()

    #plt.xlabel('Attended [hours]')
    #plt.ylabel('#participants')

    ##plt.show()
    #plt.savefig('plots/timeHist.png')
    #plt.close()





def plotDurationsCum(df):
    df['hours'] = df['mins']/60.
    res = df.groupby('name')['hours'].agg(np.sum)

    pd.set_option('display.max_rows', None)

    xx = np.arange(0,25, 0.1)
    vals = []

    for x in xx:
        vals.append(len(res[res > x]))

    plt.plot(xx, vals)
    plt.xlabel('At least attended [hours]')
    plt.ylabel('#participants')
    #plt.show()
    plt.savefig('plots/timeHistCum.png')
    plt.close()


def plotDaysVisited(df):
    df = df[df['mins'] >= 15]

    res = df.groupby('name')['day'].agg(np.unique).agg(np.size)

    hist = np.histogram(res, bins = np.arange(0.5,5.5,1) )

    plt.bar(["1","2","3","4"], hist[0])

    plt.xlabel('#attended days')
    plt.ylabel('#participants')

    #plt.show()
    plt.savefig('plots/daysVisited.png')
    plt.close()



import scipy
import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    #print(idx)

    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return idx, corr_array.iloc[idx, :].T.iloc[idx, :]
    return idx, corr_array[idx, :][:, idx]





def SessionsCorr(df):
    df = df[df['mins'] >= 15]
    
    sess =  df['session'].unique()
    sess = np.sort(sess)


    corrs = np.zeros(shape=(len(sess), len(sess)))

    res = df.groupby('name').agg(np.unique)['session'].tolist()
    for i1,s1 in enumerate(sess):
        for i2,s2 in enumerate(sess):
            n12a = n12o =0
            n1=n2= 0
            for a in res:
                n1   += (s1 in a)
                n2   += (s2 in a)
                n12a += (s1 in a) and (s2 in a)
                n12o += (s1 in a) or (s2 in a)

            #corrs[i1][i2] = int(round(100 * n12a / n12o))
            corrs[i1][i2] = int(round(100 * n12a / min(n1,n2)))


    #idx, corrs = cluster_corr(corrs)
    #sessN =  [sess[i] for i in idx]

    for i in range(len(sess)):
        corrs[i,i] = 0

    plt.figure(figsize=(10, 9))
    plt.imshow(corrs)

    plt.xticks(np.arange(len(sess)), sess, rotation=90)
    plt.yticks(np.arange(len(sess)), sess)

    # Loop over data dimensions and create text annotations.
    for i in range(len(sess)):
        for j in range(len(sess)):
            if i == j: continue
            plt.text(j, i, int(corrs[i, j]),
                ha="center", va="center", color="w")


    plt.subplots_adjust(left=0.14, right=0.9, top=0.9, bottom=0.2)

    #plt.show()
    #print(corrs)
    plt.savefig('plots/SessionsCorr.png')
    plt.close()


def SessionsIsol(df):
    # Fraction of participants who visited only the particular session
    df = df[df['mins'] >= 15]
    
    sess =  df['session'].unique()
    sess = np.sort(sess)[::-1]


    corrs = np.zeros(shape=(len(sess), len(sess)))

    res = df.groupby('name').agg(np.unique)['session'].tolist()

    isol = []

    for s1 in sess:
        nIsol = 0
        nTot = 0
        for a in res:
            if s1 in a:
                nIsol += isinstance(a,str)
                nTot  += 1
        isol.append(nIsol / nTot * 100)

    plt.barh(sess, isol)
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    plt.xlabel('Isolation of participants [%]')
    #plt.show()
    plt.savefig('plots/SessionsIsolation.png')
    plt.close()


def plotSessionsGender(df):
    sess = df['session'].unique()
    sess = np.sort(sess)[::-1]
    vals = []

    namesDict = {}
    fNames = open('namesAll')
    for l in fNames:
        if 'null' in l: continue
        l = l.split(' ')
        l[2] = float(l[2])
        if l[1] == 'female': l[2] = 1 - l[2]
        #print(l[0], l[2])
        namesDict[l[0]] = l[2]

    for s in sess:
        names = df[(df['session'] == s) & (df['mins'] >= 15)]['name'].unique()

        nKnown = 0
        isMale = 0
        genderL = []
        for n in names:
            nameF = ""
            for i in range(len(n)):
                if n[i].isalpha():
                    nameF += n[i]
                else:
                    break
            if nameF in namesDict:
                nKnown += 1
                isMale += namesDict[nameF]
                genderL.append(namesDict[nameF])

        p = sum(genderL)/len(genderL)
        for i in range(10):
            gNow = 0
            for m in genderL:
                gNow += m*p / (m*p + (1-m)*(1-p))
            p = gNow/len(genderL)

        vals.append((1-p)*100)


    plt.barh(sess, vals)
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    plt.xlabel('Female fraction [%]')
    #plt.show()
    plt.savefig('plots/SessionsGender.png')
    plt.close()


def analyze():
    df = loadAll()
    dfPlen = loadPlenary()

    #printStats(df)
    plotSessionsGender(df)
    plotDaysTotal(df, dfPlen)
    plotSessionsTotal(df)
    plotDurations(df)
    plotDurationsCum(df)

    plotDaysVisited(df)

    SessionsCorr(df)
    SessionsIsol(df)


analyze()

