#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
#sns.set()


def loadFile(fName):
    print('Loading ', fName)
    if not os.path.isfile(fName):
        return None
    with open(fName) as f:
        arr = []
        for ll in f:
            if 'Total Duration' in ll:
                continue
            if 'ucjf-zoom' in ll:
                continue
            ll = [x.strip() for x in ll.split(',')]
            arr.append([ll[0], ll[1], int(ll[2])])

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
        d = loadFile('Parallels/'+day+'/participants_'+s+'_'+day+'.csv')
        if d is not None:
            allDays[s] = d
    return allDays


def loadPlenary():
    dfAll = pd.DataFrame(columns=['name', 'mail', 'mins', 'day', 'session', 'type'])
    for i, d in  enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday']):
        data = loadFile('Plenaries/ICHEP2020_Plenary'+str(i+1)+'.csv')
        df = pd.DataFrame(data, columns=['name', 'mail', 'mins'])
        df['day'] = 'b'+d
        df['session'] = 'Plenary'
        df['type'] = 'Plenary'
        dfAll = dfAll.append(df)

    return dfAll


def loadPoster():
    dfAll = pd.DataFrame(columns=['name', 'mail', 'mins', 'day', 'session', 'type'])
    for f in ['Wednesday/participants_02_Neutrino_poster_Wednesday.csv', 'Wednesday/participants_03_BSM_posters_Wednesday.csv',
               'Thursday/participants_02_Neutrino_poster_Thursday.csv',
               'Friday/participants_02_Neutrino_Poster3_Friday.csv', 'Friday/participants_02_Neutrino_Poster4_Friday.csv']:

        data = loadFile('Posters/' + f)
        df = pd.DataFrame(data, columns=['name', 'mail', 'mins'])
        df['day'] = 'a' + f[0:f.find('/')]
        import re
        df['session'] = re.findall('0[0-9]_[a-zA-Z]*', f)[0]
        df['type'] = 'Poster'
        print('Loading poster', f, df.count)
        dfAll = dfAll.append(df)

    return dfAll


def loadParallel():

    dfAll = pd.DataFrame(columns=['name', 'mail', 'mins', 'day', 'session', 'type'])

    for day in ['Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        data = loadDay(day)

        for s in data:
            df = pd.DataFrame(data[s], columns=['name', 'mail', 'mins'])
            df['day'] = 'a'+day
            df['session'] = s
            df['type'] = 'Parallel'
            dfAll = dfAll.append(df)
            # print(df)

    dfAll['mins'] = np.clip(dfAll['mins'], a_min=0, a_max=60*6)

    return dfAll


def loadPanels():
    dfAll = pd.DataFrame(columns=['name', 'mail', 'mins', 'day', 'session', 'type'])
    for f in ['ERC_session', 'Plenary1_PanelA', 'Plenary1_PanelB', 'Plenary2_PanelA', 'Plenary2_PanelB', 'Plenary2_PanelC', 'Plenary3_PanelA', 'Plenary3_PanelB', 'Plenary3_PanelC']:

        day = 'Tuesday'
        if f != 'ERC_session':
            import re
            Days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
            day = Days[int(re.findall('[0-9]', f)[0])-1]


        data = loadFile('Panels/participants_'+f+'.csv')
        df = pd.DataFrame(data, columns=['name', 'mail', 'mins']) 
        df['day'] = 'b' + day
        df['session'] = f
        df['type'] = 'Panels'
        dfAll = dfAll.append(df)

    return dfAll


def loadAll():
    df1 = loadParallel()
    df2 = loadPoster()
    df3 = loadPlenary()
    df4 = loadPanels()
    
    dfAll = pd.DataFrame(columns=['name', 'mail', 'mins', 'day', 'session', 'type'])
    dfAll = dfAll.append(df1)
    dfAll = dfAll.append(df2)
    dfAll = dfAll.append(df3)
    dfAll = dfAll.append(df4)

    return dfAll


def plotDaysTotal(df):
    days = []
    vals = []
    for d in df['day'].unique():
        days.append(d)
        val = len(df[(df['day'] == d) & (df['mins'] >= 15)]['name'].unique())
        vals.append(val)

    days = [d[:] for d in days]
    days = ['Tue', 'Wed', 'Thu', 'Fri', ' Mon ', ' Tue ', ' Wed ', ' Thu ']
    plt.figure(figsize=(9, 3))
    plt.bar(days, vals)
    plt.plot([3.5,3.5], [0, 1300], '--r')

    plt.title('Total number of unique participants')
    plt.ylabel('#participants')
    #plt.show()
    plt.savefig('plots/DaysTotal.png')
    plt.close()


def plotSessionsTotal(df):
    sess = (df[df['type']=='Parallel']['session']).unique()
    sess = np.sort(sess)[::-1]
    vals = []
    for s in sess:
        #sess.append(s)
        val = len(df[(df['session'] == s) & (df['type'] == 'Parallel') & (df['mins'] >= 15)]['name'].unique())
        vals.append(val)


    plt.barh(sess, vals)
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    plt.xlabel('#participants')
    #plt.show()
    plt.savefig('plots/SessionsTotal.png')
    plt.close()


def plotTypesTotal(df):
    #types = (df['type']).unique()
    #types = np.sort(types)[::-1]
    types = ['Parallel', 'Poster', 'Plenary', 'Panels'][::-1]
    vals = []
    for t in types:
        #sess.append(s)
        val = len(df[(df['type'] == t) & (df['mins'] >= 15)]['name'].unique())
        vals.append(val)


    plt.barh(types, vals)
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)

    plt.plot([0,2400], [1.5, 1.5], '--r')

    plt.xlabel('#participants')
    #plt.show()
    plt.savefig('plots/TypesTotal.png')
    plt.close()


def plotDurations(df):
    df['hours'] = df['mins']/60.
    res = df.groupby('name')['hours'].agg(np.sum)

    plt.hist(res, bins=np.arange(0, 25, 0.25))
    #res.hist()

    plt.xlabel('Attended [hours]')
    plt.ylabel('#participants')
    plt.yscale('log')
    plt.xlim(0, 25)

    #plt.show()
    plt.savefig('plots/timeHist.png')
    plt.close()


def checkNames(df):
    df = df[df['mins'] >= 15]
    df['hours'] = 1./60. * df['mins']
    res = df.groupby('name')['hours'].agg(np.sum)

    nPart = len(res)

    print('Helenka')
    names = res.index.tolist()

    import sklearn.cluster
    import distance

    #words = "YOUR WORDS HERE".split(" ") #Replace this line
    words = names[0:100]
    words = np.asarray(words) #So that indexing with a list will work
    lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])
    print(lev_similarity)

    vals = []
    for idx, x in np.ndenumerate(lev_similarity):
        vals.append([x, idx])
      #print(idx, x)

    vals = sorted(vals, key=lambda s: s[0])
    for v in vals:
        print(v[0], words[v[1][0]],' , ',  words[v[1][1]])




def printStats(df):
    df = df[df['mins'] >= 15]
    df['hours'] = 1./60. * df['mins']
    res = df.groupby('name')['hours'].agg(np.sum)

    nPart = len(res)
    totTime = res.sum()

    print('nPart: ', nPart)
    print('totTime: ', int(round(totTime, 0)), 'PersonHours')
    print('totTime: ', round(totTime/(24*365.), 1), 'PersonYears')
    print('timeAvg: ', round(totTime/nPart, 1), 'hours')
    print('timeMed: ', round(res.median(), 1), 'hours')
    
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

    hist = np.histogram(res, bins = np.arange(0.5, 9.5, 1))

    plt.bar(["1", "2", "3", "4", "5", "6", "7", "8"], hist[0])

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





def SessionsCorr(df, selType='session'):
    df = df[df['mins'] >= 15]
    
    sess = df[selType].unique()
    sess = np.sort(sess)
    if selType == 'type':
        sess = ['Parallel', 'Poster', 'Plenary', 'Panels']


    corrs = np.zeros(shape=(len(sess), len(sess)))

    res = df.groupby('name').agg(np.unique)[selType].tolist()
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
        for j in range(len(sess)):
            if i <= j:
                corrs[i,j] = np.nan


    plt.figure(figsize=(10, 9))
    plt.imshow(corrs)

    plt.xticks(np.arange(len(sess)), sess, rotation=90)
    plt.yticks(np.arange(len(sess)), sess)

    # Loop over data dimensions and create text annotations.
    for i in range(len(sess)):
        for j in range(len(sess)):
            if math.isnan(corrs[i, j]): continue
            plt.text(j, i, int(corrs[i, j]),
                ha="center", va="center", color="w")


    plt.subplots_adjust(left=0.14, right=0.9, top=0.9, bottom=0.2)

    #plt.show()
    #print(corrs)
    plt.savefig('plots/'+selType+'Corr.png')
    plt.close()


def SessionsIsol(df, selType='session'):
    # Fraction of participants who visited only the particular session
    df = df[df['mins'] >= 15]
    
    sess =  df[selType].unique()
    sess = np.sort(sess)[::-1]


    corrs = np.zeros(shape=(len(sess), len(sess)))

    res = df.groupby('name').agg(np.unique)[selType].tolist()

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
    plt.savefig('plots/'+selType+'Isolation.png')
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
    plt.plot([50, 50], [-0.4, 16.4], '--r')
    plt.xlim(0,76)
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    plt.xlabel('Fraction of female participants [%]')
    #plt.show()
    plt.savefig('plots/SessionsGender.png')
    plt.close()

def plotSessionsGenderSpeakers(df):
    sess = df['session'].unique()
    sess = np.sort(sess)[::-1]

    vals = []
    for n in sess:
        with open('speakers/'+n+".gen", 'r') as file:
            data = file.read().replace('\n', '')
            data = '[' + data + ']'
            data = data.replace('null','None')
            #print(len(data.splitlines()))
            Names = eval(data)
            #print(data)
            #print(Names)

            genderL = []
            for i in Names:
                if i['gender'] != None:
                    #print(i['gender'], i['probability'])
                    prob = i['probability']
                    if i['gender'] == 'female' : prob = 1 - prob
                    genderL.append(prob)

            p = sum(genderL)/len(genderL)
            print(n,(1-p)*100, len(genderL))
            for i in range(10):
                gNow = 0
                for m in genderL:
                    gNow += m*p / (m*p + (1-m)*(1-p))
                p = gNow/len(genderL)

            res = sum([1 for x in genderL if x <= 0.5])/len(genderL)*100
            print(n,  res)
            vals.append(res)

    plt.barh(sess, vals)
    plt.plot([50,50], [-0.4, 16.4], '--r')
    plt.xlim(0,76)
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    plt.xlabel('Fraction of female speakers [%]')
    #plt.show()
    plt.savefig('plots/SessionsGenderSpeakers.png')
    plt.close()


def plotTotalGender(df):
    #From the participants
    namesDict = {}
    fNames = open('namesAll')
    for l in fNames:
        if 'null' in l: continue
        l = l.split(' ')
        l[2] = float(l[2])
        if l[1] == 'female': l[2] = 1 - l[2]
        #print(l[0], l[2])
        namesDict[l[0]] = l[2]

    names = df[(df['mins'] >= 15)]['name'].unique()

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

    fracPartic = (1-p)*100
    print(fracPartic)

    # From the speakers
    sess = df['session'].unique()
    sess = np.sort(sess)[::-1]

    genderL = []
    for n in sess:
        with open('speakers/'+n+".gen", 'r') as file:
            data = file.read().replace('\n', '')
            data = '['+ data + ']'
            data = data.replace('null','None')
            #print(len(data.splitlines()))
            Names = eval(data)
            #print(data)
            #print(Names)

            for i in Names:
                if i['gender'] != None:
                    #print(i['gender'], i['probability'])
                    prob = i['probability']
                    if i['gender'] == 'female' : prob = 1 - prob
                    genderL.append(prob)

    p = sum(genderL)/len(genderL)
    print(n,(1-p)*100, len(genderL))
    for i in range(10):
        gNow = 0
        for m in genderL:
            gNow += m*p / (m*p + (1-m)*(1-p))
        p = gNow/len(genderL)

    #fracSpeakers = sum([1 for x in genderL if x <= 0.5])/len(genderL)*100
    fracSpeakers = 100*(1-p)

    #print(fracSpeakers, 100*(1-p))

    lab  = ["From participants", "From speakers"]
    vals = [fracPartic, fracSpeakers]

    print('fracParticipants', fracPartic)
    print('fracSpeakrs', fracSpeakers)

    plt.figure(figsize=(7, 3))
    plt.barh(lab, vals)
    plt.plot([50,50], [-0.4, 1.4], '--r')
    plt.xlim(0,76)
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.14)
    plt.xlabel('Fraction of females [%]')
    #plt.show()
    plt.savefig('plots/TotalGender.png')
    plt.close()




def analyze():
    df = loadAll()

    printStats(df)
    #return
    #plotSessionsGenderSpeakers(df)

    #plotTotalGender(df)

    #checkNames(df)

    plotSessionsGender(df[df['type']=='Parallel'])
    plotDaysTotal(df)
    plotSessionsTotal(df)
    plotTypesTotal(df)
    plotDurations(df)
    plotDurationsCum(df)

    plotDaysVisited(df)

    SessionsCorr(df[df['type']=='Parallel'])
    SessionsCorr(df, 'type')
    SessionsIsol(df[df['type']=='Parallel'], 'session')
    SessionsIsol(df, 'type')


analyze()

