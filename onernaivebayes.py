import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def naivebayesclassify(instance, d, traindf):
    outcome_arr = list(traindf["Outcome"].to_numpy())
    yescount = outcome_arr.count(1)
    nocount = outcome_arr.count(0)
    yesprobs = []
    noprobs = []
    # Calculating P(X | C = "no"/0) and P(X | C = "yes"/1)
    keys = list(d.keys())
    for i in range(len(instance)):
        for j in range(5 * i, 5 * i + 5):
            if j % 5 == 4 and instance[i] == keys[j][1]: 
                yesprobs.append((d[keys[j]][1]) / (d[keys[j]][0] + d[keys[j]][1]))
                noprobs.append((d[keys[j]][0]) / (d[keys[j]][0] + d[keys[j]][1]))
            elif instance[i] >= keys[j][0] and instance[i] < keys[j][1]:
                yesprobs.append((d[keys[j]][1]) / (d[keys[j]][0] + d[keys[j]][1]))
                noprobs.append((d[keys[j]][0]) / (d[keys[j]][0] + d[keys[j]][1]))
    # Calculating P(C | X) = P(X | C) * P(C)
    yestotal = 1
    nototal = 1
    for i in range(len(yesprobs)):
        yestotal *= yesprobs[i]
        nototal *= noprobs[i]
    probyes = yestotal * (yescount / len(outcome_arr))
    probno = nototal * (nocount / len(outcome_arr))
    if(probyes > probno): 
        print("The Naive Bayes model calculated that the given instance has diabetes", probyes, ">", probno)
        return 1
    else: 
        print("The Naive Bayes model calculated that the given instance does not have diabetes", probno, ">", probyes)
        return 0


def onerclassify(instance, ruleset):
    names = [col for col in traindf.columns]
    names = names[:-1]
    targetvalue = instance[ruleset[1]]
    templist = list(ruleset[2].keys())
    outcome = ["negative", "positive"]
    for i in range(5):
        if i == 4:
            print("Based on attribute", names[ruleset[1]], "the model classified the instance as ", outcome[ruleset[2][templist[4]][2]])
            return ruleset[2][templist[4]][2]
        if targetvalue >= templist[i][0] and targetvalue < templist[i][1]:
            print("Based on attribute", names[ruleset[1]], "the model classified the instance as ", outcome[ruleset[2][templist[i]][2]])
            return ruleset[2][templist[i]][2]
    
     
def onerruleset(d):
    names = [col for col in traindf.columns]
    names = names[:-1]
    print("OneR Classifier Rule Set:")
    err = [] #[error rate] first is pregnancies and so on
    temp = [] #[bin, wrong, total] ex. ["Pregnancies", 3, 14]
    ind = 0
    for key in d.keys():
        if d[key][0] > d[key][1]:
            temp.append([names[ind // 5], d[key][1], d[key][0] + d[key][1], 0])
        else:
            temp.append([names[ind // 5], d[key][0], d[key][0] + d[key][1], 1])
        ind += 1
    # print(temp)
    minind = 9999
    minerr = 9999
    for i in range(8):
        temperr = (temp[5 * i][1] + temp[5 * i + 1][1] + temp[5 * i + 2][1] + temp[5 * i + 3][1] + temp[5 * i + 4][1]) / + (temp[5 * i][2] + temp[5 * i + 1][2] + temp[5 * i + 2][2] + temp[5 * i + 3][2] + temp[5*     i + 4][2])
        # print(i, "error:", temperr)
        err.append(temperr)
        if minerr > temperr:
            minerr = temperr
            minind = i
    templist = list(d.keys())
    newdict = dict()
    for i in range(5 ):
        newdict[templist[minind * 5 + i]] = temp[minind * 5 + i][1:]
    return [minerr, minind, newdict]

def freqdict(allbins):
    names = [col for col in traindf.columns]
    names = names[:-1]
    freqdict = dict()
    for bins in allbins:
        for bin in bins:
            freqdict[bin] = [0, 0] #nocount, #yescount
    for i in range(len(names)):
        array = traindf[names[i]].to_numpy()
        # print(allbins[i])
        for j in range(len(array)):
            outcome = int(traindf["Outcome"].values[j])
            b = False
            for k in range(4):
                if traindf[names[i]].values[j] >= allbins[i][k][0] and traindf[names[i]].values[j] < allbins[i][k][1]: 
                    freqdict[allbins[i][k]][outcome] += 1
                    b = True
            if b == False: freqdict[allbins[i][4]][outcome] += 1
    return freqdict
    
def binning(traindf): #equal-width binning by means into 5 bins
    names = [col for col in traindf.columns]
    names = names[:-1]
    allbins = []
    for name in names:
        array = traindf[name].to_numpy()
        min = np.min(array)
        max = np.max(array)
        increment = (max - min) / 5
        bin1 = (min, min + increment)
        bin2 = (min + increment, min + 2 * increment)
        bin3 = (min + 2 * increment, min + 3 * increment)
        bin4 = (min + 3 * increment, min + 4 * increment)
        bin5 = (min + 4 * increment, min + 5 * increment)
        allbins.append([bin1, bin2, bin3, bin4, bin5])
    return allbins

df = pd.read_csv('diabetes.csv')
names = [col for col in df.columns]
data = df.to_numpy()
X = data[:, :-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8, stratify=y)
xtraindf = pd.DataFrame(X_train, columns = names[:-1])
ytraindf = pd.DataFrame(y_train, columns = ["Outcome"])
xtestdf = pd.DataFrame(X_test, columns = names[:-1])
ytestdf = pd.DataFrame(y_test, columns = ["Outcome"])
train = pd.concat([xtraindf, ytraindf], axis = 1)  
test = pd.concat([xtestdf, ytestdf], axis = 1)
train.to_csv('train2.csv', index = False, header = True)
test.to_csv('test2.csv', index = False, header = True)
traindf = pd.read_csv('train2.csv')
testdf = pd.read_csv('test2.csv')
names = [col for col in traindf.columns]
allbins = binning(traindf)
dictionary = freqdict(allbins)
print(dictionary)
ruleset = onerruleset(dictionary)
print("The attribute used for OneR classification is", names[ruleset[1]], ", with an error of", ruleset[0], "with [wrong counts, total counts, and class] counts of:", ruleset[2])
onercorrect = 0
nbcorrect = 0
count = 0
outcome = ["No", "Yes"]
y_pred_or = []
y_pred_nb = []
y_test = []
for i in range(len(testdf)):
    temp = testdf.iloc[i].values
    y_test.append(temp[-1])
    print("For instance:", temp)
    oner = onerclassify(temp[:-1], ruleset)
    nb = naivebayesclassify(temp[:-1], dictionary, traindf)
    y_pred_or.append(oner)
    y_pred_nb.append(nb)
    if (oner == temp[-1]): 
        print("One R Classified Correctly" + " (Model said " + outcome[oner] + ")")
        onercorrect += 1
    else: print("One R Classified Incorrectly" + " (Model said " + outcome[oner] + ")")
    if (nb == temp[-1]): 
        print("NaiveBayes Classified Correctly" + " (Model said " + outcome[nb] + ")")
        nbcorrect += 1
    else: print("NaiveBayes Classified Incorrectly" + " (Model said " + outcome[nb] + ")")
# print("OneR Classifer Testing Accuracy:", (onercorrect/count))
# print("Naive Bayes Classifer Testing Accuracy:", (nbcorrect/count))
# print("OR", y_pred_or)
# print("NB", y_pred_nb)
print("Test", y_test)
print("OneR Classifier Accuracy:", accuracy_score(y_test, y_pred_or))
print("OneR Classifier Confusion Matrix:\n", confusion_matrix(y_test, y_pred_or))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Confusion Matrix:\n",confusion_matrix(y_test, y_pred_nb))