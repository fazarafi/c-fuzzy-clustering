# baca file dataset kemudian ditulis kembali ke file
# preprocessing
# - remove atribute (data atribut 3)
# - missing value handling
# - nominal to numeric (hanya yang tidak nominal tidak diubah)
# - normalisasi dataset (z-score) yang merupakan numeric
# ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
import pandas as pd
from sklearn import preprocessing

# Ada masalah, nanti elemennya jadinya ada spasi depannya karena file-nya abis koma ada spasi
class Preprocess(object):
    def __init__(self,nameOfFile):
        self.dataset = pd.read_csv('dataset/' + nameOfFile,header = None)
        self.numberOfAtr = 12

    def printDataSet(self):
        print(self.dataset.head(5))
        # print(self.dataset.describe())

    def printHeader(self):
        print(self.dataset.columns.values)

    def NominalToNumeric(self):
        l_pre = preprocessing.LabelEncoder()
        self.dataset = self.dataset.apply(l_pre.fit_transform)
        # enc = preprocessing.OneHotEncoder()
        # enc.fit(self.dataset)                
        # onehotlabels = enc.transform(self.dataset).toarray()
        # print(onehotlabels)
    
    def saveToCSV(self):
        self.dataset.to_csv('CencusIncome.data.preprocessing.txt',header = None)

    def printAtribute(self,number):
        # TBD
        print(self.dataset[number].values)
    
    def printRow(self,row):
        print(self.dataset.values[row])

    def removeAtribute(self,Atribute):
        # TBD
        return 0

    def missingValueHandling(self):
        # TBD
        return 0

    def normalize(self,Atribute):
        # TBD
        return 0
    
p = Preprocess('CencusIncome.data.txt')
# p.NominalToNumeric()
# p.printDataSet()
# p.saveToCSV()
p.printAtribute(1)
# p.printRow(1)
# p.printHeader()