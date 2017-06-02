import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        # useClassProb False olursa class larin posterior olasiliklari esit kabul edilir.
        # true olursa egitim kumesindeki sayilara gore posterior olasiliklar belirlenir.
        useClassProb = False

        self.training_data = X.toarray() 
        self.training_labels = np.asarray(y)
        self.unique_labels = np.unique(self.training_labels)
        self.CP= np.zeros((len(self.unique_labels)))        
        self.Tpl=np.zeros( (len(self.unique_labels), len(self.training_data[0, :])))
        

        self.MEAN=np.zeros((len(self.unique_labels), len(self.training_data[0, :])))
        self.VAR=np.zeros((len(self.unique_labels), len(self.training_data[0, :])))
          
        snf = -1
        for i in self.unique_labels: # ornek icin bu dongu 2 kez calisir ama daha cok elemanli problemlere de uygundur
            snf += 1
            # ilgili sinifin elemanlari bulunuyor
            I=(self.training_labels == i)
            data=self.training_data[I] 
            self.CP[snf]=len(data[:,0])
            for j in range(0, len(data[0, :])): # tum farkli kelimemelr icin bu dongu doner
                self.Tpl[snf, j] = np.sum(data[:, j])
                self.MEAN[snf, j] = np.mean(data[:, j])
                self.VAR[snf, j] = np.var(data[:, j]) + 0.1  # varyns 0 olmasi durumunda olasiligi sifirlanmasini engellemek icin
            self.Tpl[snf, :] = 1+(self.Tpl[snf, :])/(np.sum(self.Tpl[snf, :]))
            
        if useClassProb==False:
            for i in range(0,len(self.unique_labels)):
                # false ise 2 class icin her iki class posterior unu 0.5 yapar
                self.CP[i]=1.0/len(self.unique_labels)
        else:
            xx=np.sum(self.CP)
            for i in range(0,len(self.unique_labels)):
                # true ise egitim kuseinde bulunan elemanlara gore agirlik ayarlar
                self.CP[i]=self.CP[i]/xx
    

    def predict(self, testing_data):
        testing_data=testing_data.toarray()
        labels = np.zeros( (len(testing_data[:, 0])))

        for i in range(0, len(testing_data[:, 0])):
             sonuc=0
             probsonuc=-1
             for snf in range(0, len(self.unique_labels)):
                 prob=np.ones(len(self.unique_labels))
                 prob[snf] = self.CP[snf]
                 for j in range(0, len(testing_data[0, :])):
                     feat=testing_data[i, j]
                     if feat>0:
                        prob[snf] = prob[snf]*1/np.sqrt(2*np.pi*self.VAR[snf, j])*np.exp(-(feat-self.MEAN[snf, j])*(feat-self.MEAN[snf, j])/(2*self.VAR[snf, j]))
                 if  prob[snf] > probsonuc:
                     probsonuc = prob[snf]
                     sonuc = snf
             labels[i] = self.unique_labels[sonuc]
        return labels
