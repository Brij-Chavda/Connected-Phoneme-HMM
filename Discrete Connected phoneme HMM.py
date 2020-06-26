#!/usr/bin/env python
# coding: utf-8

# In[1381]:


#mfcc feature
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
Obs = {}
part = {}
n_state = 4
for j in range(1,2):
    if j == 0:
        n_phone = 4
    else:
        n_phone = 3
    for i in range(10):
        filename = "Data_Processed/Train_" + str(j) + "_Example_" + str(i+1) + ".wav"
        (Fs,data) = wav.read(filename)
        Obs[j,i] = mfcc(data,samplerate=Fs,winlen=0.025,winstep=0.01,numcep=13)
        divide = np.array_split(Obs[j,i],(n_phone * (n_state - 1)))
        count = 0
        for k in range((n_phone * n_state)):
            if ((k + 1) % 4) != 0:
                part[j,i,k] = divide[count]
                count = count + 1
            else:
                continue


# In[ ]:





# In[1382]:


part_wise = np.vstack((Obs[1,0],Obs[1,1]))
for i in range(2,10):
    part_wise = np.vstack((part_wise,np.array(Obs[1,i])))


# In[1383]:


#divide data into 32 clusters 
from sklearn.cluster import KMeans
import numpy as np
Obs_cluster = {}
j = 1
nstate = 4
nphone = 3

part = {}

#clusters = KMeans(n_clusters=32).fit(part_wise)
clusters110t = KMeans(n_clusters=32).fit(part_wise)
for i in range(10):
    Obs_cluster[j,i] = clusters110t.predict(Obs[j,i])
    devide_three = np.array_split(Obs_cluster[j,i],((nstate-1)*nphone))
    count = 0
    for k in range((nstate)*nphone):
        if (k + 1) % 4 != 0 :
            part[j,i,k] = devide_three[count]
            count = count + 1
        else:
            continue


# In[1384]:


#for speech recording zero or one
j = 1
n_phone = 3
n_state = 4
#initialize 
a = np.zeros(((n_phone*n_state),(n_phone*n_state)))
b = np.zeros((32,(n_phone*(n_state))))
n_element = np.zeros((n_phone*n_state))
for i in range(n_phone*n_state):
    if (i + 1) % 4 == 0:
        if i != ((n_phone * n_state) - 1):
            a[i, i + 1] = 1
        a[i, i] = 0
    else:
        for l in range(10):
            n_element[i] = n_element[i] + len(part[j,l,i])
        n_element[i] = n_element[i] - 10
        a[i, i + 1] = 10 / n_element[i]
        a[i, i] = 1 - a[i, i + 1]

for i in range(n_phone*n_state):
    for j in range(10):
        for k in range(32):
            if (i+1) % 4 != 0:
                for clu in part[1,j,i]:
                    if clu == k:
                        b[k,i] = b[k,i] + 1
        total = np.sum((b[:,i]),axis=0)
        b[:,i] = [b[l,i]/10 for l in range(32)]
pie = np.zeros(n_phone * n_state)
pie[0] = 1


# In[1385]:


for i in range(n_phone*n_state):
    for j in range(32):
        if b[j,i] == 0:
            b[j,i] = 1e-8


# In[1388]:


#connected phone HMM algorithm:
import math
class ConnectedphoneHMM:
    def __init__(self,Obs):
        self.Obs = Obs
        self.coef = {}
        self.alpha_dict = {}
        self.beta_dict = {}
        self.nstate = 12
        self.phone_nstate = 4
        self.n_GMM = 2
        self.si = {}
        self.delta_dict = {}
        self.gamma_dict = {}
        self.zeta_dict = {}
        self.gamma_dict = {}
        self.cls = 1
        self.n_phone = 3
        
    def alpha(self,t,Obs_time,q):
        def coef(alpha):
            total = np.sum(alpha)
            if total == 0:
                total = 1
            return (1/total)
                
        if (Obs_time,self.Obs_n,q) in self.alpha_dict:
            return self.alpha_dict[Obs_time,self.Obs_n,q]
        
        elif t == 1:
            al1 = np.zeros((self.phone_nstate,1))
            pie_phone = [1,0,0,0]
            b1 = self.b[self.Obs[self.cls,self.Obs_n][int(Obs_time-1)], :]
            al1 = np.multiply(pie_phone,b1[(q*4):((q+1)*4)])
            return al1
        
        else:
            temp_alpha = np.zeros((self.phone_nstate))
            cal_recursion = self.alpha(t-1,Obs_time-1,q)
            coeff = coef(cal_recursion)
            
            if (Obs_time-1,self.Obs_n,q) not in self.alpha_dict:
                self.alpha_dict[Obs_time-1,self.Obs_n,q] = cal_recursion
                
            if (Obs_time-1,self.Obs_n,q) not in self.coef:
                self.coef[Obs_time-1,self.Obs_n,q] = coeff
                
            scaled_alpha = np.multiply(cal_recursion, coeff)
            
            for (j,zj) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
                for (i,zi) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
                    if (j + 1) % 4 != 0:
                        temp_alpha[j] = temp_alpha[j] + (scaled_alpha[i]*self.a[zi][zj]*self.b[self.Obs[self.cls,self.Obs_n][int(Obs_time-1)],j])
                    else:
                        end = j + 1
                        start = j - 3
                        for (k,zk) in zip(range(start,start+end),range(q*4,(q+1)*4)):
                            temp_alpha[j] = temp_alpha[j] + (temp_alpha[k] * self.a[zk][zj])
            if t-1 == self.T-1:
                self.coef[Obs_time,self.Obs_n,q] = coef(temp_alpha)
        return temp_alpha
        
    def beta(self,t,Obs_time,q):
        if (Obs_time,self.Obs_n,q) in self.beta_dict:
            return self.beta_dict[Obs_time,self.Obs_n,q]
        
        elif t == self.T:
            return np.ones((self.phone_nstate))
        
        else:
            temp_beta = np.zeros((self.phone_nstate))
            cal_recursion = self.beta(t+1, Obs_time+1, q)
            
            if (Obs_time+1,self.Obs_n,q) not in self.beta_dict:
                self.beta_dict[Obs_time+1,self.Obs_n,q] = cal_recursion
            
            if (Obs_time+1,self.Obs_n,q) not in self.coef:
                print(self.coef)
            scaled_beta = np.multiply(self.coef[Obs_time+1,self.Obs_n,q],cal_recursion)
            
            for i in range(self.phone_nstate):
                if scaled_beta[i] > (2^31) - 1:
                    scaled_beta[i] = (2^28)
            for (j,zj) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
                for (i,zi) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
                    if (j + 1) % 4 != 0:
                        temp_beta[i] = temp_beta[i] + (self.a[zi][zj]*self.b[self.Obs[self.cls,self.Obs_n][int(Obs_time)+1],j]*scaled_beta[i])
                    else:
                        end = 4
                        start = 0
                        for (k,zk) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
                            temp_beta[i] = temp_beta[i] + (temp_beta[k] * self.a[zi][zk])
                            
        return temp_beta
    

    def delta(self,t):

        if (t, self.Obs_n) in self.delta_dict:
            return self.delta_dict[t, self.Obs_n],self.si[t, self.Obs_n]

        elif t == 1:
            b1 = np.log(self.b[self.Obs[self.cls,self.Obs_n][t-1],:])
            al1 = np.sum((pie,b1),axis=0)
            return al1,[0]*(self.phone_nstate*self.n_phone)

        else:

            cal,calsi = self.delta(t-1)
            if (t-1,self.Obs_n) not in self.delta_dict:
                self.delta_dict[t-1,self.Obs_n] = cal
            if (t-1,self.Obs_n) not in self.si:
                self.si[t-1,self.Obs_n] = calsi
            delta_t_1 = np.zeros((self.nstate))
            si_t_1 = np.zeros((self.nstate))
            
            
            for j in range(self.nstate):

                if (j + 1) % 4 != 0:
                    maxcal = (max([cal[i]+np.log(self.a[i][j]) for i in range(self.nstate) if np.log(self.a[i][j]) != np.inf or np.log(self.a[i][j]) != -np.inf]))
                    delta_t_1[j] = (maxcal+np.log(self.b[Obs_cluster[self.cls,self.Obs_n][t-1],j]))
                    index = np.argmax(np.asarray([cal[i] + np.log(self.a[i][j]) for i in range(self.nstate) if np.log(self.a[i][j]) != np.inf or np.log(self.a[i][j]) != -np.inf]))
                else:
                    maxcal = (max([cal[i] + np.log(self.a[i][j]) for i in range(self.nstate) if np.log(self.a[i][j]) != np.inf or np.log(self.a[i][j]) != -np.inf]))
                    delta_t_1[j] = (maxcal)
                    index = np.argmax(np.asarray([cal[i] + np.log(self.a[i][j]) for i in range(self.nstate) if np.log(self.a[i][j]) != np.inf or np.log(self.a[i][j]) != -np.inf]))  
                si_t_1[j] = index
            
            if t-1 == self.T-1:
                self.si[self.T,self.Obs_n] = si_t_1
                
        return delta_t_1,si_t_1

    def backtrack(self,si_T,T):
        t = T
        qt = []
        while t >= 1:
            temp = self.si[t,self.Obs_n]
            if temp is not None:
                si_T = temp[int(si_T)]
                qt.append(si_T)
                t = t - 1
            else:
                print('calculate delta first')
        return qt
    def viterbialignment(self):
        from collections import Counter
        alignment = {}
        alignment_count = {}
    
        for i in range(10):
            self.Obs_n = i
            self.T = len(self.Obs[self.cls,self.Obs_n]) - 1
            delta, si = self.delta(self.T)
            #print(self.si[self.T - 2,self.Obs_n])
            index = np.argmax(si)
            alignment[i] = self.backtrack(si[index],self.T)
            alignment_count[i] = Counter(alignment[i])
        #print(alignment[4])
        return alignment_count
    
    def gamma(self,t,Obs_time,q):
        if (Obs_time,self.Obs_n) in self.gamma_dict:
            return self.gamma_dict[Obs_time,self.Obs_n]

        al_t = self.alpha(int(t),int(Obs_time),q)
        bl_t = self.beta(int(t),int(Obs_time),q)
        denominator = np.dot(al_t,bl_t)
        
        if denominator == 0:
            print('gamma_den {},{},{}'.format(Obs_time,q,self.Obs_n))
            denominator = 1
        
        temp_gamma = np.zeros((self.phone_nstate))
        for i in range(self.phone_nstate):
            if ((i+1) % 4) != 0:
                temp_gamma[i] = (al_t[i] * bl_t[i]) / denominator
        
        if (Obs_time,self.Obs_n) not in self.gamma_dict:
            self.gamma_dict[Obs_time,self.Obs_n] = temp_gamma
            
        return temp_gamma
    
    def zeta(self,t,Obs_time,q):
        if (Obs_time,self.Obs_n) in self.zeta_dict:
            return self.zeta_dict[Obs_time,self.Obs_n]
        
        zhai = np.zeros((self.phone_nstate,self.phone_nstate))
        al = self.alpha(t,Obs_time,q)
        be = self.beta(t+1, Obs_time+1,q)
        total_den = 0
        
        for (j,zj) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
            for (i,zi) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
                if (j+1) % 4 != 0:
                    total_den = total_den + (al[i] * self.a[zi][zj] * self.b[self.Obs[self.cls,self.Obs_n][int(Obs_time)+1],j] * be[j])
                else:
                    total_den = total_den + ((al[i] * self.a[zi][zj]) * be[j])
        if total_den == 0:
            print('zhai_den {},{},{}'.format(Obs_time,q,self.Obs_n))
            total_den = 1
        for (j,zj) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
            for (i,zi) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
                if (j+1) % 4 != 0:
                    zhai[i][j] = (al[i] * self.a[zi][zj] * self.b[self.Obs[self.cls,self.Obs_n][int(Obs_time)+1],j] * be[j]) / total_den
                else:
                    zhai[i][j] = (al[i] * self.a[zi][zj] * be[j]) / total_den
        
        if (Obs_time,self.Obs_n) not in self.zeta_dict:
            self.zeta_dict[Obs_time,self.Obs_n] = zhai
            
        return zhai
    
    def continu_param(self,a,pie,b,align_index):
        def coef(alpha):
            total = np.sum(alpha)
            if total == 0:
                total = 1
            return (1/total)
        
        def sum_al(x):
            return np.sum(x)
        
        def cal_gamma(t,Obs_time):
            gamma_val = self.gamma(t,Obs_time,q)
            return gamma_val
        
        self.a = a
        self.b = b
        self.pie = pie
        
        if align_index != 0:
            align_Obs = self.viterbialignment()

        else:
            align_Obs = {}
            for i in range(10):
                diction = {}
                for k in range(self.nstate):
                    self.Obs_n = i
                    if (k+1) % 4 != 0:
                        diction[k] = len(self.Obs[self.cls,self.Obs_n])/self.nstate
                    else:
                        diction[k] = 0
                align_Obs[i] = diction
        
        
        re_pie = np.zeros((self.nstate))
        re_a = np.zeros((self.nstate,self.nstate))
        re_b = np.zeros((32,self.nstate))
        phone_align = np.zeros((10,self.n_phone))
        for k in range(10):
            for i in range(self.n_phone):
                phone_align[k,i] = np.sum([align_Obs[k][(4*i)], align_Obs[k][(4*i) + 1], align_Obs[k][(4*i) + 2], align_Obs[k][(4*i) + 3]])
        
        for q in range(self.n_phone):
            

            #reestimation of a and pie
            sum_zhai = np.zeros((10, self.phone_nstate, self.phone_nstate))
            sum_gamma = np.zeros((10, self.phone_nstate))
            for exp in range(10):
                self.Obs_n = exp
            
                for h in range(q+1):
                    if h == 0:
                        start = 0
                        end = (int(phone_align[exp,h]))
                    else:
                        start = start + (int(phone_align[exp,h-1]))
                        end = end + (int(phone_align[exp,h]))
                end = end - 1
                self.T = (int(phone_align[exp,q] - 1))
                alpha_T = self.alpha(self.T,end,q)
                for (t,Obs_time) in zip(range(1,self.T+1),range(start,end)):
                    if t != self.T:
                        sum_zhai[exp, :, :] = sum_zhai[exp, :, :] + self.zeta(t,Obs_time,q)
                    sum_gamma[exp, :] = sum_gamma[exp, :] + cal_gamma(t,Obs_time)
                re_pie[q*4:(q+1)*4] = re_pie[q*4:(q+1)*4] + cal_gamma(1,start)
            

            total_zhai = np.zeros((self.phone_nstate,self.phone_nstate))
            total_gamma = np.zeros((self.phone_nstate))
            for exp in range(10):
                for i in range(self.phone_nstate):
                    for j in range(self.phone_nstate):
                        total_zhai[i,j] = total_zhai[i,j] + (sum_zhai[exp,i,j])
                    total_gamma[i] = total_gamma[i] + (sum_gamma[exp, i])
            
            for i in range(self.phone_nstate):
                if total_gamma[i] == 0:
                    total_gamma[i] = 1
            for (i,zi) in zip(range(q*4,(q+1)*4),range(self.phone_nstate)):
                for (j,zj) in zip(range(q*4,(q+1)*4),range(self.phone_nstate)):
                    re_a[i,j] = total_zhai[zi,zj] / total_gamma[zi]
            
            if q == self.n_phone - 1:
                total_coef = 0
                for exp in range(10):
                    self.Obs_n = exp
            
                    for h in range(q+1):
                        if h == 0:
                            start = 0
                            end = (int(phone_align[exp,h]))
                        else:
                            start = start + (int(phone_align[exp,h-1]))
                            end = end + (int(phone_align[exp,h]))
                    end = end - 1                    
                    self.T = (int(phone_align[exp,q] - 1))
                    
                    for Obs_time in range(start + 1,(end) + 1):
                        if (Obs_time,self.Obs_n,q) in self.coef:
                            total_coef = total_coef + math.log(self.coef[Obs_time,self.Obs_n,q])
                    
                likelihood = total_coef / 10
            
            re_b_num = np.zeros((32,self.phone_nstate))
            
            re_b_den= np.zeros((self.phone_nstate))
            total_numa= np.zeros((32,10,self.phone_nstate))
            c_product = 1
            for exp in range(10):
                self.Obs_n = exp
            
                for h in range(q+1):
                    if h == 0:
                        start = 0
                        end = (int(phone_align[exp,h]))
                    else:
                        start = start + (int(phone_align[exp,h-1]))
                        end = end + (int(phone_align[exp,h]))
                
                end = end - 1
                self.T = int(phone_align[exp,q] - 1)
                for (t,Obs_time) in zip(range(1,self.T+1),range(start,end)):
                    k = self.Obs[self.cls,self.Obs_n][int(Obs_time)]
                   
                    total_numa[k,exp,:] = self.gamma(t,Obs_time,q) + total_numa[k,exp,:]

                    for j in range(self.phone_nstate):
                        if j != 3:
                            for k in range(32):
                                re_b_num[k,j] = re_b_num[k,j] +(total_numa[k,exp,j])
                            re_b_den[j] = re_b_den[j] + (sum_gamma[exp,j])
                    
                        if re_b_den[j] == 0:
                            re_b_den[j] = 1
            
            for (j,zj) in zip(range(self.phone_nstate),range(q*4,(q+1)*4)):
                for i in range(32):
                    re_b[i,zj] = re_b_num[i,j] / re_b_den[j]
        print(re_pie)
        return re_a,re_pie,re_b,likelihood
    
    def five_iteration(self,a,pie,b):
        for i in range(10):
            a,pie,b,likelihood = self.continu_param(a,pie,b,i)
            pie = [i/10 for i in pie]
            self.coef = {}
            self.alpha_dict = {}
            self.beta_dict = {}
            self.si = {}
            self.delta_dict = {}
            self.gamma_dict = {}
            self.zeta_dict = {}
            self.gamma_dict = {}
            pie = np.zeros(self.nstate)
            pie[0] = 1
            print(likelihood)
            print(pie)
            for i in range(4,(self.n_phone * self.phone_nstate) + 1,4):
                a[i-1,i-1] = 0
                if i != (self.n_phone * self.phone_nstate):
                    a[i-1,i] = 1
            
            for i in range(32):
                for j in range(self.nstate):
                    if b[i,j] == 0:
                        b[i,j] = 1e-8
                    if math.isnan(b[i,j]):
                        b[i,j] = 0
           
        return a,pie,b,likelihood

        


# In[1389]:


import math
phone_hmm = ConnectedphoneHMM(Obs_cluster)
rea,repie,reb,likelihood = phone_hmm.five_iteration(a,pie,b)


# In[1390]:


class Test:
    def __init__(self,test_obs):
        self.Obs = test_obs
        self.test_coef = {}
        self.alpha_dict = {}
        self.nstate = 12
        
    def test_alpha(self,t):
        def coef(alpha):
            total = np.sum(alpha)
            if total == 0:
                total = 1
            return (1/total)
                
        if (t) in self.alpha_dict:
            return self.alpha_dict[t]
        
        elif t == 1:
            al1 = np.zeros((self.nstate,1))
            b1 = self.b[self.test[t-1], :]
            al1 = np.multiply(self.pie,b1)
            return al1
        
        else:
            temp_alpha = np.zeros((self.nstate))
            cal_recursion = self.test_alpha(t-1)
            coeff = coef(cal_recursion)
            
            if (t-1) not in self.alpha_dict:
                self.alpha_dict[t-1] = cal_recursion
                
            if (t-1) not in self.test_coef:
                self.test_coef[t-1] = coeff
                
            scaled_alpha = np.multiply(cal_recursion, coeff)
            
            for (j) in (range(self.nstate)):
                for (i) in (range(self.nstate)):
                    if (j + 1) % 4 != 0:
                        temp_alpha[j] = temp_alpha[j] + (scaled_alpha[i]*self.a[i][j]*self.b[self.test[t-1],j])
                    else:
                        if j!= (self.nstate)-1:
                            end = j + 1
                            start = j - 3

                            for (k) in (start,end):
                                temp_alpha[j] = temp_alpha[j] + (temp_alpha[k] * self.a[k][j])
            if t-1 == self.T-1:
                self.test_coef[self.T] = coef(temp_alpha)
        return temp_alpha
    
    def testing(self,a,b,pie,exp):
        self.a = a
        self.b = b
        self.pie = pie
        self.test = self.Obs[exp]
        self.T = len(self.test) - 1
        alpha_T = self.test_alpha(self.T)
        total_coef = 0
        for Obs_time in range(1,self.T+1):
            total_coef = total_coef + math.log(self.test_coef[Obs_time])
        return total_coef


# In[1392]:


test_Obs = {}
for i in range(ord('a'),ord('k')):
    filename = "Data_Processed/Test_Sample_" + chr(i) + ".wav"
    (Fs,data) = wav.read(filename)
    test_Obs[chr(i)] = clusters110t.predict(mfcc(data,samplerate=Fs,winlen=0.025,winstep=0.01,numcep=13))
test = Test(test_Obs)     
for i in range(ord('a'),ord('k')):
    likelihood = test.testing(rea,reb,repie,chr(i))
    print(likelihood)
#zero data_processed pie[0] new


# In[1311]:


test_Obs = {}
for i in range(ord('a'),ord('k')):
    filename = "Data_Processed/Test_Sample_" + chr(i) + ".wav"
    (Fs,data) = wav.read(filename)
    test_Obs[chr(i)] = clusters110t.predict(mfcc(data,samplerate=Fs,winlen=0.025,winstep=0.01,numcep=13))
test = Test(test_Obs)     
for i in range(ord('a'),ord('k')):
    likelihood = test.testing(rea,reb,repie,chr(i))
    print(likelihood)
#zero data_processed pie[0] new
