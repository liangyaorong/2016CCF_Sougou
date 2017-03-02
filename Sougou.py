#coding:utf-8

import jieba
from scipy.sparse import hstack
import numpy as np
import math
from scipy.sparse import csr_matrix, spdiags
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer,TfidfVectorizer
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,chi2,f_classif,mutual_info_classif
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier





def clean_train():
    fr = open('./2W.TRAIN')
    content = fr.readlines()
    fr.close()
    data = []
    for i in content:
        line = i.strip().split('\t')
        data.append(line)
    data[0][0] = data[0][0][3:]
    fr = open('./train2.csv','w')
    for i in data:
        fr.write(i[0]+'\t')
        fr.write(i[1] + '\t')
        fr.write(i[2] + '\t')
        fr.write(i[3] + '\t')
        for j in i[4:]:
            fr.write(j)
        fr.write('\n')
    fr.close()


def clean_test():
    fr = open('./2W.TEST')
    content = fr.readlines()
    fr.close()
    data = []
    for i in content:
        line = i.strip().split('\t')
        data.append(line)
    data[0][0] = data[0][0][3:]
    fr = open('./test2.csv','w')
    for i in data:
        fr.write(i[0]+'\t')
        for j in i[1:]:
            fr.write(j)
        fr.write('\n')
    fr.close()


def no_zero_data(index):
    fr = open('./train2.csv')
    content = fr.readlines()
    fr.close()
    content = [i.strip().split('\t') for i in content]
    new_content = []
    for i in content:
        if i[int(index)]!='0':
            new_content.append(i)

    fr = open('./%s_no_0_train2.csv'%index,'w')
    for i in new_content:
        fr.write(i[0] + '\t')
        fr.write(i[1] + '\t')
        fr.write(i[2] + '\t')
        fr.write(i[3] + '\t')
        fr.write(i[4]+'\n')
    fr.close()

#-------------------------------------------------------------------------------------------------------------------

def get_data(filename):
    fr = open(filename)
    data = fr.readlines()
    data = [i.strip().split('\t') for i in data]
    return data

def cut_words_and_write(data,filename):
    words_list = []
    for i in data:
        words_list.append(' '.join(jieba.cut(i[-1])).split())
    fr = open(filename,'w')
    for i in words_list:
        for j in i:
            fr.write(j.encode("utf-8")+'\t')
        fr.write('\n')
    fr.close()


def get_word_list(filename):#从分好的词中取词向量
    fr = open(filename)
    words_list = fr.readlines()
    fr.close()
    return words_list

def get_class_list(filename,index):#index为data中列标
    fr = open(filename)
    content = fr.readlines()
    fr.close()
    label = [int(i.strip().split('\t')[index]) for i in content]
    return label

def get_test_user_list():
    fr = open('./test2.csv')
    content = fr.readlines()
    name_list = [i.strip().split('\t')[0] for i in content]
    return name_list

def get_ans_list(filename):
    fr = open(filename)
    content = fr.readlines()
    name_list = [i.strip() for i in content]
    return name_list


def get_stop_words():
    fr = open('stop_words.txt')
    content = fr.readlines()
    stop_words = [i.strip() for i in content]
    stop_words[0] = stop_words[0][3:]
    return stop_words

def get_character_matrix(filename):#从写好的特征矩阵文本中读取矩阵
    fr = open(filename)
    matrix = [map(float,i.strip().split(' ')) for i in fr.readlines()]
    return csr_matrix(matrix)



def get_nonzero_index(index):
    index_list = []
    fr = open('./train2.csv')
    content = fr.readlines()
    fr.close()
    content = [i.strip().split('\t') for i in content]
    for i in range(len(content)):
        if content[i][int(index)]!='0':
            index_list.append(i)
    return index_list

#--------------------------------------------------------------------------

#组间方差
class OutVarTransformer(object):
    def __init__(self,col=None, row = None, weight=[]):
        self.row = row
        self.col = col
        self.weight = weight

    def _outvar(self, X, index_dict):
        Tf = []
        X2 = X.todense()
        for c in index_dict.keys():
            Tf.append(np.mean(X2[index_dict[c]]))
        return np.var(Tf)

    def fit(self, sparse_matrix, classify):
        self.row, self.col = sparse_matrix.shape
        cla = set(classify)
        index_dict = {}
        for c in cla:
            index_dict[c] = []

        for index in range(self.row):
            index_dict[classify[index]].append(index)
        for i in range(self.col):
            weig = self._outvar(sparse_matrix[:, i], index_dict)
            self.weight.append(weig)
        self._invar_diag = spdiags(self.weight, diags=0, m=self.col, n=self.col, format='csr')
        return self

    def transfrom(self, sparse_matrix):
        sparse_matrix = sparse_matrix * self._invar_diag
        return sparse_matrix



# 入口----------------------------------------------------
if __name__ == "__main__":

#--------前期处理-----------------------------------------------------------------------------------------------

    # clean_train()#将搜索内容合并
    # clean_test()

    # no_zero_data('3')#获取非0数据并写入%s_no_0_train2.csv
    # jieba.load_userdict('./usr_dict.txt')
    # data = get_data('D:/3_no_0_train2.csv')#获取合并后的数据（全部）
    # cut_words_and_write(data,'D:/3_cut_words.csv')#将切分好的词向量写入cut_words.csv，方便读取





    # #后面的要读进内存中处理

#------------调试--------------------------------------------------------------------------------------------
    # train_word = get_word_list('./3_cut_words.csv')#将非0的对应搜索文本读入
    # classify = get_class_list('./3_no_0_train2.csv',3)#将非0的对应标签读入
    #
    # # m = len(train_word)
    # # all_theme_matrix = get_character_matrix('D:/train_word_matrix.txt')
    # # nonzero_theme_matrix = all_theme_matrix[get_nonzero_index(1)]
    #
    # test_size = 0.2
    # # #主题特征（已经count）
    # # train_theme_matrix = nonzero_theme_matrix[0:int(math.floor(m * test_size)), :]
    # # test_theme_matrix = nonzero_theme_matrix[int(math.floor(m * test_size)):, :]
    #
    # #划分训练集与测试集（word）
    # X_train, X_test, y_train, y_test = train_test_split(train_word, classify, test_size=test_size)
    #
    # stop_words = get_stop_words()
    #
    # #将词转化为计数矩阵
    # vectorizer = TfidfVectorizer(stop_words = stop_words)#设置停用词
    # counted_train_data = vectorizer.fit_transform(X_train)
    # counted_test_data = vectorizer.transform(X_test)
    #
    # # 特征选取
    # selector = SelectKBest(chi2, k='all')
    # selected_train_data = selector.fit_transform(counted_train_data, y_train)
    # selected_test_data = selector.transform(counted_test_data)
    #
    #
    # # #将主题特征与计数矩阵合并
    # # train_data = hstack([train_word_matrix,train_theme_matrix])
    # # test_data = hstack([test_word_matrix,test_theme_matrix])
    # #
    # # #tf-idf
    # # transformer = TfidfTransformer()
    # # tfidf_train_data = transformer.fit_transform(counted_train_data)
    # # tfidf_test_data = transformer.transform(counted_test_data)
    #
    # # #方差系数加权（组间总体方差都不好）
    # # print 'start cv transform'
    # # vartansformer = OutVarTransformer()
    # # vartansformer.fit(selected_train_data,y_train)#用count_matrix训练
    # # var_train_data = vartansformer.transfrom(tfidf_train_data)
    # # var_test_data = vartansformer.transfrom(tfidf_test_data)
    # # print 'finished cv transform'
    #
    # clf1 = SVC(kernel='linear', C=1, tol=1e-4, max_iter =10000, random_state=0)
    # clf2 = LinearSVC(C=0.5, tol=1e-4, penalty='l2', max_iter=10000)
    # clf3 = BernoulliNB(alpha=0.01)
    # clf4 = MultinomialNB(alpha=0.01)
    # clf5 = LogisticRegression(C = 10)
    # # # parameters = {'C': [0.1, 0.5, 1, 10, 100,1000]}
    # # # eclf = GridSearchCV(clf4, parameters, cv=3)
    #
    # # eclf = clf1
    # eclf = VotingClassifier(estimators=[('MNB',clf4),('Lsvc',clf2),('BNB',clf3)], voting='hard', n_jobs=-1)
    # eclf.fit(selected_train_data, y_train)
    # # # print eclf.best_params_
    # pred = eclf.predict(selected_test_data)
    # print np.mean(y_test==pred)
    # # print classification_report(y_test, pred)



#------出答案--------------------------------------------------------------------------------------------------------
    for j in [1, 2, 3]:
        #获取文本
        train_word = get_word_list('./%s_cut_words.csv' % j)#将非0的对应搜索文本读入
        test_word = get_word_list('./cut_test_words.csv')
        classify = get_class_list('./%s_no_0_train2.csv' % j, j)#将非0的对应标签读入

        # #获取特征矩阵
        # all_train_theme_matrix = get_character_matrix('./train_word_matrix.txt')
        # train_theme_matrix = all_train_theme_matrix[get_nonzero_index(j)]
        # test_theme_matrix = get_character_matrix('./test_word_matrix.txt')

        stop_words = get_stop_words()

        #将文本转化为词矩阵
        vectorizer = TfidfVectorizer(stop_words=stop_words)  # 设置停用词
        counted_train_data = vectorizer.fit_transform(train_word)
        counted_test_data = vectorizer.transform(test_word)

        # #将词矩阵与特征矩阵合并
        # train_data = hstack([train_word_matrix])
        # test_data = hstack([test_word_matrix])


        #tf-idf
        transformer = TfidfTransformer()
        tfidf_train_data = transformer.fit_transform(counted_train_data)
        tfidf_test_data = transformer.transform(counted_test_data)

        #特征选取
        selector = SelectKBest(chi2, k=200000）
        selected_train_data = selector.fit_transform(tfidf_train_data,classify)
        selected_test_data = selector.transform(tfidf_test_data)



        # # 方差加权
        # print 'start var transform'
        # vartansformer = OutVarTransformer()
        # vartansformer.fit(selected_train_data,classify)  # 用count_matrix训练
        # var_train_data = vartansformer.transfrom(tfidf_train_data)
        # var_test_data = vartansformer.transfrom(tfidf_test_data)
        # print 'finished var transform'


        clf1 = SVC(kernel='linear', C=1, tol=1e-4, max_iter =10000, random_state=0)
        clf2 = LinearSVC(C=0.5, tol=1e-4, max_iter=10000)
        clf3 = BernoulliNB(alpha=0.012)
        clf4 = MultinomialNB(alpha=0.012)
        clf5 = LogisticRegression(C = 10)

        # eclf = clf1
        eclf = VotingClassifier(estimators=[('lsvc', clf2),('MNB', clf4),('BNB', clf3)], voting='hard', n_jobs=-1)
        eclf.fit(counted_train_data, np.array(classify))
        pred = eclf.predict(counted_test_data)

        fr = open('./ans_%s.csv' % j, 'w')
        for i in pred:
            fr.write(str(i)+'\n')
        fr.close()

    user_list = get_test_user_list()
    ans1 = get_ans_list('./ans_1.csv')
    ans2 = get_ans_list('./ans_2.csv')
    ans3 = get_ans_list('./ans_3.csv')
    fr = open('./final_ans.csv','w')
    for i in range(len(user_list)):
        fr.write(user_list[i].encode("gbk")+' ')
        fr.write(ans1[i].encode("gbk") + ' ')
        fr.write(ans2[i].encode("gbk") + ' ')
        fr.write(ans3[i].encode("gbk") + '\n')
    fr.close()
