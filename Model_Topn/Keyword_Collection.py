# -*- coding: utf-8 -*-
"""
Created on April ~ September. 2019

        for KT IT Solution Day

@author: SungJun Park, @KT
"""

from konlpy.tag import Okt
from collections import Counter
from datetime import date
import os

os.chdir("/Users/sungjunpark/POC_BizJarvis_2.0_TA/Data")

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split(',') for line in f.read().splitlines()]
        data = data[1:]  # header 제외 #
    return data

def read_stopwords(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split(',') for line in f.read().splitlines()]
    return data

stopwords_list = read_stopwords('stop_words.csv')

#poc_data = read_data('poc_train_data.csv')

pos_tagger = Okt()

def noun_select(sentence):
    return pos_tagger.nouns(sentence)

'''
def noun_select(sentence):
    morph = pos_tagger.pos(sentence)
    noun_adj_list = []
    
    if tag in ['Noun', 'Adjective']:
        noun_adj_list.append(word)
    return noun_adj_list
'''
#tokens = [noun_select(row[4]) for row in poc_data]


def get_tags(text, ntags = 20):
    collector = Okt()
    # konlpy의 Okt객체
    #nouns = collector.nouns(text)

    # nouns 함수를 통해서 text 에서 명사만 분리/추출
    element_list = []
    for element in text:
        element_list.extend(element)

    #list를 extend로 나열해준다.
    stopwords_result = []
    for w in stopwords_list:
        stopwords_result.extend(w)

    result_sentence = []
    for w in element_list:
        if w not in stopwords_result:
            result_sentence.append(w)

    #result_sentence = [w for w in element_list if not w in stopwords_list]

    count = Counter(result_sentence)

    # Counter 객체를 생성하고 참조변수 nouns 할당
    return_list = [] #명사 빈도수 저장할 변수
    for n, c in count.most_common(ntags):
        temp = {'tag': n, 'count': c}
        return_list.append(temp)

    # most_common 메소드는 정수를 입력받아 객체 안의 명사중 빈도수
    # 큰 명사부터 순서대로 입력받은 정수 갯수만큼 저장되어 있는 객체 반환
    # 명사와 사용된 갯수를 return_list에 저장합니다.
    return return_list

def main():
    #text_file_name = "poc_train_data.csv"
    # 분석할 파일

    noun_count = 20
    # 최대 많은 빈도수 부터 10개 명사 추출
    output_file_name = "topn_result.csv"
    # topn_result.csv 에 저장
    #open_text_file = open(text_file_name, 'r', -1, "utf-8")
    poc_data = read_data('poc_train_data2.csv')
    print(poc_data[0])
    tokens = [noun_select(row[4]) for row in poc_data]
    print(tokens)


    # 분석할 파일을 open
    #text = open_text_file.read()  # 파일을 읽습니다.
    tags = get_tags(tokens, noun_count)  # get_tags 함수 실행
    #open_text_file.close()  # 파일 close
    open_output_file = open(output_file_name, 'w', encoding='euc-kr')
    # 결과로 쓰일 topn_result.csv 열기
    for tag in tags:
        noun = tag['tag']
        count = tag['count']
        open_output_file.write('{}, {}\n'.format(noun, count))
    # 결과 저장
    open_output_file.close()


if __name__ == '__main__':
    main()