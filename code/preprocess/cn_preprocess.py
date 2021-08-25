from bs4 import BeautifulSoup
import re

def remove_html(text):
    """remove html label
    """
    return BeautifulSoup(text, "lxml").text

def remove_urls(text):
    """remove url
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def delete_useless_symbols(setence):
    """delete useless symbols
    """
    useless_symbols = [" ","\n",">","<","\"","‘","’"]
    for item in useless_symbols:
        setence = setence.replace(item,'')
    return setence

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def keep_chinese_only(setence):
    """keep chinese only
    """
    content_str = ''
    for i in setence:
        if is_chinese(i):
            content_str = content_str + ｉ
    return content_str

def read_stopwords(file_path):
    stopwords = []
    stopword_file = open(file_path)
    for line in stopword_file:
        stopwords.append(line.strip())
    # print(stopwords)
    return stopwords

def delete_stopwords(setence,stopwords):
    for item in stopwords:
        setence = setence.replace(item,'')
    return setence

def test():
    setence = "<bbl>fa我是填身份https:\\line.com不符大部分是爱,毒霸cabdbv擦边大V啊,阿VB大V吧差别大V此卡的才达到！，，、n\n\n"
    print("Original: "+setence)
    #setence = remove_html(setence)
    #print(setence)
    setence = remove_urls(setence)
    print("Remove url: "+setence)
    setence = delete_useless_symbols(setence)
    print("Remove useless symbol: "+setence)
    setence = keep_chinese_only(setence)
    print("Keep Chinese only: "+setence)
    stopwords = read_stopwords("./data/cn_stopword.txt")
    setence = delete_stopwords(setence,stopwords)
    print("Remove stop words: "+setence)

if __name__ == "__main__":
    test()
