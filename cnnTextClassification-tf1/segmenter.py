
"""
本模块为分词器类：
    1 构造各种分词器，对外提供cut_words方法
    2 实现分词前后的预处理
"""
import os
import re

from pyhanlp import AhoCorasickDoubleArrayTrieSegment




def replace_url(message):
    """
    替代url 格式化
    """    
    message = re.sub(r'(http|ftp)[s]{0,1}\s{0,3}[:：]\s{0,3}(//|／／).*?(?=[\s\"\'）)\]}】》>;；,，。\n【［\[\u4e00-\u9fa5]|$)', '<urla>', message)
    message = re.sub(
        r'[?？a-zA-Z0-9/／.．\-]{0,30}[.．]c(om|[con])[/／][?？a-zA-Z0-9/.．／\-]{0,30}(?=[]\s\"\'）)}】》>;；,，。\n【［\[\u4e00-\u9fa5]|$)', '<urlb>', message)
    return message 

class Segmenter(object):
    def __init__(self, stop_words_path, seg_dict_path):
        self.stop_words = set(line.strip() for line in open(stop_words_path, "r", encoding="utf8").readlines())
        self.platform_keywords = {"注册", "登录", "登陆", "验证码", "已到期"}
        self.segmenter = self._init_segmenter(seg_dict_path)

    def _init_segmenter(self, seg_dict_path):
        segmenter = AhoCorasickDoubleArrayTrieSegment([seg_dict_path])  # 字典分词
        segmenter.enablePartOfSpeechTagging(True)
        return segmenter

    def cut_words(self, content):
        """ 一句话分词函数
        流程：输入文本 - 过滤文本字段 - 分词 - 过滤分词字段 - 输出
        :param content: 待分词句子
        :return: 分词结果，各分词间用空格分隔。返回值过滤了数值、英文单词、【】中所包含的平台信息

        """
        # 文本在分词前需要经过的过滤器
        content = self._filter_before_cut_words_(content) if str(content)!='nan' else ""

        # 过滤后为空字符串，则直接返回空字符串
        if not content:
            return ""

        # 字典极速匹配分词
        cuts = self.segmenter.seg(content)
        cuts = [term.word for term in cuts]
        # 分词在输出前需要经过的过滤器
        cut = self._filter_after_cut_words_(cuts)
        # 将列表转化为分隔符为空格的字符串返回
        return " ".join(x for x in cut)

    def text_format(self, message):
        url_regex = re.compile("(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#*]*[\w\-\@?^=%&/~\+#])?",
                               re.A)
        clean_patt = re.compile('^\"{3,6}|\"{3,6}$|\s')
        message = url_regex.sub(' <url> ', message)
        message = clean_patt.sub('', message)

        return message

    def _filter_before_cut_words_(self, content):
        """
        文本在分词前需要经过的过滤器
        :param content: 文本文本
        :return: 数据清洗后的文本文本
        """
        content = re.sub(r'\s|^\"{3,}|\"{3,}$', "", content)
        content = replace_url(content)
        num_regex = re.compile(r'-?[1-9]\d*(?:,\d+)*(?:\.\d+)?|(?:-?0\.)?\d+')
        # 数字统一替换
        content = re.sub(num_regex, '<num>', content)

        # 字符串所有的全角转半角
        content = self.stringQ2B(content)
        # 英文所有的字符转小写
        content = content.lower()
        # 过滤掉文本文本的借贷平台信息
        # content = self._platform_filter_(content)
        return content

    def _platform_filter_(self, content):
        """ 删除文本中的平台信息
        平台信息包含“注册|登录|登陆”等文本信息时，在删除平台信息的同时在相应位置添加“注册|登录|登陆验证码”信息

        :param content: 待处理文本
        :return: 删除平台信息后的文本

        For example:
            :param content = "【爱信钱包快捷登录】您的验证码为5780，有效期15分钟。"
            :return = "(登录)您的验证码为5780，有效期15分钟。"
        """
        # 匹配：【*】
        platform_info_list = re.findall(r"^【[^【】]*?】", content)
        keyword_patten = re.compile(r"(?:" + "|".join(x for x in self.platform_keywords) + ")")
        # 原文中所有【*】 替换为 (防过滤字典关键字)
        for i, platform_info in enumerate(platform_info_list):
            instead_info = "".join(re.findall(keyword_patten, platform_info))
            if len(instead_info) > 0:
                instead_info = "(" + instead_info + ")"
            content = content.replace(platform_info, instead_info)
        return content

    # 文本在分词后需要经过的过滤器
    def _filter_after_cut_words_(self, cut_words):
        """
        文本分词后，分词结果在输出前需要经过的过滤器
        :param cut_words: 文本文本分词
        :return: 数据清洗后的文本文本分词
        """
        stop_regex = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
        for item in cut_words:
            # 保留特殊标识符
            if item in ('DOU分期', '余额-', '360', '<num>', '<urla>', '<urlb>', 'etc', 'atm', 'pos', '[', ']', '(', ')', '（', '）', '【', '】'):
                yield item
            # 过滤停止词, 停止词stop_words赋值见class_segmentation.py
            if item in self.stop_words:
                continue
            # 过滤数值
            if item.isdigit():
                continue
            # 过滤英文与ascii码<128的符号
            if item not in ['etc', 'atm', 'pos', '[', ']', '(', ')', '（', '）'] and all(ord(j) < 128 for j in item):
                continue
            # 过滤特殊符号
            item = re.sub(stop_regex, '', item)
            if not item:
                continue
            yield item

    def Q2B(self, uchar):
        """单个字符 全角转半角"""
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
            return uchar
        return chr(inside_code)

    def stringQ2B(self, ustring):
        """把字符串全角转半角"""
        return "".join([self.Q2B(uchar) for uchar in ustring])

    def _check_plat_type(self, plat_name, plats_dict):
        """
        根据平台的名称，查字典获得平台的类型
        :param plat_name: 平台名称
        :param plats_dict: 平台字典
        :return: 平台类型
        """
        for plat_type in plats_dict:
            if plat_name in plats_dict[plat_type]:
                return plat_type
        return "NF"

    def _get_plats_dict(self):
        """
        获取平台类型划分字典，{"银行":[中国银行,农业银行,...]，"消费金融":[支付宝,京东金融...],...}
        :return: 平台类型划分字典
        """
        plats_type_dict = {}
        with open("segmenter_resources/plats_division_dict.txt", "r") as fr:
            text_lines = fr.readlines()
            for text_line in text_lines:
                plat_name = text_line.split("|")[0]
                plat_type = text_line.split("|")[2][:2]
                if plat_type not in plats_type_dict:
                    plats_type_dict[plat_type] = [plat_name]
                else:
                    plats_type_dict[plat_type].append(plat_name)
        return plats_type_dict

    def merge_cutwords(self, cuts, pos="t"):
        """
        融合连续多个相同词性的词，比如 [2020年，12月，21日]三个词的词性均为 t，则将这3个词合到一起
        :param cuts:分词结果，包含分词和词性
        :param pos: 指定要融合的词性，连续多次出现的相同词性的词，合并这多个分词
        :return: 融合之后的分词
        """
        last_pos = "none"
        term_list = []
        for term in cuts:
            if str(term.nature) == pos:
                term.word = "<时间>"
                if last_pos == pos:
                    last_pos = pos
                    pass
                else:
                    term_list.append(term)
                    last_pos = pos
            else:
                term_list.append(term)
        return term_list


stopwords_path = "Resources/segmenter_resources/stopwords.txt"
seg_dict_path = "Resources/segmenter_resources/jieba_dict_v1.txt"
segmenter = Segmenter(stopwords_path, seg_dict_path)

if __name__ == "__main__":
    # dataset_path = "Dataset/beta_update_20210111.txt"
    # dataset_df = pd.read_csv(dataset_path, sep="|")

    test_message = ""
    # test_message = text_clean(test_message)
    cut_words = segmenter.cut_words(test_message)
    print(cut_words)
