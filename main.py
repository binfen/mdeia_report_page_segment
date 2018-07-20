#!/usr/bin/env python
# -*- coding:utf-8 -*-
# TODO: 推送2018.3.24,相较于v4改变:
#  1. 不在重新修订"标题栏"分割结果,"标题栏"的初步分割结果来分离"核心指标"栏.
#  2. 不再返回固定四列,但是依然修改块,属性待识别完成再进行判断
#  3. 加入判断图片是否合法,即判断是时间栏,检验名称,核心指标.
# TODO: 推送2018.4.18 ,相比较于v5改变：
#  1. 在最后的文本行分割阶段不再使用中间二值化结果，转而使用最初的二值化结果
#  2. 在最后的文本行分割阶段不再使用全局估算的文本宽度，转而使用局部估算的文本宽度
#  3. 在最后加入去除噪声干扰的操作


from __future__ import print_function
import shutil
from pylab import *
import os.path
import cv2
import math
import ocrolib
from ocrolib import psegutils, morph, sl
from ocrolib.toplevel import *
import numpy as np
import scipy
from scipy import stats
from ocrolib.common import read_image_gray
from scipy.ndimage import interpolation, morphology
from scipy.ndimage import filters
import pdb
# import scipy
# print('python:',sys.version)
# print('cv2:',cv2.__version__)
# print('numpy:',numpy.__version__)
print('scipy:',scipy.__version__)
# pdb.set_trace()


#单个文本行记录,包括检索位置,真实的图像数据
class name_dic:
    def __init__(self, index, data):
        self.index = index      # 检索位置
        self.array = data      # 图像数据

#计算文本区域的起始位置
def compute_index(colseps, th, n=0):
    '''
    :param colseps:分割器图像
    :return: 以列表形式返回colseps中的满足条件的列的位置,条件是此列中n连续的出现的次数大于th
    '''
    height, width = colseps.shape
    list_index = []

    # 计算列表 l 中连续=m的数目，返回最大连续数
    def checknum(l, m):
        res = []
        count = 0
        for i in l:
            if i == m:
                count += 1
            else:
                res.append(count)
                count = 0
        res.append(count)
        return max(res)

    for i in range(width):

        l=list(colseps[:, i])
        num = checknum(l, n)
        list_index.append(num)

    text_index = [i for i, a in enumerate(list_index) if a > th]

    indexs = []
    if len(text_index) > 0:
        beg_index = text_index[0]
        end_index = text_index[0]
        for i in range(1, len(text_index) - 1):
            end_index = text_index[i]
            if text_index[i] - text_index[i - 1] != 1:
                end_index = text_index[i - 1]
                indexs.append([beg_index, end_index])
                beg_index = text_index[i]
        indexs.append([beg_index, end_index])
    else:
        beg_index = 0
        end_index = width
        indexs.append([beg_index, end_index])
    return indexs

#二值化
def Binarization(raw):
    '''
    :param fname:医助手动切割后的图像
    :return: 二值化结果
    '''
    def check_image(gray_img):
        #页面安全检查
        image = gray_img - np.amin(gray_img)
        if np.amax(image) == np.amin(image):
            return None
        image /= np.amax(image)
        image_temp = np.amax(image) - image
        if len(image_temp.shape)==3 and np.mean(image_temp)<np.median(image_temp):
            return None
        h, w = image_temp.shape

        if h<5 and h>10000 and w<100 and w>10000:
            return None
        return image
    def adjust_whitelevel(image, zoom=0.5, perc=80, range=20):
        #调整白平衡
        extreme = (np.sum(image < 0.05) + np.sum(image > 0.95)) * 1.0 / np.prod(image.shape)
        if extreme > 0.95:
            flat = image
        else:

            m = interpolation.zoom(image, zoom)
            m = filters.percentile_filter(m, perc, size=(range, 2))
            m = filters.percentile_filter(m, perc, size=(2, range))
            m = interpolation.zoom(m, 1.0/zoom)
            w, h = np.minimum(np.array(image.shape), np.array(m.shape))
            flat = np.clip(image[:w, :h]-m[:w, :h]+1, 0, 1)
            import pdb
            #pdb.set_trace()
        return flat
    def normalization(flat, lo=5, hi=95):
        #归一化
        v = filters.gaussian_filter(flat, e*20.0)
        v = flat - v
        v = filters.gaussian_filter(v**2, e*20.0)**0.5
        v = (v>0.3*np.amax(v))
        v = morphology.binary_dilation(v,structure=np.ones((20, 1)))
        v = morphology.binary_dilation(v,structure=np.ones((1, 20)))

        est = flat[v]
        lo = stats.scoreatpercentile(est.ravel(), lo)
        hi = stats.scoreatpercentile(est.ravel(), hi)
        flat -= lo
        flat /= (hi - lo)
        flat = np.clip(flat, 0, 1)
        return flat
    def adjust_rotate(flat, bina):
        height, width = flat.shape
        #旋转调整
        edges = bina.astype(np.uint8)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=250, maxLineGap=10)
        if lines is not None:
            lines1 = lines[:, 0, :]  # 提取为二维
            angle_sum = 0
            i = 0
            angle = 0
            if len(lines1) > 0:
                for x1, y1, x2, y2 in lines1[:]:
                    angle_line = (180 / np.pi) * (math.atan(1.0 * (y1 - y2) / (x2 - x1+1)))
                    if angle_line > -45 and angle_line < 45 and angle_line != 0:
                        angle_sum += angle_line
                        i += 1
                if i != 0:
                    angle = 1.0 * angle_sum / i
            flat = interpolation.rotate(flat, -angle, mode='constant',cval=1)
        return flat

    image = check_image(raw)
    if image is None:
        print('input image is invalid.')
        return None,None
    flat = adjust_whitelevel(image)
    flat = normalization(flat)

    bina1 = 1*(flat < 0.6)
    flat = adjust_rotate(flat, bina1)
    bina = 1 * (flat < 0.5)#黑底白字
    return flat, bina

#垂直分块
def split_columns_vertical(gray, bina, scale):
    '''
    :param gray: "核心内容"二值化后的灰度图
    :param bina: "核心内容"二值化后的二值图
    :param scale: "核心内容"字体宽度
    :return: 根据黑线的分割结果
    '''
    # 计算垂直黑线图
    height, width = bina.shape
    edges = bina.astype(np.uint8)
    colsep_black = np.zeros(shape=bina.shape)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 90, int(5*scale), minLineLength=height/5, maxLineGap=scale / 2)
    if lines is not None:
        lines1 = lines[:, 0, :]  # 提取为二维
        for x1, y1, x2, y2 in lines1[:]:
            angle = (180 / np.pi) * (math.atan(1.0 * (y1 - y2) / (x2 - x1 + 1)))
            if angle > 80 or angle < -80:
                colsep_black[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] = 1
    # 计算垂直黑线真实位置
    indexs_black = compute_index(colsep_black, th=2* height / 3, n=0)

    # 过滤掉太窄的文本列
    indexs1 = []
    for index in indexs_black:
        if index[1] - index[0] >= 5 * scale:
            indexs1.append(index)

    bina_lists = []
    gray_lists = []
    for i, index1 in enumerate(indexs1):
        gray1 = gray[:, index1[0]:index1[1]]
        bina1 = bina[:, index1[0]:index1[1]]
        gray_lists.append(gray1)
        bina_lists.append(bina1)
    return gray_lists, bina_lists

#提取标题栏与核心指标栏
def get_caption_mainbody(gray, binary, scale):
    '''
    :param gray:二值化的那些核心内容图
    :param binary::二值化的那些核心内容图
    :param scale: 普通的字体宽度
    :return: 分割后的"标题栏","核心指标"'''
    height, width = binary.shape
    edges = binary.astype(np.uint8)
    horiz = np.zeros(shape=binary.shape)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, min(width/2,int(5*scale)), minLineLength=width/3, maxLineGap=scale/3)
    if lines is not None:
        lines1 = lines[:, 0, :]  # 提取为二维
        for x1, y1, x2, y2 in lines1[:]:
            angle = (180 / np.pi) * (math.atan(1.0 * (y1 - y2) / (x2 - x1 + 1)))
            if angle > -10 and angle < 10:
                horiz[min(y1, y2):max(y1, y2) + 1, min(x1, x2):max(x1, x2) + 1] = 1

    number_index = np.sum(horiz, axis=1)
    number_index = list(number_index)
    text_index = [i for i, a in enumerate(number_index) if a==0]  #非黑色直线所在位置集合
    if len(text_index) < 2:
        return None,None,None,None

    # 统计满足条件的文本位置
    indexs = []
    beg_index = 0
    for i in range(1, len(text_index)):
        if text_index[i] - text_index[i-1]!=1:   #文本间距超过一定宽度认为文本重新开始计算开始
            end_index = (text_index[i] + text_index[i-1]) / 2
            indexs.append([beg_index, end_index])
            beg_index = end_index
    indexs.append([beg_index, height])

    #从前到后排序
    max_height = 0
    indexs.sort(key=lambda x:x[0])

    if len(indexs) > 1:        #存在标题栏:合法或不合法
        cp_index = []
        for i, index in enumerate(indexs):
            if index[1]-index[0] > max_height:
                mb_index = index
                if i != 0 and indexs[max(i - 1, 0)][1] - indexs[max(i - 1, 0)][0] > scale:
                    cp_index = indexs[max(i-1, 0)]
                max_height = index[1] - index[0]

        if cp_index!=[]:       #存在合法标题栏
            cp_gray = gray[cp_index[0]:cp_index[1], :]
            cp_bina = binary[cp_index[0]:cp_index[1], :]
        else:                  #存在不合法的标题栏
            cp_gray = None
            cp_bina = None
        mb_gray = gray[mb_index[0]:mb_index[1], :]
        mb_bina = binary[mb_index[0]:mb_index[1], :]

    else:                      # 图片不存在标题栏
        cp_gray = None
        cp_bina = None
        mb_gray = gray
        mb_bina = binary

    return cp_gray, cp_bina, mb_gray, mb_bina

#标题栏初步：最大值滤波
def caption_segment(binary):
    '''
    :param gray:待分析的析"标题栏"
    :param bina:
    :return:
    '''
    # 排除边界处干扰部分
    bina = ocrolib.remove_noise(binary, 8)
    scale = psegutils.estimate_scale(bina)
    lines = morph.select_regions(bina, sl.dim1, min=2*scale)
    bina = bina - lines
    bina = morph.select_regions(bina, sl.dim0, min=scale/3)
    #扩大文本区域,连接相邻文本
    textlines = filters.maximum_filter(bina, (scale, scale/2))
    #计算候选文本区域起始位置
    indexs_white = compute_index(textlines, th=scale / 2, n=1)
    indexs_lists = []
    if len(indexs_white) > 2:
        index_fir = indexs_white[0]
        #排除过小同时连接相邻的候选文本区域
        for i, index in enumerate(indexs_white):
            if index[1] - index[0] > scale/2:                      #排除过小
                if i != 0 and index[0] - index_fir[1] < scale/3:   #连接相近

                    index_acc = [index_fir[0], index[1]]
                    indexs_lists.remove(index_fir)
                    indexs_lists.append(index_acc)
                    index_fir = index_acc
                else:
                    indexs_lists.append(index)
                    index_fir = index
    return indexs_lists

#分割核心指标栏: 空白区域
def mainbody_segment(gray, binary, scale, index_list):
    '''
    :param gray:
    :param binary: 待测的"核心指标栏"
    :param scale: 字符宽度, float类型
    :param index_list: "标题栏"中文本沿着水平方向所在的位置列表, list类型
    :return: 沿着空白区域分割的分割图, array类型
    '''

    # 当存在"标题栏"情况下,计算垂直空白分割位置
    def search_sep_index1(bina, th, n=2):
        '''
        :param bina:待检测图像
        :param n: 匹配的模板列数
        :return:  返回待测图中连续n列白点最少的位置,如果存在多个最少,则取两个最少中间的位置
        '''
        height, width = bina.shape
        beg_index = []
        end_index = []
        min_sum = n * height
        all_sum = np.sum(bina, axis=0)
        for i in range(0, width - 1):    #以非重复方式递进
            num_sum = sum(all_sum[i:i + n])
            if num_sum < min_sum:
                min_sum = num_sum
                beg_index = [i, i + 1]
                end_index = [i, i + 1]
            elif num_sum == min_sum:
                end_index = [i, i + 1]
        if len(beg_index)>0 and len(end_index)>0:
            res_index = (beg_index[1] + end_index[0]) / 2

            if np.sum(bina[:, res_index]) < th:   #白像素个数小于一定数目,才认为是真的分割位置
                return res_index
        else:
            return None

    # 当不存在"标题栏"情况下,计算垂直白色空白位置
    def search_sep_index2(bina, scale):
        '''
        :param binary:待检测的"核心指标"图,array类型
        :param scale: 字体宽度, float类型
        :return: 返回待测图中各空白区域处的中间位置, int类型
        '''
        indexs = np.sum(bina, axis=0)

        indexs = list(1.0 * indexs / scale)       # 排除噪声干扰:当某列中像素点数小于一定量时候,排除干扰
        text_index_temp = [i for i, index in enumerate(indexs) if index > 1]  # 候选文本位置列表

        text_index_acct = []  # 真正的文本位置列表

        if len(text_index_temp) > 0:
            beg_index = text_index_temp[0]
            end_index = text_index_temp[0]
            for i in range(1, len(text_index_temp)):
                end_index = text_index_temp[i]
                if text_index_temp[i] - text_index_temp[i - 1] > 4:  # 当文本间隔超过一定阈值时候,才认为文本从新开始
                    end_index = text_index_temp[i - 1]
                    if end_index - beg_index > scale:                # 当文本宽度大于一个字符跨度时候,才认为是真正的文本
                        text_index_acct.append([beg_index, end_index])
                    beg_index = text_index_temp[i]
            text_index_acct.append([beg_index, end_index])
        text_index_acct.sort(key=lambda x: x[0])
        res_index = []
        for i in range(len(text_index_acct) - 1):
            index = (text_index_acct[i + 1][0] + text_index_acct[i][1]) / 2
            res_index.append(index)
        return res_index

    # import pdb
    # pdb.set_trace()
    # 排除边界处干扰部分
    bina = ocrolib.remove_noise(binary, 8)
    lines = morph.select_regions(bina, sl.dim1, min=2 * scale)
    bina = bina - lines
    lines = morph.select_regions(bina, sl.dim0, min=2 * scale)
    bina = bina - lines

    # 存在"标题栏"
    if 6 >len(index_list)>3 :
        colsep_index = []
        # 线性扩张:白色区域变大
        bina_d = filters.maximum_filter(bina, (scale, scale))
        i=0
        while len(index_list):
            # 取标题栏中对应的连续两个位置的中间值,组成新的位置,作为待测位置
            # eg:假设"标题栏"中两个相邻的文本区域,在对水平方向应的位置分别是是[x00,x01],[x10,x11],
            # 则:分割线应该出现在"核心指标栏"中水平方向[(x00+x01)/2, (x10+x11)/2]范围内.
            # TODO:此处有个bug,即当标题栏只有一个属性的情况下.该如何分割指标区域
            sep_index = None
            while sep_index is None and i<len(index_list)-1:

                index = [(index_list[i][0] + index_list[i][1]) / 2,
                         (index_list[i + 1][0] + index_list[i + 1][1]) / 2]

                bina_i = bina_d[:, index[0]:index[1]]
                sep_index = search_sep_index1(bina_i, 10 * scale)  # 返回计算得到分割位置

                if sep_index is None:  # 意味着标题栏初始分割失败
                    index_re1 = index_list[i]
                    index_re2 = index_list[i + 1]
                    index_new = [index_re1[0], index_re2[1]]
                    index_list.remove(index_re1)
                    index_list.remove(index_re2)
                    index_list.insert(i, index_new)
                    if len(colsep_index) > 0:
                        b=colsep_index.pop()
                    if i>0:
                        i = i - 1
            if sep_index is not None:
                sep_index = sep_index + index[0]
                colsep_index.append(sep_index)
                if i > 0:
                    i = i - 1
                    index_list.remove(index_list[i])
            else:
                index_list.remove(index_list[0])
            i += 1

    # 不存在"标题栏"
    else:
        bina_d = filters.maximum_filter(bina, (scale, scale/2))#改为２＊scale？
        colsep_index = search_sep_index2(bina_d, scale)

    colsep_index.append(0)
    colsep_index.append(bina.shape[1])
    colsep_index.sort(key=lambda x: x)

    # 返回最终的文本位置列表
    res_index = []
    for i in range(len(colsep_index) - 1):
        beg_index = colsep_index[i]
        end_index = colsep_index[i + 1]
        res_index.append([beg_index, end_index])

    bina_lists = []
    gray_lists = []
    for index in res_index:
        gray_i = gray[:, index[0]:index[1]]
        bina_i = binary[:, index[0]:index[1]]
        gray_lists.append(gray_i)
        bina_lists.append(bina_i)
        # plt.imshow(bina_i, 'gray'), plt.show()
    return gray_lists,bina_lists

#核心指标栏内文本行分割
def mainbody_textline_segment(gray, bina, scale, black_id, col_id, dictionary):
    '''
    :param gray: "核心指标栏"中某属性列灰度图
    :param bina: "核心指标栏"中某属性列二值图
    :param black_id: "核心指标栏"中某属性列所属块id
    :param col_id: "核心指标栏"中某属性列所属列id
    :param dictionary: 文件存储记录
    :return: 文件存储记录和此属性列所含行数
    '''

    #排除多种干扰
    bina = 1 * (gray < 0.5)
    bina = ocrolib.remove_noise(bina, 5)                       #希望排除一定的噪声干扰
    scale = psegutils.estimate_scale(bina)
    height, width = gray.shape
    lines = morph.select_regions(bina, sl.dim0, min=2*scale)   #希望排除水平方向边缘处的亮斑干扰
    bina = bina - lines
    lines = morph.select_regions(bina, sl.dim1, min=2*scale)   #希望排除垂直方向边缘处的亮斑干扰
    bina = bina - lines

    #字符合并
    textlines = filters.maximum_filter(bina, (0, scale))
    textlines = morph.rb_erosion(textlines, (3, 0))
    textlines = morph.rb_dilation(textlines, (0, scale))

    #统计文本行位置
    textpixe_num = np.sum(textlines, axis=1)
    textpixe_num = 1 * ((1.0 * textpixe_num / scale) > 1)
    textpixe_num = list(textpixe_num)

    text_index = [i for i, a in enumerate(textpixe_num) if a == 1]
    indexs = []
    max_row = 0
    if len(text_index) > 0:
        beg_index = text_index[0]
        end_index = text_index[0]
        for i in range(1, len(text_index) - 1):
            if text_index[i] - text_index[i - 1] != 1:
                end_index = text_index[i - 1]
                indexs.append([beg_index, end_index])
                beg_index = text_index[i]
            end_index = text_index[i]
        indexs.append([beg_index, end_index])

        #选取有效的文本行
        results_indexs = []
        if len(indexs)>0:
            for index in indexs:
                if index[1] - index[0] >= scale / 4:
                    results_indexs.append(index)

        # res_index = []
        # if len(results_indexs)>0:
        #     i=0
        #     beg_index=results_indexs[i][0]/2
        #     for i in range(len(results_indexs)-1):
        #         end_index=(results_indexs[i][1]+results_indexs[i+1][0])/2
        #         res_index.append([beg_index, end_index])
        #         beg_index = end_index
        #     if i==0:
        #         end_index = (results_indexs[i][1] + height) / 2
        #     else:
        #         end_index = (results_indexs[i+1][1] + height) / 2
        #
        #     res_index.append([beg_index,end_index])

        for row_id, index in enumerate(results_indexs):
            key = '%d.%d.%d.png' % (black_id, col_id, row_id)
            data =255 * gray[max(0,index[0]-5):min(height,index[1]+5), :]
            value = name_dic(index, data)
            dictionary[key] = value
            max_row = row_id
    return dictionary, max_row

#修改"核心指标栏"中所有文件名称记录: 删除空白行
def modify_fname_dictionary(dic, dic_new):
    '''
    :method:    每次都检查第一行,然后删除，更新，重复直至dic为空，并返回一个新的dic_new
    :param dic: 某以块中文件存储记录
    :param dic_new: 删除空白文本行后的文件记录
    :return:   删除空白文本行后的文件记录
    '''

    # 统计列总数目
    def compute_col_number(dic):
        col_num = 0
        for i, key in enumerate(dic):
            _, a, _, _ = key.split('.')
            a = int(a)
            if a > col_num:
                col_num = a
        col_num = col_num + 1
        return col_num

    # 统计col_id列中行总数目
    def compute_row_number(dic, col_id):
        row_num = 0
        for key in dic:
            _, a, b, _ = key.split('.')
            a = int(a)
            b = int(b)
            if a == col_id:
                if b > row_num:
                    row_num = b
        row_num = row_num + 1
        return row_num

    # 查找第row_id行的最高文本框位置
    def compute_high_index(dic, row_id=0):
        first_top = 10000
        for i, key in enumerate(dic):
            _, _, b, _ = key.split('.')
            b = int(b)
            index = dic[key].index
            if b == row_id:
                if first_top > index[0]:
                    first_top = index[0]
                    key_first = key
        return key_first

    def compute_iou(bound_left, bound_right):
        #计算两个框的重叠率
        iou = 0
        if bound_left[0] < bound_right[0]:
            left = bound_left
            right = bound_right
        else:
            left = bound_right
            right = bound_left

        if right[0] > left[1]:
            return iou
        else:
            iou = 1
            return iou

    def adjument_column_dictionary(dic, black_id, col_id):
        #调整每列的记录：　此列中的所有行记录值减一
        row_num = compute_row_number(dic, col_id)
        for row_id in range(1, row_num):
            key = '%d.%d.%d.png' % (black_id, col_id, row_id)
            if dic.has_key(key):
                index = dic[key].index
                data = dic[key].array
                del dic[key]
                key_new = str(black_id) + '.'+str(col_id) + '.' + str(row_id - 1) + '.png'
                value = name_dic(index, data)
                dic[key_new] = value
        return dic

    def update_dictionary(dic_new, dic, ite, key_first, col_num, adjument_num):
        #通过将每行中排序正确的文本行删除来更新字典
        black_id, _, _, _ = key_first.split('.')
        black_id = int(black_id)

        bound_left = dic[key_first].index
        for i in range(col_num):
            key = '%d.%d.0.png' % (black_id, i)
            if dic.has_key(key):
                bound_right = dic[key].index
                iou = compute_iou(bound_left, bound_right)
                if iou:
                    index = dic[key].index
                    data = dic[key].array
                    value = name_dic(index, data)
                    del dic[key]
                    key_new = '%d.%d.%d.png' % (black_id, i, ite)
                    dic_new[key_new] = value
                    dic = adjument_column_dictionary(dic, black_id, i)
                    adjument_num[i] += 1
        return dic_new, dic

    col_num = compute_col_number(dic)
    adjument_num = [0] * 20 #每列调整次数，以列表形式记录
    ite = 0 # 更新操作进行到第几行，最终应该与最大行数相同　
    while dic:
        key_first= compute_high_index(dic)
        dic_new, dic = update_dictionary(dic_new, dic, ite, key_first, col_num, adjument_num)
        ite += 1
    return dic_new

#对"核心指标栏"中所有文件进行显示格式的修改: 显示四个属性
def modify_mainbody_display(dic, lll):
    '''
    :method:  罗列多种表单格式,然后进行按规则删除显示
    :param dic: 存储"核心指标栏"内的所有文件记录
    :param lll: 记录块id,对应块内的行数col,列数row
    :return: 关于"核心指标栏",最终正确显示的所有文件记录
    '''

    new_dic = {}
    names = dic.keys()
    if len(lll) == 1:
        for old_name in names:
            new_dic[old_name] = dic[old_name]

    elif len(lll) == 2:
        max_row = lll[0][2]
        for old_name in names:
            a, b, c, d = old_name.split('.')
            a, b, c = int(a), int(b), int(c)
            if a > 0:
                a = a - a
                c = c + max_row
                new_name = str(a) + '.' + str(b) + '.' + str(c) + '.' + d
                new_dic[new_name] = dic[old_name]
            else:
                new_dic[old_name] = dic[old_name]

    elif len(lll) == 8:
        max_row = 0
        lll.sort(key=lambda x: x[0])
        for i, index in enumerate(lll):
            if i < 4:
                if index[2] > max_row:
                    max_row = index[2]

        for old_name in names:
            a, b, c, d = old_name.split('.')
            a, b, c = int(a), int(b), int(c)

            if a < 4:
                b = b + a
                a = a - a
                new_name = str(a) + '.' + str(b) + '.' + str(c) + '.' + d
                new_dic[new_name] = dic[old_name]
            else:
                b = b + a - 4
                a = a - a
                c = c + max_row
                new_name = str(a) + '.' + str(b) + '.' + str(c) + '.' + d
                new_dic[new_name] = dic[old_name]

    return new_dic

# 对"核心指标栏"的文件存储字典记录加入区别"标题栏"和"核心指标栏"的标识符
def add_flag(mb_dic):
    names = mb_dic.keys()
    if len(names) > 0:
        for old_name in names:
            _, b, c, d = old_name.split('.')
            new_name = b + '.' + c + '.' + d
            mb_dic[new_name] = mb_dic[old_name]
            del mb_dic[old_name]
    return mb_dic

#通过文件保存记录存储图像
def save_img_from_dic(save_path, dictionary):
    for name in dictionary:
        data = dictionary[name].array
        if data.shape[0]<15:
            data = cv2.resize(data, (data.shape[1],15))
        path = os.path.join(save_path, name)
        cv2.imwrite(path, data)

#分割主函数
def Segment(fname, save_path):
    # pdb.set_trace()
    # 清理上次执行的缓存结果
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    #读取图像数据
    raw = read_image_gray(fname)

    #二值化,抗旋转,抗明暗度变化
    gray_o, bina_o = Binarization(raw)

    #出现类型错误，返回分割失败标识0
    if gray_o is None and bina_o is None:
        new_fname = os.path.basename(fname)
        cv2.imwrite(os.path.join(save_path,new_fname))
        return 0

    #估计文本宽度
    bina_o = ocrolib.remove_noise(bina_o, 8)
    scale = psegutils.estimate_scale(bina_o)

    #页面分块
    block_grays, block_binas = split_columns_vertical(gray_o, bina_o, scale)


    if len(block_grays) > 2:  # 图片格式出现特殊情况,即指标栏之间均以垂直黑线分隔开,直接进行行分割
        mb_dics = {}          # 存储删除空白行记录后的所有文件记录
        mb_block = []         # 记录每块中的行列数
        for i, gray_i in enumerate(block_grays):
            mb_dic = {}
            bina_i = block_binas[i]

            #分离属性栏区域和核心指标栏
            cp_gray, cp_bina, mb_gray, mb_bina = get_caption_mainbody(gray_i, bina_i, scale)

            #行分割，并将结果分割结果及存储名称以字典形式存放
            mb_dic, max_row = mainbody_textline_segment(mb_gray, mb_bina, scale, i, 0, mb_dic)
            max_col = 0
            max_row += 1

            #表单结构化初步调整：调整空白栏
            mb_dics = modify_fname_dictionary(mb_dic, mb_dics)
            mb_block.append([i, max_col, max_row])

        #表单结构化后处理：合并多块
        res_mb_dics = modify_mainbody_display(mb_dics, mb_block)

        #去除分块标识
        res_mb_dics = add_flag(res_mb_dics)

    else:               # 正常的格式,即单块或者两块
        mb_dics = {}    # 存储删除空白行记录后的所有文件记录
        mb_block = []   # 记录每块中的行列数

        for i, gray_i in enumerate(block_grays):
            bina_i = block_binas[i]

            #分离属性栏区域和核心指标栏区域
            cp_gray, cp_bina, mb_gray, mb_bina = get_caption_mainbody(gray_i, bina_i, scale)

            #属性栏列方向分割,获得各属性分割位置,以列表形式存放
            cp_index_list = []
            if cp_bina is not None:
                cp_index_list = caption_segment(cp_bina)

            #核心指标栏区域列方向分割,获得各属性列分隔位置,并截取图像数据,以列表形式存放
            mb_grays, mb_binas = mainbody_segment(mb_gray, mb_bina, scale, cp_index_list)

            ######----------"核心指标栏":文本行分割----------########
            max_col = 0   # 第一块中列数
            max_row = 0   # 第一块中最大行数
            mb_dic = {}   # 存储每块中的文件记录
            for j, bina_j in enumerate(mb_binas):
                if j > max_col:
                    max_col = j
                gray_j = mb_grays[j]

                #文本行分割,获得文本行分割结果，以字典形式存储
                mb_dic, row = mainbody_textline_segment(gray_j, bina_j, scale, i, j, mb_dic)
                if row > max_row:
                    max_row = row

            max_col += 1
            max_row += 1

            if mb_dic is not {}:
                # 表单结构化初步调整：调整空白栏
                mb_dics = modify_fname_dictionary(mb_dic, mb_dics)
                mb_block.append([i, max_col, max_row])

        # 表单结构化后处理：合并多块
        res_mb_dics = modify_mainbody_display(mb_dics, mb_block)    # 存储经过显示调整的所有文件记录
        #　去除块标识
        res_mb_dics = add_flag(res_mb_dics)

    #根据字典数据保存分割结果，并返回分割成功标识１
    save_img_from_dic(save_path, res_mb_dics)
    return 1

if __name__ == '__main__':
    fname = 'images/1.jpg'
    out_path = 'outputs'
    state=Segment(fname, out_path)



