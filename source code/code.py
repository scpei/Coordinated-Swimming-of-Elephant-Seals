from operator import itemgetter
from math import *
import math
import numpy
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from PIL import Image, ImageSequence
import sys, os
import cv2
import operator
import random
from datetime import date



EARTH_RADIUS=6371

def hav(theta):
    s = math.sin(theta / 2)
    return s * s

def CalDis(lef, rig, lef_mean, rig_mean):
    lat0 = math.radians(lef)
    lat1 = math.radians(lef_mean)
    lng0 = math.radians(rig)
    lng1 = math.radians(rig_mean)

    dlng = math.fabs(lng0 - lng1)
    dlat = math.fabs(lat0 - lat1)
    h = hav(dlat) + math.cos(lat0) * math.cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * math.asin(math.sqrt(h))

    return distance

def getLng(di, lef, rig):
    d = math.fabs(di)
    h = math.sin(d/(2*EARTH_RADIUS))*math.sin(d/(2*EARTH_RADIUS))
    dlng = math.asin(math.sqrt(h / ((math.cos(math.radians(lef))) * math.cos(math.radians(lef))))) * 2
    de = math.degrees(dlng)
    n_rig = rig + de
    if di < 0:
        n_rig = rig - de
    return n_rig

def getLat(di, lef, rig):
    d = math.fabs(di)
    h = math.sin(d/(2*EARTH_RADIUS))*math.sin(d/(2*EARTH_RADIUS))
    dlng = math.asin(math.sqrt(h))*2
    de = math.degrees(dlng)
    n_lef = lef + de
    if di < 0:
        n_lef = lef - de
    return n_lef

def tt():
    d = CalDis(0, 0, -5, 0)
    h = math.sin(d/(2*EARTH_RADIUS))*math.sin(d/(2*EARTH_RADIUS))
    dlng = math.asin(math.sqrt(h))*2
    de = math.degrees(dlng)
    print(de)

# def CalDis(Lat_A, Lng_A, Lat_B, Lng_B):
#     ra = 6378.140  # 赤道半径 (km)
#     rb = 6356.755  # 极半径 (km)
#     flatten = (ra - rb) / ra  # 地球扁率
#     rad_lat_A = radians(Lat_A)
#     rad_lng_A = radians(Lng_A)
#     rad_lat_B = radians(Lat_B)
#     rad_lng_B = radians(Lng_B)
#     pA = atan(rb / ra * tan(rad_lat_A))
#     pB = atan(rb / ra * tan(rad_lat_B))
#     xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
#     c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
#     c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
#     dr = flatten / 8 * (c1 - c2)
#     distance = ra * (xx + dr)
#     return distance

def CalGyr(list):
    lef = []
    rig = []
    s = []
    for i in list:
        lef.append(i[1])
        rig.append(i[2])
    lef_mean = numpy.mean(lef)
    rig_mean = numpy.mean(rig)
    for i in list:
        d = CalDis(i[1], i[2], lef_mean, rig_mean)
        s.append(pow(d, 2))
    return (sum(s)**0.5)/len(s)

def CalCenPoint(list):
    lef = []
    rig = []
    for i in list:
        lef.append(i[1])
        rig.append(i[2])
    lef_mean = numpy.mean(lef)
    rig_mean = numpy.mean(rig)
    return lef_mean, rig_mean

def CalCoh(list):
    ss = []
    sc = []
    for i in list:
        ss.append(i[4])
        sc.append(i[5])

    return (pow((sum(sc)/len(sc)), 2) + pow((sum(ss)/len(ss)), 2))**0.5

def CalCohForTotalRandom(list):
    ss = []
    sc = []
    for i in list:
        ss.append(i[3])
        sc.append(i[4])

    return (pow((sum(sc)/len(sc)), 2) + pow((sum(ss)/len(ss)), 2))**0.5

# f = open("male_data_oneyear.txt", 'r')
# f = open("female_data_oneyear.txt", 'r')
f = open("NESE_format0.dat",'r')
dat = []
for line in f:
    id = int(line.split(' ')[0])
    lef = float(line.split(' ')[1])
    rig = float(line.split(' ')[2])
    if rig > 0:
        rig = -rig
    time = float(line.split(' ')[3])
    dat.append([id, lef, rig, time])


def Gy():
    dat_sort = sorted(dat, key=itemgetter(3))
    time_dic = {}
    for item in dat_sort:
        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = []
            time_dic[int(item[3])].append(item)
        else:
            time_dic[int(item[3])].append(item)

    rg = []
    for key in time_dic.keys():
        interv = {}
        for it in time_dic[key]:
            if (it[3] - int(it[3]))*24 - int((it[3] - int(it[3]))*24) > 0.9999999995:
                k = math.ceil((it[3] - int(it[3]))*24)
            else:
                k = int((it[3] - int(it[3]))*24)

            if k not in interv:
                interv[k] = []
                interv[k].append(it)
            else:
                interv[k].append(it)
        ## Rg function to calculate Rg

        for hour in interv.keys():
            rg.append(CalGyr(interv[hour]))

    # print(len(rg))

    # plt.scatter(numpy.arange(0, len(rg)), numpy.asarray(rg), marker= '.', s = 20)
    # plt.xlim((0, len(rg)))
    # plt.ylim((0, 1000))
    # plt.xticks([8760, 8760*2, 8760*3, 8760*4, 8760*5, 8760*6, 8760*7, 8760*8, 8760*9, 8760*10], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # plt.xlabel('t (years)')
    # plt.ylabel('Rg(km)')
    #
    # plt.show()

    return rg


def Cohen():
    id_dic = {}
    for item in dat:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i-1][1]
            rig = id_dic[key][i][2] - id_dic[key][i-1][2]
            r = (pow(lef, 2) + pow(rig, 2))**0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))
    time_dic = {}
    for item in dat_sort:
        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = []
            time_dic[int(item[3])].append(item)
        else:
            time_dic[int(item[3])].append(item)

    coh_rate = []
    for key in time_dic.keys():
        interv = {}
        for it in time_dic[key]:
            if (it[3] - int(it[3])) * 24 - int((it[3] - int(it[3])) * 24) > 0.995:
                k = math.ceil((it[3] - int(it[3])) * 24)
            else:
                k = int((it[3] - int(it[3])) * 24)

            if k not in interv:
                interv[k] = []
                interv[k].append(it)
            else:
                interv[k].append(it)
        ## cohence function to calculate co

        for hour in interv.keys():
            coh_rate.append(CalCoh(interv[hour]))

    print(len(coh_rate))

    # plt.scatter(numpy.arange(0, len(coh_rate)), numpy.asarray(coh_rate), marker='.', s=5)
    # plt.xlim((0, len(coh_rate)))
    # plt.ylim((0, 1))
    # plt.xticks([8760, 8760 * 2, 8760 * 3, 8760 * 4, 8760 * 5, 8760 * 6, 8760 * 7, 8760 * 8, 8760 * 9, 8760 * 10],
    #            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # plt.xlabel('t (years)')
    # plt.ylabel('R')
    #
    # plt.show()


    # plt.scatter(numpy.arange(0, len(coh_rate[8760*7:8760*8])), numpy.asarray(coh_rate[8760*7:8760*8]), marker='.', s=20)
    # plt.xlim((0, len(coh_rate[8760*7:8760*8])))
    # plt.ylim((0, 1))
    # plt.xticks([720, 720 * 2, 720 * 3, 720 * 4, 720 * 5, 720 * 6, 720 * 7, 720 * 8, 720 * 9, 720 * 10, 720*11],
    #            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
    # plt.xlabel('t (months)')
    # plt.ylabel('R')
    #
    # plt.show()


    ## Work code
    # plt.xlim((0, len(coh_rate[8760*0:8760*1])))
    # plt.ylim((0, 1))
    # plt.xticks([720, 720 * 2, 720 * 3, 720 * 4, 720 * 5, 720 * 6, 720 * 7, 720 * 8, 720 * 9, 720 * 10, 720*11],
    #            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
    # plt.xlabel('t (months)')
    # plt.ylabel('R')
    # col = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'darkgreen']
    # for i in range(9):
    #     plt.scatter(numpy.arange(0, len(coh_rate[8760 * i:8760 * (i+1)])), numpy.asarray(coh_rate[8760 * i:8760 * (i+1)]),
    #                 marker='.', s=20, color = col[i], alpha=0.2)
    #
    # plt.show()


    return coh_rate

def CohVsGy(rg, co):
    le = 2420
    ri = 4737
    rg = rg[le:ri]
    co = co[le:ri]

    tmp = []
    aver_block = []
    for i in range(0, len(rg)):
        tmp.append([rg[i], co[i]])

    tmp_sort = sorted(tmp, key=itemgetter(0))
    dic_Rg = {}
    for i in range(len(tmp_sort)):
        if int(tmp_sort[i][0]) not in dic_Rg:
            dic_Rg[int(tmp_sort[i][0])] = []
            dic_Rg[int(tmp_sort[i][0])].append(tmp_sort[i])
        else:
            dic_Rg[int(tmp_sort[i][0])].append(tmp_sort[i])

    for i in dic_Rg.keys():
        lef = []
        rig = []
        for item in dic_Rg[i]:
            lef.append(item[0])
            rig.append(item[1])
        aver_block.append([sum(lef)/len(lef), sum(rig)/len(rig)])

    print(len(aver_block))

    #### For figures

    # new_rg = []
    # new_co = []
    # for i in aver_block:
    #     new_rg.append(i[0])
    #     new_co.append(i[1])
    #
    # plt.scatter(numpy.asarray(new_rg), numpy.asarray(new_co), marker= '.')
    # # plt.plot(numpy.asarray(new_rg), numpy.asarray(new_co))
    # plt.xlim((1, 900))
    # plt.ylim(0, 1)
    # plt.xlabel('Rg (km)')
    # plt.ylabel('r')
    # plt.show()

    ####################### Setting a interval
    aver_block_inter = []
    inter_gr = {}
    for i in range(len(aver_block)):
        if int(aver_block[i][0]//10) not in inter_gr:
            inter_gr[int(aver_block[i][0]//10)] = []
            inter_gr[int(aver_block[i][0]//10)].append(aver_block[i])
        else:
            inter_gr[int(aver_block[i][0]//10)].append(aver_block[i])

    for i in inter_gr.keys():
        lef = []
        rig = []
        for item in inter_gr[i]:
            lef.append(item[0])
            rig.append(item[1])
        aver_block_inter.append([sum(lef)/len(lef), sum(rig)/len(rig)])

    print(len(aver_block_inter))

    new_rg = []
    new_co = []
    for i in aver_block_inter:
        new_rg.append(i[0])
        new_co.append(i[1])

    # plt.scatter(numpy.asarray(new_rg), numpy.asarray(new_co), marker= '.')
    plt.plot(numpy.asarray(new_rg), numpy.asarray(new_co))
    plt.xticks([0, 2 **1, 2 **2, 2 **3, 2 **4, 2 **5, 2 **6, 2 **7, 2 **8, 2 **9, 2 **10],
               ['0', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])
    plt.xlim((1, 900))
    plt.ylim(0, 1)
    plt.xlabel('Rg (km)')
    plt.ylabel('r')
    plt.show()

    return aver_block

def CohVsGylog(rg, co):
    le = 48795
    ri = 51543
    rg = rg[le:ri]
    co = co[le:ri]

    tmp = []
    aver_block = []
    for i in range(0, len(rg)):
        tmp.append([rg[i], co[i]])

    tmp_sort = sorted(tmp, key=itemgetter(0))

    dic_dis = {}
    for ani in range(1, 10):
        dic_dis[ani] = []

    for item in tmp_sort:

        for i in range(1, 10):
            if int(item[0]/(2**i)) == 0:
                dic_dis[i].append(item)
                break
    total_lef = []
    total_rig = []
    for key in dic_dis.keys():
        if len(dic_dis[key]) != 0:
            l = 0
            r = 0
            for item in dic_dis[key]:
                l = l + item[0]
                r = r + item[1]
            l = l/len(dic_dis[key])
            r = r/len(dic_dis[key])
            total_lef.append(l)
            total_rig.append(r)

    # plt.scatter(numpy.asarray(new_rg), numpy.asarray(new_co), marker= '.')

    plt.plot(numpy.asarray(total_lef), numpy.asarray(total_rig))
    plt.xscale("log", basex = 2)
    plt.xticks([0, 2 **1, 2 **2, 2 **3, 2 **4, 2 **5, 2 **6, 2 **7, 2 **8, 2 **9, 2 **10],
               ['0', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])
    plt.xlim((1, max(total_lef)+100))
    plt.ylim(0, 1)
    plt.xlabel('Rg (km)')
    plt.ylabel('r')
    plt.title("6 -- 1")
    plt.show()


def CohVsGy_Interval_distr(rg, co):

    size = 10
    tmp = []
    for i in range(0, len(rg)):
        tmp.append([rg[i], co[i]])

    tmp_sort = sorted(tmp, key=itemgetter(0))
    inter_gr = {}

    for i in range(len(tmp_sort)):
        if int(tmp_sort[i][1] * 10) > 9:
            t = 9
        else:
            t = int(tmp_sort[i][1] * 10)

        if int(tmp_sort[i][0]//size) not in inter_gr:

            inter_gr[int(tmp_sort[i][0]//size)] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
            inter_gr[int(tmp_sort[i][0]//size)][t] = 1
        else:
            if t not in inter_gr[int(tmp_sort[i][0]//size)]:
                inter_gr[int(tmp_sort[i][0]//size)][t] = 1
            else:
                inter_gr[int(tmp_sort[i][0]//size)][t] = inter_gr[int(tmp_sort[i][0]//size)][t] + 1

    for key in inter_gr.keys():
        s = sum(list(inter_gr[key].values()))
        for k in inter_gr[key].keys():
            inter_gr[key][k] = inter_gr[key][k]/s

    output_dic = {}
    for key in inter_gr.keys():
        for k in inter_gr[key].keys():
            if k not in output_dic:
                output_dic[k] = []
                output_dic[k].append(inter_gr[key][k])
            else:
                output_dic[k].append(inter_gr[key][k])

    fig_list = []
    for key in output_dic.keys():
        fig_list.append(output_dic[key])

    trace = go.Heatmap(z = fig_list)
    data = [trace]
    py.iplot(data, filename = 'basic-heatmap') ### need to modify x and y axis ##


def CalPairCoh(single, l):
    s = []
    if len(l) == 0:
        return 0
    else:
        for i in l:
            s.append((pow(((single[4]+i[4])/2), 2) + pow(((single[5]+i[5])/2), 2))**0.5)

        return sum(s)/len(s)

def CalTwoCoh(l, r):
    return (pow(((l[4] + r[4]) / 2), 2) + pow(((l[5] + r[5]) / 2), 2)) ** 0.5

def LeaderFollower():

    id_dic = {}
    for item in dat:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i-1][1]
            rig = id_dic[key][i][2] - id_dic[key][i-1][2]
            r = (pow(lef, 2) + pow(rig, 2))**0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))

    count_dic = {}
    count = 0
    for item in dat_sort:
        if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.995:
            h = math.ceil((item[3] - int(item[3])) * 24)
        else:
            h = int((item[3] - int(item[3])) * 24)

        if int(item[3]) not in count_dic:
            count_dic[int(item[3])] = {}
            count_dic[int(item[3])][h] = count
            count = count + 1
        else:
            if h not in count_dic[int(item[3])]:
                count_dic[int(item[3])][h] = count
                count = count + 1
    time_dic = {}

    for item in dat_sort:

        if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.995:
            h = math.ceil((item[3] - int(item[3])) * 24)
        else:
            h = int((item[3] - int(item[3])) * 24)

        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = {}
            time_dic[int(item[3])][h] = []
            time_dic[int(item[3])][h].append(item)
        else:
            if h not in time_dic[int(item[3])]:
                time_dic[int(item[3])][h] = []
                time_dic[int(item[3])][h].append(item)
            else:
                time_dic[int(item[3])][h].append(item)

    id_dic_new = {}
    for item in dat_sort:
        if item[0] not in id_dic_new:
            id_dic_new[item[0]] = []
            id_dic_new[item[0]].append(item)
        else:
            id_dic_new[item[0]].append(item)

    all_animal = []
    for key in id_dic_new.keys():
        store_set = []
        for item in id_dic_new[key]:
            if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.995:
                h = math.ceil((item[3] - int(item[3])) * 24)
            else:
                h = int((item[3] - int(item[3])) * 24)
            ind = time_dic[int(item[3])][h].index(item)
            time_dic[int(item[3])][h].pop(ind)
            cal = time_dic[int(item[3])][h]
            ri = CalPairCoh(item, cal)
            store_set.append([key, int(item[3]), h, ri])
        all_animal.append(store_set)

    mat = []
    for i in range(321):
        mat.append([0]*89818)

    for i in range(len(all_animal)):
        for item in all_animal[i]:
            mat[i][count_dic[item[1]][item[2]]] = item[3]

    new_mat = []
    for i in range(len(mat)):
        new_mat.append(mat[i][8981*8:8981*10])

    trace = go.Heatmap(z = new_mat)
    data = [trace]
    py.iplot(data, filename = 'leader_follower_10') ### need to modify x and y axis ##

    
def pairwiseAnay():
    id_dic = {}
    for item in dat:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i-1][1]
            rig = id_dic[key][i][2] - id_dic[key][i-1][2]
            r = (pow(lef, 2) + pow(rig, 2))**0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    all_pair = {}
    for key in range(1, 31):
        print(key)
        all_pair[key] = {}
        for y in range(key+1, 31):
            all_pair[key][y] = []
            time_inter = {}
            l = id_dic[key]
            l.extend(id_dic[y])
            for item in l:

                if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.995:
                    h = math.ceil((item[3] - int(item[3])) * 24)
                else:
                    h = int((item[3] - int(item[3])) * 24)

                if int(item[3]) not in time_inter:
                    time_inter[int(item[3])] = {}
                    time_inter[int(item[3])][h] = []
                    time_inter[int(item[3])][h].append(item)
                else:
                    if h not in time_inter[int(item[3])]:
                        time_inter[int(item[3])][h] = []
                        time_inter[int(item[3])][h].append(item)
                    else:
                        time_inter[int(item[3])][h].append(item)
            for l_key in time_inter:
                for h_key in time_inter[l_key]:
                    if len(time_inter[l_key][h_key]) == 2:
                        dis = CalDis(time_inter[l_key][h_key][0][1], time_inter[l_key][h_key][0][2], time_inter[l_key][h_key][1][1], time_inter[l_key][h_key][1][2])
                        pair_coh = CalTwoCoh(time_inter[l_key][h_key][0], time_inter[l_key][h_key][1])
                        all_pair[key][y].append([dis, pair_coh])

    ind = 4
    plt.xlim((0, 2500))
    plt.ylim(0, 1)
    plt.xlabel('Dij (km)')
    plt.ylabel('rij')
    col = ['r', 'g', 'b', 'violet', 'm', 'y', 'k', 'w', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan', 'slateblue', 'purple', 'plum', 'olive', 'oldlace', 'paleturquoise', 'khaki', 'indigo', 'gold', 'gainsboro',
           'honeydew', 'hotpink', 'lightcoral', 'lightgoldenrodyellow', 'magenta', 'maroon']
    for i in range(ind+1, 31):
        n = sorted(all_pair[ind][i], key=itemgetter(0))
        lef = []
        rig = []
        for it in n:
            lef.append(it[0])
            rig.append(it[1])



        plt.scatter(numpy.asarray(lef), numpy.asarray(rig), marker= '.', s=20, color = col[i-ind-1], alpha=0.6)
    # plt.plot(numpy.asarray(lef), numpy.asarray(rig))
    # plt.semilogx(numpy.asarray(new_rg), numpy.asarray(new_co))

    plt.show()

def distOrigi():
    id_dic = {}
    for item in dat:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i - 1][1]
            rig = id_dic[key][i][2] - id_dic[key][i - 1][2]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))

    time_dic = {}

    for item in dat_sort:

        if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.99999995:
            h = math.ceil((item[3] - int(item[3])) * 24)
        else:
            h = int((item[3] - int(item[3])) * 24)

        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = {}
            time_dic[int(item[3])][h] = []
            time_dic[int(item[3])][h].append(item)
        else:
            if h not in time_dic[int(item[3])]:
                time_dic[int(item[3])][h] = []
                time_dic[int(item[3])][h].append(item)
            else:
                time_dic[int(item[3])][h].append(item)

    start_point = time_dic[731996][0]


    all_dis = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            lef, rig = CalCenPoint(time_dic[key][hour])
            dis = CalDis(lef, rig, start_point[0][1], start_point[0][2])
            all_dis.append(dis)

    # plt.scatter(numpy.arange(0, len(all_dis)), numpy.asarray(all_dis), marker='.', s=20)
    # plt.xlim((0, len(all_dis)))
    # plt.ylim((0, max(all_dis)))
    # plt.xticks([8760, 8760 * 2, 8760 * 3, 8760 * 4, 8760 * 5, 8760 * 6, 8760 * 7, 8760 * 8, 8760 * 9, 8760 * 10],
    #            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # plt.xlabel('t (years)')
    # plt.ylabel('d')
    #
    # plt.show()

    return all_dis

    # using plotly
    # Create a trace

    # trace = go.Scatter(
    #     x=numpy.arange(0, len(all_dis)),
    #     y=numpy.asarray(all_dis),
    #     mode='markers'
    # )
    #
    # data = [trace]
    #
    # # Plot and embed in ipython notebook!
    # py.iplot(data, filename='basic-scatter_new_2_year')

def nngraph():

    dic_dis = {}
    for ani in range(1, 322):
        dic_dis[ani] = {}
        for n in range(1, 322):
            dic_dis[ani][n] = 0

    id_dic = {}
    for item in dat:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i - 1][1]
            rig = id_dic[key][i][2] - id_dic[key][i - 1][2]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))

    time_dic = {}

    for item in dat_sort:

        if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.995:
            h = math.ceil((item[3] - int(item[3])) * 24)
        else:
            h = int((item[3] - int(item[3])) * 24)

        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = {}
            time_dic[int(item[3])][h] = []
            time_dic[int(item[3])][h].append(item)
        else:
            if h not in time_dic[int(item[3])]:
                time_dic[int(item[3])][h] = []
                time_dic[int(item[3])][h].append(item)
            else:
                time_dic[int(item[3])][h].append(item)

    for key in time_dic.keys():
        print(key)

        for hour in time_dic[key].keys():

            l = time_dic[key][hour]

            for item in range(len(l)):
                dic_ = (-1, 10000000)
                for new in range(len(l)):
                    if item != new:
                        new_dis = CalDis(l[item][1], l[item][2], l[new][1], l[new][2])
                        if new_dis < dic_[1]:
                            dic_ = (l[new][0], new_dis)
                if dic_[0] != -1:
                    dic_dis[l[item][0]][dic_[0]] = dic_dis[l[item][0]][dic_[0]] + 1

    all_list = []
    for key in dic_dis.keys():
        part_ = []
        for hour in dic_dis[key].keys():
            if dic_dis[key][hour] != 0:
                dic_dis[key][hour] = dic_dis[key][hour]/89817
            part_.append(dic_dis[key][hour])
        all_list.append(part_)


    trace = go.Heatmap(z = all_list)
    data = [trace]
    py.iplot(data, filename = 'nn-heatmap') ### need to modify x and y axis ##

def ShortDisFindLeader():

    co = Cohen()

    dic_dis = {}
    for ani in range(1, 322):
        dic_dis[ani] = {}
        for n in range(1, 322):
            dic_dis[ani][n] = 0

    id_dic = {}
    for item in dat:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i - 1][1]
            rig = id_dic[key][i][2] - id_dic[key][i - 1][2]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))

    time_dic = {}

    for item in dat_sort:

        if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.995:
            h = math.ceil((item[3] - int(item[3])) * 24)
        else:
            h = int((item[3] - int(item[3])) * 24)

        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = {}
            time_dic[int(item[3])][h] = []
            time_dic[int(item[3])][h].append(item)
        else:
            if h not in time_dic[int(item[3])]:
                time_dic[int(item[3])][h] = []
                time_dic[int(item[3])][h].append(item)
            else:
                time_dic[int(item[3])][h].append(item)


    count = 0

    for key in time_dic.keys():
        print(key)

        for hour in time_dic[key].keys():

            l = time_dic[key][hour]

            for item in range(len(l)):
                dic_ = (-1, 10000000)
                for new in range(len(l)):
                    if item != new:
                        new_dis = CalDis(l[item][1], l[item][2], l[new][1], l[new][2])
                        if new_dis < dic_[1]:
                            dic_ = (l[new][0], new_dis)
                if dic_[0] != -1:
                    dic_dis[l[item][0]][dic_[0]] = dic_dis[l[item][0]][dic_[0]] + dic_[1]/co[count]

            count = count + 1


    all_list = []
    for key in dic_dis.keys():
        part_ = []
        for hour in dic_dis[key].keys():
            if dic_dis[key][hour] != 0:
                dic_dis[key][hour] = dic_dis[key][hour]/89817
            part_.append(dic_dis[key][hour])
        all_list.append(part_)


    trace = go.Heatmap(z = all_list)
    data = [trace]
    py.iplot(data, filename = 'short_dis-heatmap') ### need to modify x and y axis ##

def StatNumAnimal():
    id_dic = {}
    for item in dat:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i - 1][1]
            rig = id_dic[key][i][2] - id_dic[key][i - 1][2]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))

    time_dic = {}

    for item in dat_sort:

        if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.9999999995:
            h = math.ceil((item[3] - int(item[3])) * 4)
        else:
            h = int((item[3] - int(item[3])) * 4)

        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = {}
            time_dic[int(item[3])][h] = []
            time_dic[int(item[3])][h].append(item)
        else:
            if h not in time_dic[int(item[3])]:
                time_dic[int(item[3])][h] = []
                time_dic[int(item[3])][h].append(item)
            else:
                time_dic[int(item[3])][h].append(item)

    num_animal = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            id = []
            for item in time_dic[key][hour]:
                id.append(item[0])
            h_num = list(set(id))
            num_animal.append(len(h_num))


    plt.scatter(numpy.arange(0, len(num_animal)), numpy.asarray(num_animal), marker='.', s=20)
    plt.xlim((0, len(num_animal)))
    plt.ylim((0, max(num_animal)))
    # plt.xticks([8760, 8760 * 2, 8760 * 3, 8760 * 4, 8760 * 5, 8760 * 6, 8760 * 7, 8760 * 8, 8760 * 9, 8760 * 10],
    #            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    plt.xticks([8760 * (1/6), 8760 * 2 * (1/6), 8760 * 3 * (1/6), 8760 * 4 * (1/6), 8760 * 5 * (1/6), 8760 * 6 * (1/6), 8760 * 7 * (1/6), 8760 * 8 * (1/6), 8760 * 9 * (1/6), 8760 * 10 * (1/6)],
               ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    plt.xlabel('t (years)')
    plt.ylabel('The number of animals')

    plt.show()
    # using plotly
    # Create a trace
    # trace = go.Scatter(
    #     x=numpy.arange(0, len(num_animal)),
    #     y=numpy.asarray(num_animal),
    #     mode='markers'
    # )
    #
    # data = [trace]
    #
    # # Plot and embed in ipython notebook!
    # py.iplot(data, filename='num_animal-scatter--interval_2')

def DynamicPic():
    id_dic = {}
    for item in dat:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i - 1][1]
            rig = id_dic[key][i][2] - id_dic[key][i - 1][2]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))

    time_dic = {}

    for item in dat_sort:

        if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.99999995:
            h = math.ceil((item[3] - int(item[3])) * 24)
        else:
            h = int((item[3] - int(item[3])) * 24)

        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = {}
            time_dic[int(item[3])][h] = []
            time_dic[int(item[3])][h].append(item)
        else:
            if h not in time_dic[int(item[3])]:
                time_dic[int(item[3])][h] = []
                time_dic[int(item[3])][h].append(item)
            else:
                time_dic[int(item[3])][h].append(item)

    center_point = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            lef, rig = CalCenPoint(time_dic[key][hour])
            center_point.append([lef, rig])

    for i in range(1, len(center_point)):
        lef = center_point[i][0] - center_point[i-1][0]
        rig = center_point[i][1] - center_point[i-1][1]

        r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
        s = lef / r
        c = rig / r
        center_point[i].append(s)
        center_point[i].append(c)
    center_point[0].append(0)
    center_point[0].append(0)
    return time_dic, center_point

def CountAnimalEachYear():
    time_dic, center_point = DynamicPic()
    total = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            total.append(time_dic[key][hour])
    total = total[25307:33375]
    id = []
    for item in total:
        for i in item:
            id.append(i[0])
    # print(sorted(list(set(id))))
    # print(len(set(id)))
    i_dic = {}
    count = 0
    for item in sorted(list(set(id))):
        if item not in i_dic:
            i_dic[item] = count
            count = count + 1
    return i_dic


def DrawMap():

    i_dic = CountAnimalEachYear()
    all_dis = distOrigi()
    time_dic, center_point = DynamicPic()
    symb = ['o', 'D', 'h', '8', 'p', '+', 's', '*', 'd', 'v', '<', '>', '^', 'x', '.', ',', '1', '2', '3', '4', 'd', 'H', '_', '|', 'D', 'x', 'o', 'D', 'h', '8', 'p', '+', 's', '*', 'd', 'v', '<', '>',  '.', ',', '1', '2', '3']
    col = ['r', 'g', 'b', 'violet', 'm', 'y', 'k', 'w', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan', 'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'w', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan', 'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'w', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan', 'slateblue']
    count = 0
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            if count >= 25307:
                f, axarr = plt.subplots(2, 1, figsize = (13, 13))

                for item in time_dic[key][hour]:
                    new_lef = item[1] - center_point[count][0]
                    new_rig = item[2] - center_point[count][1]
                    n_l = new_lef * (-center_point[count][2]) - new_rig * center_point[count][3]
                    n_r = new_lef * center_point[count][3] + new_rig * (-center_point[count][2])
                    axarr[0].scatter(n_l, n_r, marker=symb[i_dic[item[0]]], color = col[i_dic[item[0]]], s=30)

                axarr[0].set_xlim((-30, 30))
                axarr[0].set_ylim((-30, 30))
                axarr[0].set_xlabel('lan')
                axarr[0].set_ylabel('lon')
                axarr[0].set_title("hour: "+str(count-25307))

                axarr[1].scatter(numpy.arange(0, len(all_dis[25307:33375])), numpy.asarray(all_dis[25307:33375]), s=1)
                axarr[1].plot([count-25307, count-25307], [0, max(all_dis[25307:33375])], 'r--')
                axarr[1].set_xlim((0, len(all_dis[25307:33375])))
                axarr[1].set_ylim((0, max(all_dis[25307:33375])))
                axarr[1].set_xticks(
                    [876, 876 * 2, 876 * 3, 876 * 4, 876 * 5, 876 * 6, 876 * 7, 876 * 8, 876 * 9, 876 * 10],
                    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
                axarr[1].set_xlabel('t (years)')
                axarr[1].set_ylabel('d')


                plt.savefig('/Users/peis/Documents/Te2/' + str(count))
                plt.close()

                print("finish " + str(count))
            count = count + 1
            if count >= 33375:
                exit()

def GenerateGIF():
    fileList = os.listdir('/Users/peis/Documents/Te/')
    im = Image.open('/Users/peis/Documents/Te/0.png')
    im.save('test.gif', save_all = True, append_images = [Image.open('/Users/peis/Documents/Te/'+str(f)+'.png') for f in range(1, len(fileList)-5)], duration = 150)

def GenerateVideo():
    fileList = os.listdir('/Users/peis/Documents/Te3/')
    videoWriter = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, (1600, 1200))
    # for f in range(firstP, len(fileList)+firstP):
    for f in range(0, len(fileList)):
        print(f)
        img = cv2.imread('/Users/peis/Documents/Te3/'+str(f)+'.png')
        img = cv2.resize(img, (1600, 1200))
        videoWriter.write(img)
    cv2.destroyAllWindows()
    videoWriter.release()


def Data_interpolation():
    new_point = []

    id_dic = {}
    for item in dat:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for id in id_dic.keys():
        each_anim = id_dic[id]
        each_anim = sorted(each_anim, key=itemgetter(3))
        for i in range(1, len(each_anim)):
            time = int((each_anim[i][3] - int(each_anim[i][3])) * 24) - int((each_anim[i-1][3] - int(each_anim[i-1][3])) * 24)
            day = int(each_anim[i][3]) - int(each_anim[i-1][3])
            if day == 0 and time > 1:
                interval = time
                inter_lef = (each_anim[i][1] - each_anim[i-1][1])/interval
                inter_rig = (each_anim[i][2] - each_anim[i-1][2])/interval
                for point in range(1, time-1):
                    l = each_anim[i-1][1] + inter_lef * point
                    r = each_anim[i-1][2] + inter_rig * point
                    t = (int((each_anim[i-1][3] - int(each_anim[i-1][3])) * 24) + point)/24 + 0.01 + int(each_anim[i-1][3])
                    new_point.append([id, l, r, t])
            elif day != 0:
                interval = int((each_anim[i][3] - int(each_anim[i][3])) * 24) + (23 - int((each_anim[i-1][3] - int(each_anim[i-1][3])) * 24)) + (day - 1) * 24 + 1
                if interval > 1:
                    inter_lef = (each_anim[i][1] - each_anim[i-1][1])/interval
                    inter_rig = (each_anim[i][2] - each_anim[i-1][2])/interval
                    for point in range(1, interval-1):
                        l = each_anim[i-1][1] + inter_lef * point
                        r = each_anim[i-1][2] + inter_rig * point
                        t = (int((each_anim[i-1][3] - int(each_anim[i-1][3])) * 24) + point)/24 + 0.01 + int(each_anim[i-1][3])
                        new_point.append([id, l, r, t])
    dat.extend(new_point)
    return dat

def StayEndAppend(time_dic, begin, end):
    count = 0
    end_item = {}
    append_ele = []
    break_flag = False
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            if count >= begin:
                for name in time_dic[key][hour].keys():
                    for item in time_dic[key][hour][name]:
                        if item[0] not in end_item:
                            end_item[item[0]] = [count, item]
                        else:
                            end_item[item[0]] = [count, item]
            count = count + 1
            if count >= end:
                break_flag = True
                # print(end_item)
                break
        if break_flag == True:
            break

    for name in end_item.keys():
        item = end_item[name][1]
        id = item[0]
        l = item[1]
        r = item[2]
        s = item[4]
        c = item[5]
        # if end_item[name][0] - begin > 100:
        for i in range(1, end - end_item[name][0] + 1):
            new_time = (int((item[3] - int(item[3])) * 4) + i) / 4 + int(item[3]) + 0.001
            new_item = [id, l, r, new_time, s, c]
            append_ele.append(new_item)
    return append_ele

def StayatEnd(time_dic):
    begin = firstP
    end = secondP
    append_ele = StayEndAppend(time_dic, begin, end)
    begin = secondP
    end = thirdP
    append_ele.extend(StayEndAppend(time_dic, begin, end))

    for item in append_ele:

        if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.99999995:
            h = math.ceil((item[3] - int(item[3])) * 4)
        else:
            h = int((item[3] - int(item[3])) * 4)

        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = {}
            time_dic[int(item[3])][h] = {}
            time_dic[int(item[3])][h][item[0]] = []
            time_dic[int(item[3])][h][item[0]].append(item)
        else:
            if h not in time_dic[int(item[3])]:
                time_dic[int(item[3])][h] = {}
                time_dic[int(item[3])][h][item[0]] = []
                time_dic[int(item[3])][h][item[0]].append(item)
            else:
                if item[0] not in time_dic[int(item[3])][h]:
                    time_dic[int(item[3])][h][item[0]] = []
                    time_dic[int(item[3])][h][item[0]].append(item)
                else:
                    time_dic[int(item[3])][h][item[0]].append(item)

    return time_dic

def new_original_distance_inter6():
    dat = Data_interpolation()
    id_dic = {}
    for item in dat:
        # if item[0] in user_id_year_long_for_distance[y-1]: ######################
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    # for key in id_dic.keys():
    #     id_dic[key][0].append(0)
    #     id_dic[key][0].append(0)
    #     for i in range(1, len(id_dic[key])):
    #         lef = id_dic[key][i][1] - id_dic[key][i - 1][1]
    #         rig = id_dic[key][i][2] - id_dic[key][i - 1][2]
    #         r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
    #         s = lef / r
    #         c = rig / r
    #         id_dic[key][i].append(s)
    #         id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))

    time_dic = {}

    for item in dat_sort:

        if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.99999995:
            h = math.ceil((item[3] - int(item[3])) * 4)
        else:
            h = int((item[3] - int(item[3])) * 4)

        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = {}
            time_dic[int(item[3])][h] = {}
            time_dic[int(item[3])][h][item[0]] = []
            time_dic[int(item[3])][h][item[0]].append(item)
        else:
            if h not in time_dic[int(item[3])]:
                time_dic[int(item[3])][h] = {}
                time_dic[int(item[3])][h][item[0]] = []
                time_dic[int(item[3])][h][item[0]].append(item)
            else:
                if item[0] not in time_dic[int(item[3])][h]:
                    time_dic[int(item[3])][h][item[0]] = []
                    time_dic[int(item[3])][h][item[0]].append(item)
                else:
                    time_dic[int(item[3])][h][item[0]].append(item)

    # start_point = [2, 37.116396, -122.330756, 731996.0, 0, 0]
    start_point = 0

    # time_dic = StayatEnd(time_dic) #//////need to be modified

    all_dis = []
    center_point = []
    for key in sorted(time_dic.keys()):
        for hour in time_dic[key].keys():
            new_li = []
            for name in time_dic[key][hour].keys():

                l = time_dic[key][hour][name]
                l_lef, l_rig = CalCenPoint(l)
                new_li.append([name, l_lef, l_rig])
            lef, rig = CalCenPoint(new_li)
            center_point.append([lef, rig])
            if start_point == 0: #####
                start_point = [lef, rig] #####
            # dis = CalDis(lef, rig, start_point[1], start_point[2])
            dis = CalDis(lef, rig, start_point[0], start_point[1])
            all_dis.append(dis)

    for i in range(1, len(center_point)):
        lef =  center_point[i][0] - center_point[i-1][0]
        rig =  center_point[i][1] - center_point[i-1][1]

        r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
        s = lef / r
        c = rig / r
        center_point[i].append(s)
        center_point[i].append(c)
    center_point[0].append(0)
    center_point[0].append(0)

    return time_dic, all_dis, center_point

    # plt.scatter(numpy.arange(0, len(all_dis)), numpy.asarray(all_dis), marker='.', s=20)
    # plt.xlim((0, len(all_dis)))
    # plt.ylim((0, max(all_dis)))
    # plt.xticks([8760 * (1 / 6), 8760 * 2 * (1 / 6), 8760 * 3 * (1 / 6), 8760 * 4 * (1 / 6), 8760 * 5 * (1 / 6),
    #             8760 * 6 * (1 / 6), 8760 * 7 * (1 / 6), 8760 * 8 * (1 / 6), 8760 * 9 * (1 / 6), 8760 * 10 * (1 / 6)],
    #            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # plt.xlabel('t (years)')
    # plt.ylabel('d')
    #
    # plt.show()

    # using plotly
    # Create a trace

    # trace = go.Scatter(
    #     x=numpy.arange(0, len(all_dis)),
    #     y=numpy.asarray(all_dis),
    #     mode='markers'
    # )
    #
    # data = [trace]
    #
    # # Plot and embed in ipython notebook!
    # py.iplot(data, filename='basic-scatter_new_6_year_Interpolation')

def CountAnimalEachYearNew(first=1779, third=2802):
    time_dic, all_dis, center_point = new_original_distance_inter6()
    total = []
    for key in sorted(time_dic.keys()):
        for hour in time_dic[key].keys():
            l = list(time_dic[key][hour].keys())
            total.append(l)
    # total = total[first:third] ########need to be modified
    id = []
    for item in total:
        id.extend(item)
    i_dic = {}
    count = 0
    for item in sorted(list(set(id))):
        if item not in i_dic:
            i_dic[item] = count
            count = count + 1

    print(all_dis)
    return i_dic, all_dis, time_dic, center_point

def CountAnimal_AllYearNew():

    time_dic, all_dis, center_point = new_original_distance_inter6()
    total = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            l = list(time_dic[key][hour].keys())
            total.append(l)
    id = []
    for item in total:
        id.extend(item)
    i_dic = {}
    count = 0
    for item in sorted(list(set(id))):
        if item not in i_dic:
            i_dic[item] = count
            count = count + 1

    return i_dic, all_dis, time_dic, center_point


def DrawMapBaseMap():
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec
    import numpy as np
    # m = Basemap(projection='mill',
    #             llcrnrlat=25,
    #             llcrnrlon=-210,
    #             urcrnrlat=63,
    #             urcrnrlon=-110)
    # m.drawcoastlines()

    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew(firstP, thirdP)

    symb = ['o', 'D', 'h', '8', 'p', '+', 's', '*', 'd', 'v', '<', '>', '^', 'x', '.', ',', '1', '2', '3', '4', 'd',
            'H', '_', '|', 'D', 'x', 'o', 'D', 'h', '8', 'p', '+', 's', 'o', 'D', 'h', '8', 'p', '+', 's', '*', 'd', 'v', '<', '>', '^', 'x', '.', ',', '1', '2', '3', '4', 'd',
            'H', '_', '|', 'D', 'x', 'o', 'D', 'h', '8', 'p', '+', 's']
    col = ['r', 'g', 'b', 'violet', 'm', 'y', 'k', 'thistle', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan',
           'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'r', 'orange', 'brown', 'tomato', 'thistle', 'teal',
           'tan', 'slateblue', 'r', 'g', 'b', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'thistle', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan',
           'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'r', 'orange', 'brown', 'tomato', 'thistle', 'teal',
           'tan', 'slateblue', 'r', 'g', 'b']
    count = 0
    cen_tra = []
    for key in sorted(time_dic.keys()):
        for hour in sorted(time_dic[key].keys()):
            if count >= firstP - firstP:
                cen_tra.append(center_point[count])
                # f, axarr = plt.subplots(2, 2, figsize=(13, 13))
                gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1, 1],
                       height_ratios=[1, 1])
                pl.figure()
                ax = pl.subplot(gs[1, :])
                m = Basemap(ax= ax, projection='mill',
                llcrnrlat=25,
                llcrnrlon=-210,
                urcrnrlat=63,
                urcrnrlon=-110, resolution = 'c', epsg=4269)
                # m.arcgisimage(service='Ocean_Basemap', xpixels=1500, verbose=True)
                m.drawcoastlines()
                m.fillcontinents(color='coral')
                # draw parallels and meridians.
                # m.drawmapboundary(fill_color='aqua')

                # if count > secondP:
                #     cen_tra_ = cen_tra[secondP-firstP:]
                #     for j in range(len(cen_tra_)):
                #         l_c = cen_tra_[j][0]
                #         r_c = cen_tra_[j][1]
                #         x_c, y_c = m(r_c, l_c)
                #         m.plot(x_c, y_c, marker='.', color='r', markersize=1)
                # elif count >= firstP and count <= secondP:
                #     for j in range(len(cen_tra)):
                #         l_c = cen_tra[j][0]
                #         r_c = cen_tra[j][1]
                #         x_c, y_c = m(r_c, l_c)
                #         m.plot(x_c, y_c, marker='.', color='r', markersize=1)

                ax1 = pl.subplot(gs[0, 1])
                for name in time_dic[key][hour].keys():
                    # if name != 37 and name != 53 and name != 48:###############
                    if name != 405 and name != 402 and name != 416:
                        l = time_dic[key][hour][name]
                        l_lef, l_rig = CalCenPoint(l)
                        xpt, ypt = m(l_rig, l_lef)
                        m.plot(xpt, ypt, marker = symb[i_dic[name]], color = col[i_dic[name]], markersize = 5)
                        ax.annotate(male_list[name], xy = (xpt, ypt), xycoords='data', xytext = (xpt+0.5, ypt+0.5), fontsize=6)
                        ## add the trajectory of center points
                        if isnan(float(center_point[count][2])) and isnan(float(center_point[count][3])):
                            n_l = -0
                            n_r = 0
                        else:
                            new_lef = l_lef - center_point[count][0]
                            new_rig = l_rig - center_point[count][1]
                            n_l = new_lef * (-center_point[count][2]) - new_rig * center_point[count][3]
                            n_r = new_lef * center_point[count][3] - new_rig * (center_point[count][2])
                        ax1.scatter(-n_r, -n_l, marker=symb[i_dic[name]], color=col[i_dic[name]], s=8)
                ax.plot()
                ax1.set_xlim((-50, 50))
                ax1.set_ylim((-50, 50))
                # ax1.set_xlabel('lan')
                # ax1.set_ylabel('lon')
                # ax1.set_title("hour: " + str(count - firstP))
                dd = date.fromordinal(key)
                w = dd.isoformat()
                ax1.set_title("Time: " + str(dd.month) + '-' + str(dd.day) + " " + str(hour*6)+'h')

                ax = pl.subplot(gs[0, 0])
                # ax.scatter(numpy.arange(0, len(all_dis[firstP - firstP:thirdP - firstP])), numpy.asarray(all_dis[firstP - firstP:thirdP - firstP]), s=1)
                ax.scatter(numpy.arange(0, len(all_dis)), numpy.asarray(all_dis), s=1)
                # ax.plot([count - firstP + firstP, count - firstP + firstP], [0, max(all_dis[firstP - firstP:thirdP - firstP])], 'r--')
                ax.plot([count, count], [0, max(all_dis)], 'r--')
                # ax.set_xlim((0, len(all_dis[firstP - firstP:thirdP - firstP])))
                # ax.set_ylim((0, max(all_dis[firstP - firstP:thirdP - firstP])))
                ax.set_xlim((0, len(all_dis)))
                ax.set_ylim((0, max(all_dis)))
                ax.xaxis.set_ticks_position('top')
                # ax.set_xticks(
                #     [138, 138 * 2, 138 * 3, 138 * 4, 138 * 5, 138 * 6, 138 * 7, 138 * 8, 138 * 9, 138 * 10],
                #     ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
                ax.set_xlabel('t (hour)', fontsize=10)
                ax.set_ylabel('d (km)', fontsize=10)

                pl.savefig('/Users/peis/Documents/Te3/' + str(count), dpi=(200))
                pl.close()
                # F.title("hour: "+str(count-0))
                # F.close()
                # plt.title("hour: "+str(count-0))
                # plt.savefig('/Users/peis/Documents/Te3/' + str(count))
                # plt.close()

                print("finish " + str(count))
            count = count + 1
            # if count >= thirdP - firstP:    ####################### comment
            #     exit()
    plt.show()


def disTocenterHeapmap():
    count = 0
    heap_map = {}
    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew()
    print(i_dic.keys())
    exit()
    for user in i_dic.keys():
        heap_map[user] = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            if all_dis[count] >= 500 and count >= secondP and count <= thirdP:
                each_time = {}
                total_heap = {}
                for name in time_dic[key][hour].keys():
                    l = time_dic[key][hour][name]
                    l_lef, l_rig = CalCenPoint(l)
                    each_time[name] = CalDis(l_lef, l_rig, center_point[count][0], center_point[count][1])
                sort_each_time = sorted(each_time.items(), key=operator.itemgetter(1))
                sort_ranking = {}
                for i in range(len(sort_each_time)):
                    sort_ranking[sort_each_time[i][0]] = i + 1
                for user in i_dic.keys():
                    total_heap[user] = 0
                for user in sort_ranking.keys():
                    total_heap[user] = sort_ranking[user]

                for keys in total_heap.keys():
                    heap_map[keys].append(total_heap[keys])
            count += 1
    if r == True:
        heap_map.pop(remove)
    rec_key = []
    for k in heap_map.keys():
        if sum(heap_map[k]) == 0:
            rec_key.append(k)

    for item in rec_key:
        heap_map.pop(item)

    for key in heap_map.keys():
        for i in range(len(heap_map[key])):
            if heap_map[key][i] == 0:
                heap_map[key][i] = len(list(heap_map.keys())) + 1

    feed = []
    sum_heatmap = {}
    name_key = []
    for key in heap_map.keys():
        sum_heatmap[key] = sum(heap_map[key])
    sort_sum_heatmap = sorted(sum_heatmap.items(), key=operator.itemgetter(1))

    for key in sort_sum_heatmap:
        feed.append(list(heap_map[key[0]]))
        name_key.append('Id: '+str(key[0]))
    x = []
    for u in range(len(feed[0])):
        x.append(str(u))
    # need to plot in python
    data = [
        go.Heatmap(
            z = feed,
            x = x,
            y = name_key,
            colorscale = 'Viridis',
        )
    ]
    layout = go.Layout(
        title=Year,
        xaxis=dict(ticks='', nticks=36),
        yaxis=dict(ticks='')
    )

    fig = go.Figure(data = data, layout=layout)
    py.iplot(fig, filename = Year)

def GetLongVector():
    count = 0
    heap_map = {}
    center_l = []
    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew(first=secondP, third=thirdP)
    for user in i_dic.keys():
        heap_map[user] = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            if all_dis[count] >= 500 and count > secondP and count <= thirdP:
                each_time = {}
                total_heap = {}
                center_l.append(center_point[count])
                for name in time_dic[key][hour].keys():
                    l = time_dic[key][hour][name]
                    l_lef, l_rig = CalCenPoint(l)
                    each_time[name] = l_rig

                for user in i_dic.keys():
                    total_heap[user] = 0
                for user in each_time.keys():
                    total_heap[user] = each_time[user]

                for keys in total_heap.keys():
                    heap_map[keys].append(total_heap[keys])
            count += 1
    if r == True:
        heap_map.pop(remove)
    rec_key = []
    for k in heap_map.keys():
        if sum(heap_map[k]) == 0:
            rec_key.append(k)

    for item in rec_key:
        heap_map.pop(item)

    # for key in heap_map.keys():
    #     flg = 0
    #     for i in range(len(heap_map[key])):
    #         if heap_map[key][i] != 0:
    #             flg = i
    #             break
    #     if flg != 0:
    #         for j in range(i):
    #             heap_map[key][j] = heap_map[key][i]
    # for key in heap_map.keys():
    #     flg = 0
    #     for i in range(len(heap_map[key])):
    #         if heap_map[key][i] == 0:
    #             flg = i
    #             break
    #     if flg != 0:
    #         for j in range(i, len(heap_map[key])):
    #             heap_map[key][j] = heap_map[key][i-1]
    print(len(center_l))
    return heap_map

def GetEachUserLocationVector():
    count = 0
    heap_map = {}
    center_l = []
    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew(first=secondP, third=thirdP)
    for user in i_dic.keys():
        heap_map[user] = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            if all_dis[count] >= 500 and count > secondP and count <= thirdP:
                each_time = {}
                total_heap = {}
                center_l.append(center_point[count])
                for name in time_dic[key][hour].keys():
                    l = time_dic[key][hour][name]
                    l_lef, l_rig = CalCenPoint(l)
                    each_time[name] = [l_lef, l_rig]

                for user in i_dic.keys():
                    total_heap[user] = []
                for user in each_time.keys():
                    total_heap[user] = each_time[user]

                for keys in total_heap.keys():
                    heap_map[keys].append(total_heap[keys])
            count += 1
    if r == True:
        heap_map.pop(remove)
    rec_key = []
    for k in heap_map.keys():
        ll = []
        for item in heap_map[k]:
            ll.append(len(item))
        if sum(ll) == 0:
            rec_key.append(k)
    for item in rec_key:
        heap_map.pop(item)
    for key in heap_map.keys():
        flg = 0
        for i in range(len(heap_map[key])):
            if len(heap_map[key][i]) != 0:
                flg = i
                break
        if flg != 0:
            for j in range(i):
                heap_map[key][j] = heap_map[key][i]
    for key in heap_map.keys():
        flg = 0
        for i in range(len(heap_map[key])):
            if len(heap_map[key][i]) == 0:
                flg = i
                break
        if flg != 0:
            for j in range(i, len(heap_map[key])):
                heap_map[key][j] = heap_map[key][i - 1]

    return heap_map, center_l

def NewdisTocenterHeapmap():
    vectors, center_point = GetEachUserLocationVector()
    # vectors.pop(3)
    # vectors.pop(251)

    heap_map = {}
    for key in vectors.keys():
        heap_map[key] = []

    ex = list(vectors.keys())[0]

    for i in range(len(vectors[ex])):
        each_time = {}
        total_heap = {}
        for key in vectors.keys():
            each_time[key] = CalDis(vectors[key][i][0], vectors[key][i][1], center_point[i][0], center_point[i][1])

        sort_each_time = sorted(each_time.items(), key=operator.itemgetter(1))
        sort_ranking = {}
        for i in range(len(sort_each_time)):
            sort_ranking[sort_each_time[i][0]] = i + 1
        for user in vectors.keys():
            total_heap[user] = 0
        for user in sort_ranking.keys():
            total_heap[user] = sort_ranking[user]

        for keys in total_heap.keys():
            heap_map[keys].append(total_heap[keys])

    feed = []
    sum_heatmap = {}
    name_key = []
    for key in heap_map.keys():
        sum_heatmap[key] = sum(heap_map[key])
    sort_sum_heatmap = sorted(sum_heatmap.items(), key=operator.itemgetter(1), reverse=True)
    tmp = sort_sum_heatmap[3]
    sort_sum_heatmap[3] = sort_sum_heatmap[4]
    sort_sum_heatmap[4] = tmp
    what = [250, 246, 243, 240, 241, 248, 237, 238, 239, 245, 242, 247, 244, 249]
    # what = [9, 7, 12, 15, 13, 10, 8, 14, 11, 6]
    # what = [297, 304, 290, 294,  296, 295,302, 299, 291, 298, 303,301,293, 292,300]
    # what = [273, 271, 276, 268, 267, 278, 269, 274, 277, 272, 270, 275]
    # what = [216, 200, 203, 204, 205, 207, 214, 206, 210, 211, 212, 213, 215, 209, 208]
    # what = [184, 180, 182, 183, 181, 179]
    # what = [161, 160, 156, 158, 159, 157, 152, 154, 151, 155, 153]
    # what = [129, 124, 111, 122, 128, 112, 115, 110, 126, 121, 127, 123, 116, 117, 113, 118, 114, 119, 125, 120]
    # what = [78, 82, 83, 89, 91, 90, 80, 84, 79, 85, 81, 86, 88, 87,92]
    # what = [48, 53, 58, 37, 42, 33, 51, 31, 43, 35, 39, 56, 40, 44, 47, 57, 46, 55, 52, 38, 45, 41, 50, 36, 49, 32, 54, 34]
    for key in reversed(what):
        feed.append(list(heap_map[key]))
        name_key.append('Id: '+str(key))
    # vvvv = name_key[1:]
    # vvvv.insert(2, name_key[0])

    x = []
    zzzz = 0
    for u in range(len(feed[0])):
        if u % 4 == 0:
            zzzz += 1
            x.append(str(zzzz))
    # y = x[1:]
    # y.insert(2, x[0])

    # need to plot in python
    # data = [
    #     go.Heatmap(
    #         z = feed,
    #         x = x,
    #         y = name_key,
    #         colorscale = 'Viridis',
    #     )
    # ]
    # layout = go.Layout(
    #     title=Year,
    #     xaxis=dict(tickvals=[80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880],
    #             ticktext=['20', '40', '60', '80', '100', '120',
    #                       '140', '160', '180', '200', '220']),
    #     yaxis=dict(ticks='')
    # )
    #
    # fig = go.Figure(data = data, layout=layout)
    # py.iplot(fig, filename = Year)
    plt.figure(figsize=(4, 6))
    import seaborn as sns
    sns.set()
    g = sns.heatmap(np.array(feed), yticklabels= np.array(name_key), cmap="viridis")
    # g.set(xticklabels = [80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880])
    # g.set(xticklabels = ['20', '40', '60', '80', '100', '120',
    #                        '140', '160', '180', '200', '220'])
    plt.xticks([80, 160, 240, 320, 400, 480, 560, 640, 720, 800], ['20', '40', '60', '80', '100', '120',
                           '140', '160', '180', '200'], rotation=315)
    plt.yticks(rotation=0, fontsize = 15)
    # plt.xlabel('Time (days)', fontsize = 20, fontname = 'times new roman')
    plt.tight_layout()
    plt.show()


def GetRemoveAnimal(i_dic, all_dis, time_dic, center_point, second=1779, third=2802):
    count = 0
    heap_map = {}
    # i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew()
    for user in i_dic.keys():
        heap_map[user] = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            # all_dis[count] >= 500 and
            if count > second and count <= third:
                each_time = {}
                total_heap = {}
                for name in time_dic[key][hour].keys():
                    l = time_dic[key][hour][name]
                    l_lef, l_rig = CalCenPoint(l)
                    each_time[name] = l_rig

                for user in i_dic.keys():
                    total_heap[user] = 0
                for user in each_time.keys():
                    total_heap[user] = each_time[user]

                for keys in total_heap.keys():
                    heap_map[keys].append(total_heap[keys])
            count += 1
    if r == True:
        heap_map.pop(remove)
    rec_key = []
    for k in heap_map.keys():
        if sum(heap_map[k]) == 0:
            rec_key.append(k)

    for item in rec_key:
        heap_map.pop(item)

    return rec_key

def CalculateCorrleation():
    import numpy as np
    import seaborn as sns
    import pandas as pd
    sns.set(color_codes=True)
    vectors = GetLongVector()
    array_list = []
    name_ = []
    for key in vectors.keys():
        if key != 217 and key != 251:
            name_.append(key)
            array_list.append(np.array(vectors[key]))

    array_list = np.array(array_list)
    cof = np.corrcoef(array_list)
    pa = {}
    for i in range(len(cof)):
        pa[name_[i]] = cof[i]
    df = pd.DataFrame(data=pa, index = np.array(name_))
    g = sns.clustermap(df, metric='euclidean', cmap="YlGnBu")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize = 15)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, fontsize=15)
    plt.show()

    # sum_cof = []
    # for item in cof:
    #     sum_cof.append(sum(item)/len(item))
    # print(sum_cof)
    # return cof
    # trace = go.Heatmap(z = cof)
    # data = [trace]
    # py.iplot(data, filename = 'correlation-heatmap-y7') ### need to modify x and y axis ##

def timeLaggedPlot():
    import numpy as np
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec

    vectors = GetLongVector()

    v1 = vectors[204]
    v2 = vectors[205]
    v3 = vectors[208]
    v4 = vectors[215]
    plot_vec = [v1, v2, v3, v4]

    gs = gridspec.GridSpec(4, 4)

    for i in range(len(plot_vec)):
        for j in range(len(plot_vec)):
            new_value = []
            ax = pl.subplot(gs[i, j])
            value = numpy.correlate(np.array(plot_vec[i]), np.array(plot_vec[j]), "full")
            value_max = max(value)
            for item in value:
                new_value.append(item / value_max)
            ax.plot(numpy.arange(0, len(new_value)), numpy.asarray(new_value))
            ax.set_title(str(i) + " " + str(j))

    pl.show()

def MatrixW():
    import numpy as np
    vectors = GetLongVector()
    name_ = list(vectors.keys())
    user_tau = {}
    for i in range(len(name_)):
        tau_ = []
        for j in range(len(name_)):
            length = len(vectors[name_[i]])
            corr = numpy.correlate(np.array(vectors[name_[i]]), np.array(vectors[name_[j]]), "full")
            corr = corr[length - 9: length + 8]
            index, value = max(enumerate(corr), key=operator.itemgetter(1))
            tau = index - 8
            tau_.append(tau/4)
        user_tau[name_[i]] = np.array(tau_)

    import seaborn as sns
    name_heat = []
    feed = []
    for key in year8:
        name_heat.append(key)
        feed.append(user_tau[key])
    sns.set()
    sns.heatmap(np.array(feed), annot=True, xticklabels=np.array(name_heat), yticklabels= np.array(name_heat))
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.show()

def average_time(l):
    total_time = 0
    for item in l:
        total_time = total_time + item[3]
    aver_time = total_time/len(l)
    return aver_time

def SpeedDistribution(i_dic, all_dis, time_dic, center_point):
    count = 0
    heap_map = {}
    remove_list = GetRemoveAnimal(i_dic, all_dis, time_dic, center_point)
    # i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew()
    for user in i_dic.keys():
        heap_map[user] = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            if all_dis[count] >= 500 and count >= secondP and count <= thirdP:
                for name in time_dic[key][hour].keys():
                    l = time_dic[key][hour][name]
                    l_lef, l_rig = CalCenPoint(l)
                    t = average_time(l)
                    heap_map[name].append([l_lef, l_rig, t])
            count += 1
    if r == True:
        heap_map.pop(remove)
    for item in remove_list:
        heap_map.pop(item)

    s = []
    for key in heap_map.keys():
        for i in range(1, len(heap_map[key])):
            d = CalDis(lef= heap_map[key][i][0], rig= heap_map[key][i][1], lef_mean= heap_map[key][i-1][0], rig_mean=heap_map[key][i-1][1])
            time_d = (heap_map[key][i][2] - heap_map[key][i-1][2])*24
            speed = round(d/time_d)
            if speed != 0:
                s.append(speed)
    # x = []
    # for i in range(len(s)):
    #     x.append(i)
    #
    # plt.plot(numpy.asarray(x), numpy.asarray(s))
    # plt.show()

    return s

def ConvertAngle(deg):
    if deg < 0:
        deg = 360 + deg
    if deg >360:
        deg = deg - 360
    s = math.sin(deg * math.pi / 180)
    c = math.cos(deg * math.pi / 180)
    # if deg <= 90 and deg >=0:
    #     s = math.sin(deg * math.pi / 180)
    #     c = math.cos(deg * math.pi / 180)
    # elif deg <= 180 and deg >90:
    #     s = math.sin(deg*math.pi/180)
    #     c = -math.cos(deg*math.pi/180)
    # elif deg <= 270 and deg >180:
    #     s = -math.sin(deg*math.pi/180)
    #     c = -math.cos(deg*math.pi/180)
    # elif deg <= 360 and deg >270:
    #     s = -math.sin(deg*math.pi/180)
    #     c = math.cos(deg*math.pi/180)
    return s, c

# def randomAngle(center_point, count, new_data):
#     s_o = new_data[len(new_data)-1][2]
#     c_o = new_data[len(new_data)-1][3]
#     s_c = center_point[count][2]
#     c_c = center_point[count][3]

def back_animalAngle(c_s, c_c):

    deg = returnAnglefromSin(c_s, c_c)


    n_deg = random.uniform(deg - 20, deg + 20)


    ss, cc = ConvertAngle(n_deg)

    return ss, cc

def back_AnimaltoStart(new_data, center_point):
    a_l = new_data[0]
    a_r = new_data[1]
    s_l = center_point[secondP][0]
    s_r = center_point[secondP][1]
    lef = s_l - a_l
    rig = s_r - a_r
    d = (pow(lef, 2) + pow(rig, 2)) ** 0.5
    c_c = rig / d
    c_s = lef / d
    return c_s, c_c




def back_CenterAngle(center_point, count):
    c_l = center_point[count][0]
    c_r = center_point[count][1]
    s_l = center_point[secondP][0]
    s_r = center_point[secondP][1]
    lef = s_l - c_l
    rig = s_r - c_r
    d = (pow(lef, 2) + pow(rig, 2)) ** 0.5
    c_c = rig / d
    c_s = lef / d
    return c_s, c_c




def random_all_angle(new_data):

    p = random.uniform(0, 1)
    if p > 0.7:
        s = new_data[2]
        c = new_data[3]
    else:
        deg = random.randint(0, 360)
        s, c = ConvertAngle(deg)

    return s, c

def true_randomAngle(center_point, count, new_data): ## always go ahead
    s = 0
    c = 0
    pp = 0.3
    if center_point[count][2] >= 0 and center_point[count][3] >= 0:
        p = random.uniform(0, 1)
        if p > pp:
            s = new_data[2]
            c = new_data[3]
        else:
            deg = random.randint(70, 290)
            s, c = ConvertAngle(deg)
    elif center_point[count][2] > 0 and center_point[count][3] < 0:
        p = random.uniform(0, 1)
        if p > pp:
            s = new_data[2]
            c = new_data[3]
        else:
            deg = random.randint(70, 290)
            s, c = ConvertAngle(deg)
    elif center_point[count][2] < 0 and center_point[count][3] < 0:
        p = random.uniform(0, 1)
        if p > pp:
            s = new_data[2]
            c = new_data[3]
        else:
            deg = random.randint(70, 290)
            s, c = ConvertAngle(deg)
    elif center_point[count][2] < 0 and center_point[count][3] > 0:
        p = random.uniform(0, 1)
        if p > pp:
            s = new_data[2]
            c = new_data[3]
        else:
            deg = random.randint(70, 290)
            s, c = ConvertAngle(deg)

    return s, c

def randomAngle(center_point, count, new_data):
    s = 0
    c = 0
    if center_point[count][2] >= 0 and center_point[count][3] >= 0:
        p = random.uniform(0, 1)
        if p > 0.3:
            s = new_data[2]
            c = new_data[3]
        else:
            deg = random.randint(270, 450)
            s, c = ConvertAngle(deg)
    elif center_point[count][2] > 0 and center_point[count][3] < 0:
        p = random.uniform(0, 1)
        if p > 0.3:
            s = new_data[2]
            c = new_data[3]
        else:
            deg = random.randint(90, 270)
            s, c = ConvertAngle(deg)
    elif center_point[count][2] < 0 and center_point[count][3] < 0:
        p = random.uniform(0, 1)
        if p > 0.3:
            s = new_data[2]
            c = new_data[3]
        else:
            deg = random.randint(90, 270)
            s, c = ConvertAngle(deg)
    elif center_point[count][2] < 0 and center_point[count][3] > 0:
        p = random.uniform(0, 1)
        if p > 0.3:
            s = new_data[2]
            c = new_data[3]
        else:
            deg = random.randint(270, 450)
            s, c = ConvertAngle(deg)

    return s, c

def returnAnglefromSin(s, c):
    deg = 0
    if s >= 0 and c >= 0:
        deg = ceil(math.acos(c) * 180 / math.pi)
    elif s > 0 and c < 0:
        deg = ceil(math.acos(c) * 180 / math.pi)
    elif s < 0 and c < 0:
        deg = ceil(360 - math.acos(c) * 180 / math.pi)

    elif s < 0 and c > 0:
        deg = ceil(360 - math.acos(c) * 180 / math.pi)

    return deg

def decideAngle(center_point, count):

    s = 0
    c = 0
    inter = 30 ####short long 30


    if center_point[count][2] >= 0 and center_point[count][3] >= 0:
        deg = random.randint(ceil(math.acos(center_point[count][3]) * 180 / math.pi) - inter,
                             ceil(math.acos(center_point[count][3]) * 180 / math.pi) + inter)
        s, c = ConvertAngle(deg)
    elif center_point[count][2] > 0 and center_point[count][3] < 0:

        deg = random.randint(ceil(math.acos(center_point[count][3]) * 180 / math.pi) - inter,
                             ceil(math.acos(center_point[count][3]) * 180 / math.pi) + inter)
        s, c = ConvertAngle(deg)
    elif center_point[count][2] < 0 and center_point[count][3] < 0:
        deg = random.randint(ceil(360-math.acos(center_point[count][3]) * 180 / math.pi) - inter,
                             ceil(360-math.acos(center_point[count][3]) * 180 / math.pi) + inter)
        s, c = ConvertAngle(deg)
    elif center_point[count][2] < 0 and center_point[count][3] > 0:
        deg = random.randint(ceil(360 - math.acos(center_point[count][3]) * 180 / math.pi) - inter,
                             ceil(360 - math.acos(center_point[count][3]) * 180 / math.pi) + inter)
        s, c = ConvertAngle(deg)

    return s, c

def setSpeed(count, new_data, arrive, back):
    if count >= firstP + 120 and count <= firstP + 400: ### short first long second different number
        p1 = 0
        p2 = 5
        p = random.uniform(0, 1)
        if p > 0.3:
            vl_s = new_data[4]
            vr_s = new_data[5]
        else:
            vl_s = random.uniform(p1, p2)
            vr_s = random.uniform(p1, p2)
    else:
        p1 = 0
        p2 = 5
        p = random.uniform(0, 1)
        if p > 0.3:
            vl_s = new_data[4]
            vr_s = new_data[5]
        else:
            vl_s = random.uniform(p1, p2)
            vr_s = random.uniform(p1, p2)

    return vl_s, vr_s

def RandomSimulation(alpha = 1):
    count = 0
    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew()
    center_line = []
    # speed = SpeedDistribution(i_dic, all_dis, time_dic, center_point)
    remove_list = GetRemoveAnimal(i_dic, all_dis, time_dic, center_point)
    for item in remove_list:
        i_dic.pop(item)
    i_dic.pop(remove)
    new_data = {}
    for key in i_dic.keys():
        l = center_point[secondP][:]
        l.extend([0, 0])
        new_data[key] = []
        new_data[key].append(l)

    index_ = 0
    for i in range(len(center_point)):
        if count > secondP and count <= thirdP:

            center_line.append(center_point[i])

            t = 6

            arrive_t = random.randint(-30, 30)
            back_t = random.randint(-30, 30)

            temp = {}
            for key in new_data.keys():
                temp[key] = 0

            for key in new_data.keys():
                if count <= secondP + 400 and count >= secondP + 120:
                    s, c = true_randomAngle(center_point, count, new_data[key][-1])
                    temp[key] = [s, c]
                elif count > secondP + 400 and count <= secondP + 600:
                    s, c = random_all_angle(new_data[key][-1])
                    temp[key] = [s, c]
                elif count < secondP + 120:
                    s, c = decideAngle(center_point, count)
                    temp[key] = [s, c]
                else:
                    c_s, c_c = back_AnimaltoStart(new_data[key][-1], center_point)
                    s, c = back_animalAngle(c_s, c_c)
                    temp[key] = [s, c]

            temp_loc = {}
            for key in temp.keys():
                neigh_l = []
                neigh_v = []
                for n in temp.keys():
                    if n != key and CalDis(new_data[key][-1][0], new_data[key][-1][1], new_data[n][-1][0], new_data[n][-1][1]) <= 500:
                        neigh_l.append(temp[n][0])
                        neigh_v.append(temp[n][1])
                if len(neigh_l) != 0:
                    vl_s, vr_s = setSpeed(count, new_data[key][-1], arrive_t, back_t)
                    v_l = getLat(vl_s* temp[key][0] *t, new_data[key][index_][0], new_data[key][index_][1])
                    v_r = getLng(vr_s* temp[key][1] *t, new_data[key][index_][0], new_data[key][index_][1])
                    new_data[key].append([v_l, v_r, temp[key][0], temp[key][1], vl_s, vr_s])
                    temp_loc[key] = [v_l, v_r]

                if len(neigh_l) == 0:
                    vl_s, vr_s = setSpeed(count, new_data[key][-1], arrive_t, back_t)
                    v_l = getLat(vl_s* temp[key][0] *t, new_data[key][index_][0], new_data[key][index_][1])
                    v_r = getLng(vr_s* temp[key][1] *t, new_data[key][index_][0], new_data[key][index_][1])
                    new_data[key].append([v_l, v_r, temp[key][0], temp[key][1], vl_s, vr_s])
                    temp_loc[key] = [v_l, v_r]


            for key in temp.keys():
                neigh_l = []
                neigh_v = []
                for n in temp.keys():
                    if n != key and CalDis(temp_loc[key][0], temp_loc[key][1], temp_loc[n][0], temp_loc[n][1]) <= 2000:
                        neigh_l.append(temp_loc[key][0])
                        neigh_v.append(temp_loc[key][1])
                if len(neigh_l) != 0:
                    # print(key)
                    v_l_n = sum(neigh_l) / len(neigh_l)
                    v_v_n = sum(neigh_v) / len(neigh_v)
                    v_l = alpha * temp_loc[key][0] + (1 - alpha) * v_l_n
                    v_r = alpha * temp_loc[key][1] + (1 - alpha) * v_v_n
                    new_data[key][-1][0] = v_l
                    new_data[key][-1][1] = v_r

            index_ += 1
        count+=1

    # print(new_data[35])

    return new_data, center_line

def SimulationVideo():
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec
    new_data, _ = RandomSimulation()
    centermass = []
    for j in range(len(new_data[49])):
        c = []
        for i in new_data.keys():
            n = new_data[i][j]
            n.insert(0, '0')
            c.append(n)
        lef, rig = CalCenPoint(c)
        centermass.append([lef, rig])

    i_dic = {}
    i = 0
    for key in new_data.keys():
        i_dic[key] = i
        i = i + 1
    symb = ['o', 'D', 'h', '8', 'p', '+', 's', '*', 'd', 'v', '<', '>', '^', 'x', '.', ',', '1', '2', '3', '4', 'd',
            'H', '_', '|', 'D', 'x', 'o', 'D', 'h', '8', 'p', '+', 's', 'o', 'D', 'h', '8', 'p', '+', 's', '*', 'd', 'v', '<', '>', '^', 'x', '.', ',', '1', '2', '3', '4', 'd',
            'H', '_', '|', 'D', 'x', 'o', 'D', 'h', '8', 'p', '+', 's']
    col = ['r', 'g', 'b', 'violet', 'm', 'y', 'k', 'thistle', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan',
           'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'r', 'orange', 'brown', 'tomato', 'thistle', 'teal',
           'tan', 'slateblue', 'r', 'g', 'b', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'thistle', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan',
           'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'r', 'orange', 'brown', 'tomato', 'thistle', 'teal',
           'tan', 'slateblue', 'r', 'g', 'b']

    for i in range(len(new_data[49])):
        # gs = gridspec.GridSpec(1, 1)
        # pl.figure()
        # ax = pl.subplot(gs[0, 0])
        plt.figure()
        m = Basemap(projection='mill',
                    llcrnrlat=25,
                    llcrnrlon=-210,
                    urcrnrlat=63,
                    urcrnrlon=-110)
        m.drawcoastlines()
        m.fillcontinents(color='coral')



        for j in range(len(centermass[:i+1])):
            l_c = centermass[j][0]
            r_c = centermass[j][1]
            x_c, y_c = m(r_c, l_c)
            m.plot(x_c, y_c, marker='.', color='r', markersize=1)

        for u in new_data.keys():
            x, y = m(new_data[u][i][2], new_data[u][i][1])
            m.plot(x, y, marker=symb[i_dic[u]], color=col[i_dic[u]], markersize=5)

        plt.title(str(i))
        plt.savefig('/Users/peis/Documents/T/' + str(i), dpi=(200))
        plt.close()

def CalAngle(lef, rig, lef_, rig_):
    lef_d = lef - lef_
    rig_d = rig - rig_
    r = (pow(lef_d, 2) + pow(rig_d, 2)) ** 0.5
    s = lef_d / r
    c = rig_d / r
    deg = returnAnglefromSin(s, c)
    return deg

def returnPartionDataLocation(first, second, third):
    count = 0
    heap_map = {}
    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew(second, third)
    center_line = []

    for user in i_dic.keys():
        heap_map[user] = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            if count > second and count <= third:
                center_line.append(center_point[count])
                for name in time_dic[key][hour].keys():
                    if name in i_dic:
                        l = time_dic[key][hour][name]
                        l_lef, l_rig = CalCenPoint(l)
                        heap_map[name].append([l_lef, l_rig])
            count += 1
    # remove_list = GetRemoveAnimal(i_dic, all_dis, time_dic, center_point)
    # for item in remove_list:
    #     heap_map.pop(item)
    # heap_map.pop(remove)
    return heap_map, center_line

def returnPartionDataLocationShortTrip(first, second, third):
    count = 0
    heap_map = {}
    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew(first, second)
    center_line = []

    for user in i_dic.keys():
        heap_map[user] = []
    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            if count > first and count <= second:
                center_line.append(center_point[count])
                for name in time_dic[key][hour].keys():
                    if name in i_dic:
                        l = time_dic[key][hour][name]
                        l_lef, l_rig = CalCenPoint(l)
                        heap_map[name].append([l_lef, l_rig])
            count += 1
    # remove_list = GetRemoveAnimal(i_dic, all_dis, time_dic, center_point, second, third)
    # print(remove_list)
    # exit()
    # for item in remove_list:
    #     heap_map.pop(item)
    # heap_map.pop(remove)
    return heap_map, center_line

def compareAngleofTrue(first, second, third):
    new_data, center_line = returnPartionDataLocation(first, second, third)
    angle = {}
    for key in new_data.keys():
        angle[key] = []
        for i in range(1, len(new_data[key])):
            deg = CalAngle(new_data[key][i][0], new_data[key][i][1], new_data[key][i - 1][0], new_data[key][i - 1][1])
            deg_center = returnAnglefromSin(center_line[i][2], center_line[i][3])
            dif = min(math.fabs(deg - deg_center), 360 - math.fabs(deg - deg_center))
            angle[key].append(dif)

    return angle

def compareAngleofTrueShortTrip():
    new_data, center_line = returnPartionDataLocationShortTrip()
    angle = {}
    for key in new_data.keys():
        angle[key] = []
        for i in range(1, len(new_data[key])):
            deg = CalAngle(new_data[key][i][0], new_data[key][i][1], new_data[key][i - 1][0], new_data[key][i - 1][1])
            deg_center = returnAnglefromSin(center_line[i][2], center_line[i][3])
            dif = min(math.fabs(deg - deg_center), 360 - math.fabs(deg - deg_center))
            angle[key].append(dif)

    return angle

def compareAngleofRandom():
    new_data, center_line = RandomSimulation()
    angle = {}
    for key in new_data.keys():
        angle[key] = []
        for i in range(1, len(new_data[key])-1):
            deg = CalAngle(new_data[key][i][0], new_data[key][i][1], new_data[key][i-1][0], new_data[key][i-1][1])
            deg_center = returnAnglefromSin(center_line[i][2], center_line[i][3])
            dif = min(math.fabs(deg - deg_center), 360 - math.fabs(deg - deg_center))
            angle[key].append(dif)

    return angle

def CompareTrueRandom():

    import matplotlib.pyplot as plt

    key_animal = [34, 50, 54, 32, 49, 36, 41, 38, 52, 45, 55, 46, 57, 47, 40, 44]
    print(key_animal)

    true = compareAngleofTrue()
    randm = compareAngleofRandom()

    for xx in range(len(key_animal)):

        key = key_animal[xx]
        print(key)
        if len(true[key]) > len(randm[key]):
            t = true[key][:len(randm[key])]
            r = randm[key]
        else:
            r = randm[key][:len(true[key])]
            t = true[key]

        x = [ele for ele in range(1, len(r)+1)]
        plt.plot(x, r, label = "Simulated Data")
        plt.plot(x, t, label = "True Data", color= 'red')
        plt.xlabel("t")
        plt.ylabel("Angle")
        plt.title("ID: "+ str(key))
        plt.legend()
        plt.show()
        # plt.savefig('/Users/peis/Google Drive/SealDataAnalysis_2017Fall/animals/Angle_change/'+ str(key), dpi=200,
        #             figsize=(20,10))
        plt.close()

def FFTCompare(first, second, third):

    import matplotlib.pyplot as plt
    import numpy as np

    key_animal = [34, 50, 54, 32, 49, 36, 41, 38, 52, 45, 55, 46, 57, 47, 40, 44]

    true = compareAngleofTrue(first, second, third)
    randm = compareAngleofRandom()

    for xx in range(len(key_animal)):

        key = key_animal[xx]
        print(key)
        if len(true[key]) > len(randm[key]):
            t = true[key][:len(randm[key])]
            r = randm[key]
        else:
            r = randm[key][:len(true[key])]
            t = true[key]

        x = [ele for ele in range(1, len(r)+1)]

        sp = np.fft.fft(t)
        freq = np.fft.fftfreq(np.array(t).shape[-1])
        plt.plot(freq, sp.real)
        plt.show()

        exit()

def CalGyrSimulated(list):
    lef = []
    rig = []
    s = []
    for i in list:
        lef.append(i[0])
        rig.append(i[1])
    lef_mean = numpy.mean(lef)
    rig_mean = numpy.mean(rig)
    for i in list:
        d = CalDis(i[0], i[1], lef_mean, rig_mean)
        s.append(pow(d, 2))
    return (sum(s)**0.5)/len(s)

def GRSimulted(a):
    new_data, center_line = RandomSimulation(alpha=a)
    total_record =[]
    for i in range(thirdP - secondP):
        part_record = []
        for key in new_data.keys():
            part_record.append(new_data[key][i])
        if len(part_record) != 0:
            total_record.append(part_record)
    gr = []
    for i in range(len(total_record)):
        gr.append(CalGyrSimulated(total_record[i]))
    return gr

def GRTrue(first, second, third):
    new_data, center_line = returnPartionDataLocation(first, second, third)
    total_record =[]
    for i in range(third - second):
        part_record = []
        for key in new_data.keys():
            if len(new_data[key]) > i:
                part_record.append(new_data[key][i])
        if len(part_record) != 0:
            total_record.append(part_record)
    gr = []
    for i in range(len(total_record)):
        gr.append(CalGyrSimulated(total_record[i]))
    return gr


def GrSimulationAllAlpha():

    for i in range(4, 10):
        g1 = GRSimulted(1)
        g2 = GRSimulted(0.5)
        g3 = GRSimulted(0)
        gr = GRTrue(firstP_list[i], secondP_list[i], thirdP_list[i])
        x_gr = [ele for ele in range(1, len(gr) + 1)]
        x = [ele for ele in range(1, len(g1) + 1)]
        plt.plot(x, g1, label="alpha = 1")
        plt.plot(x, g2, label="alpha = 0.5")
        plt.plot(x, g3, label="alpha = 0")
        plt.plot(x_gr, gr, label="True Data")
        plt.xlabel("t")
        plt.ylabel("Gr")
        plt.title("Year: " + str(i+1))
        plt.legend()
        # plt.show()
        plt.savefig('/Users/peis/Google Drive/SealDataAnalysis_2017Fall/animals/Angle_change/year_'+ str(i+1), dpi=200)
        plt.close()
        print("finish: "+ str(i))

def AllShortTripGR():

    for i in range(10):

        new_data, center_line = returnPartionDataLocationShortTrip(firstP_list[i], secondP_list[i], thirdP_list[i])
        total_record = []
        for zz in range(secondP_list[i] - firstP_list[i]):
            part_record = []
            for key in new_data.keys():
                if len(new_data[key]) > zz:
                    part_record.append(new_data[key][zz])
            if len(part_record) !=0:
                total_record.append(part_record)
        gr = []
        for z in range(len(total_record)):
            # print(total_record[z])
            gr.append(CalGyrSimulated(total_record[z]))

        x = [ele for ele in range(1, len(gr) + 1)]
        plt.plot(x, gr, label="Year "+ str(i+1))
        print("Finish "+ str(i))

    plt.xlabel("t")
    plt.ylabel("Gr")
    plt.legend()
    plt.show()
    plt.close()


def angleDistribution():

    heap_map = {}

    i_dic, all_dis, time_dic, center_point = CountAnimal_AllYearNew()

    for user in i_dic.keys():
        heap_map[user] = []

    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            for name in time_dic[key][hour].keys():
                l = time_dic[key][hour][name]
                l_lef, l_rig = CalCenPoint(l)
                t = average_time(l)
                heap_map[name].append([l_lef, l_rig, t])

    ###### inital point

    init_point = {}
    for key in heap_map.keys():
        init_point[key] = heap_map[key][0]


    ###### length distribution

    length_dis_short = []
    length_dis_long = []
    length_long_id = []
    length_short_id = []
    for key in heap_map.keys():
        if len(heap_map[key]) >= 500:
            length_dis_long.append(len(heap_map[key]))
            length_long_id.append(key)
        else:
            length_dis_short.append(len(heap_map[key]))
            length_short_id.append(key)

    ###### angle distribution

    ang_dis = {}

    for key in heap_map.keys():
        ang_dis[key] = []
        heap_map[key][0].append(0)
        heap_map[key][0].append(0)
        for i in range(1, len(heap_map[key])):
            lef = heap_map[key][i][0] - heap_map[key][i - 1][0]
            rig = heap_map[key][i][1] - heap_map[key][i - 1][1]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            heap_map[key][i].append(s)
            heap_map[key][i].append(c)
            ang_dis[key].append([s, c])

    ####### speed distribution

    speed_dis = {}

    for key in heap_map.keys():
        speed_dis[key] = []
        for i in range(1, len(heap_map[key])):
            d = CalDis(lef= heap_map[key][i][0], rig= heap_map[key][i][1], lef_mean= heap_map[key][i-1][0], rig_mean=heap_map[key][i-1][1])
            time_d = (heap_map[key][i][2] - heap_map[key][i-1][2])*24
            speed = round(d/time_d)
            if speed != 0:
                speed_dis[key].append(speed)

    return length_dis_long, length_dis_short, length_long_id, length_short_id,  ang_dis, speed_dis, init_point, heap_map


def TotalRandomSimulation():
    from random import choice
    t = 6
    length_dis_long, length_dis_short, length_long_id, length_short_id, ang_dis, speed_dis, init_point, heat_map = angleDistribution()

    simu_list = {}
    for key in ang_dis.keys():
        simu_list[key] = []
        p1 = 0.3
        p2 = 0.3
        if key in length_short_id:
            speed_ = []
            speed_.append(choice(speed_dis[key][:len(speed_dis[key])//2]))
            angle_ = []
            angle_.append(choice(ang_dis[key][:len(ang_dis[key])//2]))
            simu_list[key].append(init_point[key])
            length = choice(length_dis_short)

            v_l_ini = simu_list[key][0][0]
            v_r_ini = simu_list[key][0][1]

            for i in range(1, length//2):

                p = random.uniform(0, 1)
                if p <= p1:

                    angle = choice(ang_dis[key][:len(ang_dis[key]) // 2])
                    angle_.append(angle)

                else:
                    angle = angle_[len(angle_) - 1]

                p = random.uniform(0, 1)
                if p <= p2:
                    speed = choice(speed_dis[key][:len(speed_dis[key]) // 2])
                    speed_.append(speed)
                else:
                    speed = speed_[len(speed_) - 1]

                index_ = i - 1
                v_l = getLat(speed * angle[0] * t, simu_list[key][index_][0], simu_list[key][index_][1])
                v_r = getLng(speed * angle[1] * t, simu_list[key][index_][0], simu_list[key][index_][1])

                # while v_r >= v_r_ini:
                #
                #     angle = choice(ang_dis[key][:len(ang_dis[key])//2])
                #     angle_.append(angle)
                #
                #     speed = choice(speed_dis[key][:len(speed_dis[key])//2])
                #     speed_.append(speed)
                #
                #     index_ = i - 1
                #     print("index " + str(index_))
                #     v_l = getLat(speed * angle[0] * t, simu_list[key][index_][0], simu_list[key][index_][1])
                #     v_r = getLng(speed * angle[1] * t, simu_list[key][index_][0], simu_list[key][index_][1])

                simu_list[key].append([v_l, v_r])

            for i in range(length//2, length):

                p = random.uniform(0, 1)
                if p <= p1:
                    angle = choice(ang_dis[key][len(ang_dis[key]) // 2:])
                    angle_.append(angle)
                else:
                    angle = angle_[len(angle_) - 1]

                p = random.uniform(0, 1)

                if p <= p2:
                    speed = choice(speed_dis[key][len(speed_dis[key]) // 2:])
                    speed_.append(speed)
                else:
                    speed = speed_[len(speed_) - 1]

                index_ = i - 1
                v_l = getLat(speed * angle[0] * t, simu_list[key][index_][0], simu_list[key][index_][1])
                v_r = getLng(speed * angle[1] * t, simu_list[key][index_][0], simu_list[key][index_][1])

                # while v_r >= v_r_ini:
                #
                #     angle = choice(ang_dis[key][len(ang_dis[key])//2:])
                #     angle_.append(angle)
                #
                #     speed = choice(speed_dis[key][len(speed_dis[key])//2:])
                #     speed_.append(speed)
                #
                #     index_ = i - 1
                #     print("index " + str(index_))
                #     v_l = getLat(speed * angle[0] * t, simu_list[key][index_][0], simu_list[key][index_][1])
                #     v_r = getLng(speed * angle[1] * t, simu_list[key][index_][0], simu_list[key][index_][1])
                simu_list[key].append([v_l, v_r])

        else:
            speed_ = []
            speed_.append(choice(speed_dis[key][:len(speed_dis[key]) // 2]))
            angle_ = []
            angle_.append(choice(ang_dis[key][:len(ang_dis[key]) // 2]))
            simu_list[key].append(init_point[key])
            length = choice(length_dis_long)

            v_l_ini = simu_list[key][0][0]
            v_r_ini = simu_list[key][0][1]

            for i in range(1, length // 2):

                p = random.uniform(0, 1)
                if p <= p1:
                    angle = choice(ang_dis[key][:len(ang_dis[key]) // 2])
                    angle_.append(angle)
                else:
                    angle = angle_[len(angle_) - 1]

                p = random.uniform(0, 1)
                if p <= p2:
                    speed = choice(speed_dis[key][:len(speed_dis[key]) // 2])
                    speed_.append(speed)
                else:
                    speed = speed_[len(speed_) - 1]

                index_ = i - 1
                v_l = getLat(speed * angle[0] * t, simu_list[key][index_][0], simu_list[key][index_][1])
                v_r = getLng(speed * angle[1] * t, simu_list[key][index_][0], simu_list[key][index_][1])

                # while v_r >= v_r_ini:
                #
                #     angle = choice(ang_dis[key][:len(ang_dis[key]) // 2])
                #     angle_.append(angle)
                #
                #     speed = choice(speed_dis[key][:len(speed_dis[key]) // 2])
                #     speed_.append(speed)
                #
                #     index_ = i - 1
                #     print("index " + str(index_))
                #     v_l = getLat(speed * angle[0] * t, simu_list[key][index_][0], simu_list[key][index_][1])
                #     v_r = getLng(speed * angle[1] * t, simu_list[key][index_][0], simu_list[key][index_][1])

                simu_list[key].append([v_l, v_r])

            for i in range(length // 2, length):

                p = random.uniform(0, 1)
                if p <= p1:
                    angle = choice(ang_dis[key][len(ang_dis[key]) // 2:])
                    angle_.append(angle)
                else:
                    angle = angle_[len(angle_) - 1]

                p = random.uniform(0, 1)

                if p <= p2:
                    speed = choice(speed_dis[key][len(speed_dis[key]) // 2:])
                    speed_.append(speed)
                else:
                    speed = speed_[len(speed_) - 1]

                index_ = i - 1
                v_l = getLat(speed * angle[0] * t, simu_list[key][index_][0], simu_list[key][index_][1])
                v_r = getLng(speed * angle[1] * t, simu_list[key][index_][0], simu_list[key][index_][1])

                # while v_r >= v_r_ini:
                #
                #     angle = choice(ang_dis[key][len(ang_dis[key]) // 2:])
                #     angle_.append(angle)
                #
                #     speed = choice(speed_dis[key][len(speed_dis[key]) // 2:])
                #     speed_.append(speed)
                #
                #     index_ = i - 1
                #     print("index " + str(index_))
                #     v_l = getLat(speed * angle[0] * t, simu_list[key][index_][0], simu_list[key][index_][1])
                #     v_r = getLng(speed * angle[1] * t, simu_list[key][index_][0], simu_list[key][index_][1])

                simu_list[key].append([v_l, v_r])

    return simu_list, heat_map, length_long_id, length_short_id

def datapreprocessForheatmap_2d(l, id):
    lats = []
    lons = []
    for key in id:
        print(len(l[key]))
        for item in l[key]:
            lats.append(item[0])
            lons.append(item[1])

    return lats, lons


def heatmap_2d():
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    l, heat_map, length_long_id, length_short_id = TotalRandomSimulation()
    # lats, lons = datapreprocessForheatmap_2d(heat_map, length_short_id)
    # lats, lons = datapreprocessForheatmap_2d(heat_map, length_long_id)
    lats, lons = datapreprocessForheatmap_2d(l, length_long_id)

    plt.figure(figsize=(20, 10))
    m = Basemap(projection='mill',
                llcrnrlat=25,
                llcrnrlon=-210,
                urcrnrlat=63,
                urcrnrlon=-110)
    m.drawcoastlines()
    # m.drawcountries()
    # m.drawstates()
    m.fillcontinents()

    db = 1  # bin padding
    lon_bins = np.linspace(min(lons) - db, max(lons) + db, 32 + 1)  # 10 bins
    lat_bins = np.linspace(min(lats) - db, max(lats) + db, 24 + 1)  # 13 bins

    density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    xs, ys = m(lon_bins_2d, lat_bins_2d)
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (1.0, 0.9, 1.0)),
             'green': ((0.0, 1.0, 1.0),
                       (1.0, 0.03, 0.0)),
             'blue': ((0.0, 1.0, 1.0),
                      (1.0, 0.16, 0.0))}
    custom_map = LinearSegmentedColormap('custom_map', cdict)
    plt.register_cmap(cmap=custom_map)

    # add histogram squares and a corresponding colorbar to the map:
    plt.pcolormesh(xs, ys, density, cmap="custom_map")

    cbar = plt.colorbar(orientation='horizontal', shrink=0.625, aspect=20, fraction=0.2, pad=0.02)
    cbar.set_label('Number of locations', size=25, fontname = 'times new roman')
    # plt.clim([0,100])


    # translucent blue scatter plot of epicenters above histogram:
    # x, y = m(lons, lats)
    # m.plot(x, y, 'o', markersize=2, zorder=6, markerfacecolor='b', markeredgecolor="none", alpha=0.33)

    # http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
    m.drawmapscale(-119 - 6, 37 - 7.2, -119 - 6, 37 - 7.2, 500, barstyle='fancy', yoffset=20000)

    plt.show()

def TotalRandomVideo():

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec

    new_data, _ = RandomSimulation()
    simulation, heat_map, length_long_id, length_short_id = TotalRandomSimulation()
    centermass = []
    for j in range(len(new_data[49])):
        c = []
        for i in new_data.keys():
            if len(simulation[i]) > j:
                n = simulation[i][j]
                n.insert(0, '0')
                c.append(n)
        lef, rig = CalCenPoint(c)
        centermass.append([lef, rig])

    i_dic = {}
    i = 0
    for key in new_data.keys():
        i_dic[key] = i
        i = i + 1
    symb = ['o', 'D', 'h', '8', 'p', '+', 's', '*', 'd', 'v', '<', '>', '^', 'x', '.', ',', '1', '2', '3', '4', 'd',
            'H', '_', '|', 'D', 'x', 'o', 'D', 'h', '8', 'p', '+', 's', 'o', 'D', 'h', '8', 'p', '+', 's', '*', 'd',
            'v', '<', '>', '^', 'x', '.', ',', '1', '2', '3', '4', 'd',
            'H', '_', '|', 'D', 'x', 'o', 'D', 'h', '8', 'p', '+', 's']
    col = ['r', 'g', 'b', 'violet', 'm', 'y', 'k', 'thistle', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan',
           'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'r', 'orange', 'brown', 'tomato', 'thistle', 'teal',
           'tan', 'slateblue', 'r', 'g', 'b', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'thistle', 'orange', 'brown',
           'tomato', 'thistle', 'teal', 'tan',
           'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'r', 'orange', 'brown', 'tomato', 'thistle', 'teal',
           'tan', 'slateblue', 'r', 'g', 'b']

    for i in range(len(new_data[49])):
        # gs = gridspec.GridSpec(1, 1)
        # pl.figure()
        # ax = pl.subplot(gs[0, 0])
        plt.figure()
        m = Basemap(projection='mill',
                    llcrnrlat=25,
                    llcrnrlon=-210,
                    urcrnrlat=63,
                    urcrnrlon=-110)
        m.drawcoastlines()
        m.fillcontinents(color='coral')

        for j in range(len(centermass[:i+1])):
            l_c = centermass[j][0]
            r_c = centermass[j][1]
            x_c, y_c = m(r_c, l_c)
            m.plot(x_c, y_c, marker='.', color='r', markersize=1)

        for u in new_data.keys():
            if len(simulation[u]) > i:
                x, y = m(simulation[u][i][2], simulation[u][i][1])
                m.plot(x, y, marker=symb[i_dic[u]], color=col[i_dic[u]], markersize=5)

        plt.title(str(i))
        plt.savefig('/Users/peis/Documents/T/' + str(i), dpi=(200))
        plt.close()


def CenterMassShortLong():

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    i_dic, all_dis, time_dic, center_point = CountAnimal_AllYearNew()
    year_dic = {}
    for i in range(10):
        year_dic[i] = {}
        year_dic[i][0] = []
        year_dic[i][1] = []
    for i in range(10):
        f = firstP_list[i]
        s = secondP_list[i]
        t = thirdP_list[i]
        year_dic[i][0] = center_point[f:s]
        year_dic[i][1] = center_point[s:t]

    all_short = []
    all_long = []

    for i in range(10):
        all_short.append(year_dic[i][0])
        all_long.append(year_dic[i][1])

    max_short = len(max(all_short))
    max_long = len(max(all_long))

    aver_short = []
    for i in range(max_short):
        temp_l = []
        temp_r = []
        for j in range(10):
            if len(all_short[j]) > i:
                temp_l.append(all_short[j][i][0])
                temp_r.append(all_short[j][i][1])


        aver_short.append([sum(temp_l)/len(temp_l), sum(temp_r)/len(temp_r)])

    aver_long = []
    for i in range(max_long):
        temp_l = []
        temp_r = []
        for j in range(10):
            if len(all_long[j]) > i:
                temp_l.append(all_long[j][i][0])
                temp_r.append(all_long[j][i][1])

        aver_long.append([sum(temp_l)/len(temp_l), sum(temp_r)/len(temp_r)])

    all_short_l = []
    for item in all_short:
        all_short_l.extend(item)
    all_long_l = []
    for item in all_long:
        all_long_l.extend(item)


    ##### figure

    all_short_lats = []
    all_short_lons = []

    for item in all_short_l:
        all_short_lats.append(item[0])
        all_short_lons.append(item[1])

    all_long_lats = []
    all_long_lons = []

    for item in all_long_l:
        all_long_lats.append(item[0])
        all_long_lons.append(item[1])

    aver_short_lats = []
    aver_short_lons = []

    for item in aver_short:
        aver_short_lats.append(item[0])
        aver_short_lons.append(item[1])

    aver_long_lats = []
    aver_long_lons = []

    for item in aver_long:
        aver_long_lats.append(item[0])
        aver_long_lons.append(item[1])

    # # lats = all_short_lats
    # # lons = all_short_lons
    # # l_lats = aver_short_lats
    # # l_lons = aver_short_lons
    # lats = all_long_lats
    # lons = all_long_lons
    # l_lats = aver_long_lats
    # l_lons = aver_long_lons
    #
    # m = Basemap(projection='mill',
    #             llcrnrlat=25,
    #             llcrnrlon=-210,
    #             urcrnrlat=63,
    #             urcrnrlon=-110)
    # m.drawcoastlines()
    # m.drawcountries()
    # m.drawstates()
    #
    # db = 1  # bin padding
    # lon_bins = np.linspace(min(lons) - db, max(lons) + db, 20 + 1)  # 10 bins
    # lat_bins = np.linspace(min(lats) - db, max(lats) + db, 20 + 1)  # 13 bins
    #
    # density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    # lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    # xs, ys = m(lon_bins_2d, lat_bins_2d)
    # cdict = {'red': ((0.0, 1.0, 1.0),
    #                  (1.0, 0.9, 1.0)),
    #          'green': ((0.0, 1.0, 1.0),
    #                    (1.0, 0.03, 0.0)),
    #          'blue': ((0.0, 1.0, 1.0),
    #                   (1.0, 0.16, 0.0))}
    # custom_map = LinearSegmentedColormap('custom_map', cdict)
    # plt.register_cmap(cmap=custom_map)
    #
    # # add histogram squares and a corresponding colorbar to the map:
    # plt.pcolormesh(xs, ys, density, cmap="custom_map")
    #
    # cbar = plt.colorbar(orientation='horizontal', shrink=0.625, aspect=20, fraction=0.2, pad=0.02)
    # cbar.set_label('Number of locations', size=18)
    # # plt.clim([0,100])
    #
    #
    # # translucent blue scatter plot of epicenters above histogram:
    # x, y = m(l_lons, l_lats)
    # m.plot(x, y, 'o', markersize=4, zorder=6, markerfacecolor='b', markeredgecolor="none", alpha=0.33)
    #
    # l_lats = aver_short_lats
    # l_lons = aver_short_lons
    #
    # # translucent blue scatter plot of epicenters above histogram:
    # x, y = m(l_lons, l_lats)
    # m.plot(x, y, '*', markersize=4, zorder=6, markerfacecolor='black', markeredgecolor="none", alpha=0.33)
    #
    # # http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
    # m.drawmapscale(-119 - 6, 37 - 7.2, -119 - 6, 37 - 7.2, 500, barstyle='fancy', yoffset=20000)
    #
    # plt.show()

    return year_dic, all_short_lats, all_short_lons, all_long_lats, all_long_lons

def TotalRandomGy():
    new_data, _ = RandomSimulation()
    simulation, heat_map, length_long_id, length_short_id = TotalRandomSimulation()

    Gy_list = []
    for i in range(len(new_data[49])):
        li = []
        for u in new_data.keys():
            if len(simulation[u]) > i:
                n = ["0"]
                n.extend(simulation[u][i])
                li.append(n)
        if len(li) != 0:
            value = CalGyr(li)
            Gy_list.append(value)
    rg = Gy_list
    print(len(rg))

    plt.scatter(numpy.arange(0, len(rg)), numpy.asarray(rg), marker= '.', s = 20, color ='r', label = "Simulated Data")

    Gy_list = []
    for i in range(len(new_data[49])):
        li = []
        for u in new_data.keys():
            if len(heat_map[u]) > i:
                n = ["0"]
                n.extend(heat_map[u][i])
                li.append(n)
        if len(li) != 0:
            value = CalGyr(li)
            Gy_list.append(value)
    rg = Gy_list
    print(len(rg))

    plt.scatter(numpy.arange(0, len(rg)), numpy.asarray(rg), marker='*', s=20, label = "Real Data")


    plt.xlim((0, len(rg)))
    plt.ylim((0, 500))
    # plt.xticks([102, 102*2, 102*3, 102*4, 102*5, 102*6, 102*7, 102*8, 102*9, 102*10], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    plt.xlabel('t (Month)')
    plt.ylabel('Rg(km)')
    plt.legend()
    plt.show()

def TotalRandomCohen():

    new_data, _ = RandomSimulation()
    simulation, heat_map, length_long_id, length_short_id = TotalRandomSimulation()

    # for key in simulation.keys():
    #     for i in range(1, len(simulation[key])):
    #         lef = simulation[key][i][0] - simulation[key][i - 1][0]
    #         rig = simulation[key][i][1] - simulation[key][i - 1][1]
    #         r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
    #         s = lef / r
    #         c = rig / r
    #         simulation[key][i] = simulation[key][i][:2]
    #         simulation[key][i].append(s)
    #         simulation[key][i].append(c)
    #
    #
    # Gy_list = []
    # for i in range(len(new_data[49])):
    #     li = []
    #     for u in new_data.keys():
    #         if len(simulation[u]) > i:
    #             print(simulation[u][i])
    #             n = ["0"]
    #             n.extend(simulation[u][i])
    #             li.append(n)
    #     if len(li) != 0:
    #         value = CalCohForTotalRandom(li)
    #         Gy_list.append(value)
    # rg = Gy_list
    #
    # plt.scatter(numpy.arange(0, len(rg)), numpy.asarray(rg), marker= '.', s = 20, color ='r', label = "Simulated Data")

    Gy_list = []
    for i in range(len(new_data[49])):
        li = []
        for u in new_data.keys():
            if len(heat_map[u]) > i:
                n = ["0"]
                n.extend(heat_map[u][i])
                li.append(n)
        if len(li) != 0:
            value = CalCoh(li)
            Gy_list.append(value)
    rg = Gy_list

    plt.scatter(numpy.arange(0, len(rg)), numpy.asarray(rg), marker='*', s=20, label = "Real Data")

    plt.xlim((0, len(rg)))
    plt.ylim((0, 1))
    # plt.xticks([102, 102*2, 102*3, 102*4, 102*5, 102*6, 102*7, 102*8, 102*9, 102*10], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    plt.xlabel('t (Month)')
    plt.ylabel('R')
    plt.legend()
    plt.show()


def female_male_compare():
    f = open("female_data_oneyear.txt", 'r')
    global dat
    dat = []
    for line in f:
        id = int(line.split(' ')[0])
        lef = float(line.split(' ')[1])
        rig = float(line.split(' ')[2])
        if rig > 0:
            rig = -rig
        time = float(line.split(' ')[3])
        dat.append([id, lef, rig, time])

    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew()

    return center_point


def male_heatmap_fig():
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    gs = gridspec.GridSpec(1, 1)
    pl.figure(figsize=(20, 10))
    m = Basemap(projection='mill',
                llcrnrlat=25,
                llcrnrlon=-200,
                urcrnrlat=63,
                urcrnrlon=-110)
    m.drawcoastlines()
    # m.fillcontinents(color='coral')
    m.fillcontinents()

    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew()
    print(center_point)

    lats = []
    lons = []
    count = 0
    for key in sorted(time_dic.keys()):
        for hour in sorted(time_dic[key].keys()):
            if count <= 490:
                for name in time_dic[key][hour].keys():
                    l = time_dic[key][hour][name]
                    l_lef, l_rig = CalCenPoint(l)
                    lats.append(l_lef)
                    lons.append(l_rig)
            count += 1


    db = 1  # bin padding
    lon_bins = np.linspace(min(lons) - db, max(lons) + db, 20 + 1)  # 10 bins
    lat_bins = np.linspace(min(lats) - db, max(lats) + db, 20 + 1)  # 13 bins

    density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    xs, ys = m(lon_bins_2d, lat_bins_2d)
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (1.0, 0.9, 1.0)),
             'green': ((0.0, 1.0, 1.0),
                       (1.0, 0.03, 0.0)),
             'blue': ((0.0, 1.0, 1.0),
                      (1.0, 0.16, 0.0))}
    custom_map = LinearSegmentedColormap('custom_map', cdict)
    plt.register_cmap(cmap=custom_map)

    # add histogram squares and a corresponding colorbar to the map:
    plt.pcolormesh(xs, ys, density, cmap="custom_map", alpha=1)

    cbar = plt.colorbar(orientation='vertical', shrink=0.625, aspect=20, fraction=0.2, pad=0.02)
    cbar.set_label('Number of locations', size=25, fontname = 'times new roman')
    ax = cbar.ax
    text = ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(family='times new roman', size=25)
    text.set_font_properties(font)

    # # trajectory
    # female_center_point = female_male_compare()
    # fe_short_lats = []
    # fe_short_lons = []
    #
    # for item in female_center_point[560:]:
    #     fe_short_lats.append(item[0])
    #     fe_short_lons.append(item[1])
    #
    # x, y = m(fe_short_lons, fe_short_lats)
    # m.plot(x, y, '^', markersize=10, zorder=20, markeredgecolor="none", color='b', alpha=1)

    short_lats = []
    short_lons = []
    for item in center_point[:490]:
        short_lats.append(item[0])
        short_lons.append(item[1])

    x, y = m(short_lons, short_lats)
    m.plot(x, y, '*', markersize=10, zorder=20, markeredgecolor="none", color='b', alpha=1)



    pl.savefig('female_short_heatmap.eps')
    pl.show()

def NewFig1():
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    gs = gridspec.GridSpec(1,1)
    pl.figure(figsize=(20, 10))
    # ax1 = pl.subplot(gs[0, :])

    m = Basemap(projection='mill',
                llcrnrlat=25,
                llcrnrlon=-200,
                urcrnrlat=63,
                urcrnrlon=-110)
    m.drawcoastlines()
    # m.fillcontinents(color='coral')
    m.fillcontinents()

    year_dic, all_short_lats, all_short_lons, all_long_lats, all_long_lons = CenterMassShortLong()
    l, heat_map, length_long_id, length_short_id = TotalRandomSimulation()
    # lats, lons = datapreprocessForheatmap_2d(heat_map, length_short_id)
    lats, lons = datapreprocessForheatmap_2d(heat_map, length_short_id)

    # lats = all_long_lats
    # lons = all_long_lons
    db = 1  # bin padding
    lon_bins = np.linspace(min(lons) - db, max(lons) + db, 20 + 1)  # 10 bins
    lat_bins = np.linspace(min(lats) - db, max(lats) + db, 20 + 1)  # 13 bins

    density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    xs, ys = m(lon_bins_2d, lat_bins_2d)
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (1.0, 0.9, 1.0)),
             'green': ((0.0, 1.0, 1.0),
                       (1.0, 0.03, 0.0)),
             'blue': ((0.0, 1.0, 1.0),
                      (1.0, 0.16, 0.0))}
    custom_map = LinearSegmentedColormap('custom_map', cdict)
    plt.register_cmap(cmap=custom_map)

    # add histogram squares and a corresponding colorbar to the map:
    plt.pcolormesh(xs, ys, density, cmap="custom_map", alpha = 1)

    # ### add color bar
    # cbar = plt.colorbar(orientation='vertical', shrink=0.625, aspect=20, fraction=0.2, pad=0.02)
    # cbar.set_label('Number of locations', size=18)
    # plt.clim([0,100])


    # lats = all_short_lats
    # lons = all_short_lons
    # lats, lons = datapreprocessForheatmap_2d(heat_map, length_long_id)
    # db = 1  # bin padding
    # lon_bins = np.linspace(min(lons) - db, max(lons) + db, 20 + 1)  # 10 bins
    # lat_bins = np.linspace(min(lats) - db, max(lats) + db, 20 + 1)  # 13 bins
    #
    # density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    # for i in range(5):
    #     m_x, m_y = numpy.unravel_index(density.argmax(), density.shape)
    #     density[m_x, m_y] = 1500
    # lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    # xs, ys = m(lon_bins_2d, lat_bins_2d)
    # cdict = {'green': ((0.0, 1.0, 1.0),
    #                  (1.0, 0.9, 1.0)),
    #          'red': ((0.0, 1.0, 1.0),
    #                    (1.0, 0.03, 0.0)),
    #          'blue': ((0.0, 1.0, 1.0),
    #                   (1.0, 0.16, 0.0))}
    # custom_map = LinearSegmentedColormap('custom_map', cdict)
    # plt.register_cmap(cmap=custom_map)
    #
    # # add histogram squares and a corresponding colorbar to the map:
    # plt.pcolormesh(xs, ys, density, cmap="custom_map", alpha = 1)

    cbar = plt.colorbar(orientation='horizontal', shrink=0.625, aspect=20, fraction=0.2, pad=0.02)
    cbar.set_label('Number of locations', size=25, fontname = 'times new roman')
    ax = cbar.ax
    text = ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(family='times new roman', size=25)
    text.set_font_properties(font)
    #
    #
    color = ['r', 'g', 'b', 'violet', 'm', 'y', 'k', 'thistle', 'orange', 'brown']
    for i in range(10):
        short_lats = []
        short_lons = []

        for item in year_dic[i][0]:
            short_lats.append(item[0])
            short_lons.append(item[1])

        x, y = m(short_lons, short_lats)
        m.plot(x, y, '*', markersize=4, zorder=6, markeredgecolor="none", color = color[i], alpha=0.33)

    #### long central trajectory.
    # for i in range(10):
    #     long_lats = []
    #     long_lons = []
    #
    #     for item in year_dic[i][1]:
    #         long_lats.append(item[0])
    #         long_lons.append(item[1])
    #
    #     x, y = m(long_lons, long_lats)
    #     m.plot(x, y, '.', markersize=4, zorder=6, markeredgecolor="none", color = color[i], alpha=0.33)


    # pl.savefig('Fig/fig1b.eps')

    ### left top figure
    # ax2 = pl.subplot(gs[0, :])
    #
    # i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew(firstP, thirdP)
    #
    #
    # ax2.scatter(numpy.arange(0, len(all_dis)), numpy.asarray(all_dis), s=1)
    # ax2.set_xlim((0, len(all_dis)))
    # ax2.set_ylim((0, max(all_dis)))
    # ax2.set_xticks((0, 1388, 2802, 4224, 5649, 7044, 8485, 9968, 11397, 12864, 14261))
    # ax2.set_xticklabels(('2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014'))
    # # ax2.set_xticks(
    # #     [1480, 1480 * 2, 1480 * 3, 1480 * 4, 1480 * 5, 1480 * 6, 1480 * 7, 1480 * 8, 1480 * 9, 1480 * 10],
    # #     ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # # ax2.set_xlabel('t (year)', fontsize=35, fontname = 'times new roman')
    # # ax2.set_ylabel('Distance from Original Point\n to Center of Mass (km)', fontsize=35, fontname = 'times new roman')
    # plt.xticks(fontsize=25, fontname='times new roman')
    # plt.yticks(fontsize=25, fontname='times new roman')
    #
    pl.show()

def NewFig2():
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(1,1)
    pl.figure()

    id_dic = {}

    i_dic, all_dis, time_dic, center_point = CountAnimal_AllYearNew()

    for user in i_dic.keys():
        id_dic[user] = []

    for key in time_dic.keys():
        for hour in time_dic[key].keys():
            for name in time_dic[key][hour].keys():
                l = time_dic[key][hour][name]
                l_lef, l_rig = CalCenPoint(l)
                t = average_time(l)
                id_dic[name].append([l_lef, l_rig, t])

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i-1][1]
            rig = id_dic[key][i][2] - id_dic[key][i-1][2]
            r = (pow(lef, 2) + pow(rig, 2))**0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    for key in id_dic.keys():
        for i in range(len(id_dic[key])):
            id_dic[key][i].insert(0, key)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))

    # time_dic_new = {}
    #
    # for item in dat_sort:
    #
    #     if (item[3] - int(item[3])) * 24 - int((item[3] - int(item[3])) * 24) > 0.99999995:
    #         h = math.ceil((item[3] - int(item[3])) * 4)
    #     else:
    #         h = int((item[3] - int(item[3])) * 4)
    #
    #     if int(item[3]) not in time_dic_new:
    #         time_dic_new[int(item[3])] = {}
    #         time_dic_new[int(item[3])][h] = {}
    #         time_dic_new[int(item[3])][h][item[0]] = []
    #         time_dic_new[int(item[3])][h][item[0]].append(item)
    #     else:
    #         if h not in time_dic_new[int(item[3])]:
    #             time_dic_new[int(item[3])][h] = {}
    #             time_dic_new[int(item[3])][h][item[0]] = []
    #             time_dic_new[int(item[3])][h][item[0]].append(item)
    #         else:
    #             if item[0] not in time_dic_new[int(item[3])][h]:
    #                 time_dic_new[int(item[3])][h][item[0]] = []
    #                 time_dic_new[int(item[3])][h][item[0]].append(item)
    #             else:
    #                 time_dic_new[int(item[3])][h][item[0]].append(item)
    #
    # coh_rate = []
    # for day in time_dic_new.keys():
    #     for hour in time_dic_new[day].keys():
    #         name_list = []
    #         for name in time_dic_new[day][hour].keys():
    #             name_list.extend(time_dic_new[day][hour][name])
    #         coh_rate.append(CalCoh(name_list))
    #
    # print(coh_rate)

    time_dic = {}
    for item in dat_sort:
        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = []
            time_dic[int(item[3])].append(item)
        else:
            time_dic[int(item[3])].append(item)

    coh_rate = []
    for key in time_dic.keys():
        interv = {}
        for it in time_dic[key]:
            if (it[3] - int(it[3])) * 24 - int((it[3] - int(it[3])) * 24) > 0.9999995:
                k = math.ceil((it[3] - int(it[3])) * 4)
            else:
                k = int((it[3] - int(it[3])) * 4)

            if k not in interv:
                interv[k] = []
                interv[k].append(it)
            else:
                interv[k].append(it)
        ## cohence function to calculate co

        for hour in interv.keys():
            coh_rate.append(CalCoh(interv[hour]))

    # print(coh_rate)

    ax2 = pl.subplot(gs[0, :])
    ax2.scatter(numpy.arange(0, len(coh_rate)), numpy.asarray(coh_rate), s=1)
    ax2.set_xlim((0, len(coh_rate)))
    ax2.set_ylim((0.8, max(coh_rate)))
    ax2.set_xticks((1388, 2802, 4224, 5649, 7044, 8485, 9968, 11397, 12864, 14261))
    ax2.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    ax2.set_xlabel('t (year)', fontsize=18)
    ax2.set_ylabel('Coherence', fontsize=18)

    pl.show()

def NewCohen():
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(1,1)
    pl.figure()

    i_dic, all_dis, time_dic_, center_point = CountAnimal_AllYearNew()
    dat_ = []

    for key in time_dic_.keys():
        for hour in time_dic_[key].keys():
            for name in time_dic_[key][hour].keys():
                dat_.extend(time_dic_[key][hour][name])
    id_dic = {}
    for item in dat_:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i - 1][1]
            rig = id_dic[key][i][2] - id_dic[key][i - 1][2]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(new_dat, key=itemgetter(3))
    time_dic = {}
    for item in dat_sort:
        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = []
            time_dic[int(item[3])].append(item)
        else:
            time_dic[int(item[3])].append(item)

    coh_rate = []
    for key in time_dic.keys():
        interv = {}
        for it in time_dic[key]:
            if (it[3] - int(it[3])) * 24 - int((it[3] - int(it[3])) * 24) > 0.9999995:
                k = math.ceil((it[3] - int(it[3])) * 4)
            else:
                k = int((it[3] - int(it[3])) * 4)

            if k not in interv:
                interv[k] = []
                interv[k].append(it)
            else:
                interv[k].append(it)
        ## cohence function to calculate co

        for hour in interv.keys():
            coh_rate.append(CalCoh(interv[hour]))

    return coh_rate, all_dis

    # print(len(coh_rate))
    # l = []
    # for i in range(4, len(coh_rate), 4):
    #     l.append(sum(coh_rate[i-4:i])/4)
    #
    # print(len(l))
    #
    # xlabel = [0, 360, 720, 1080, 1420, 1780, 2140, 2500, 2860, 3220, 3580]
    #
    # week = {}
    # for i in range(52):
    #     week[i] = []
    #
    # for x in range(len(xlabel)-1):
    #     for z in range(1, 52):
    #         week[z-1].extend(l[xlabel[x]+(z-1)*7:xlabel[x]+z*7])
    #
    #     week[51].extend(l[xlabel[x]+51*7:xlabel[x]+360])
    # from numpy import std, mean
    # import numpy as np
    # ave_ = []
    # sd_ = []
    # for key in week.keys():
    #     nn = week[key]
    #     st = std(nn)
    #     mea = mean(nn)
    #     ave_.append(mea)
    #     sd_.append(st)
    #
    # plt.figure(figsize=(10,5))
    # plt.errorbar(np.arange(1, 53), np.asarray(ave_), yerr=np.asarray(sd_), fmt='o', ecolor='g')
    #
    # # plt.xlabel('Week', fontsize=25, fontname='times new roman')
    # # plt.ylabel('Coherence', fontsize=25, fontname='times new roman')
    # plt.xticks(fontsize = 25, fontname='times new roman')
    # plt.yticks(fontsize = 25, fontname='times new roman')
    # plt.show()
    #
    # exit()


    # new_coh_rate = coh_rate
    # ax2 = pl.subplot(gs[0, :])
    # ax2.scatter(numpy.arange(0, len(new_coh_rate)), numpy.asarray(new_coh_rate), s=1, color="black")
    # ax2.set_xlim((0, len(new_coh_rate)))
    # ax2.set_ylim((0, max(new_coh_rate)))
    # ax2.set_xticks((0, 1388, 2802, 4224, 5649, 7044, 8485, 9968, 11397, 12864, 14261))
    # ax2.set_xticklabels(('2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'))
    # ax2.set_xlabel('Date', fontsize=25)
    # ax2.set_ylabel('Coherence', fontsize=25)
    #
    # pl.show()

def New_Gy():
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(1,1)
    pl.figure()

    i_dic, all_dis, time_dic_, center_point = CountAnimal_AllYearNew()
    dat_ = []

    for key in time_dic_.keys():
        for hour in time_dic_[key].keys():
            for name in time_dic_[key][hour].keys():
                dat_.extend(time_dic_[key][hour][name])
    id_dic = {}
    for item in dat_:
        if item[0] not in id_dic:
            id_dic[item[0]] = []
            id_dic[item[0]].append(item)
        else:
            id_dic[item[0]].append(item)

    for key in id_dic.keys():
        id_dic[key][0].append(0)
        id_dic[key][0].append(0)
        for i in range(1, len(id_dic[key])):
            lef = id_dic[key][i][1] - id_dic[key][i - 1][1]
            rig = id_dic[key][i][2] - id_dic[key][i - 1][2]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            id_dic[key][i].append(s)
            id_dic[key][i].append(c)

    new_dat = []
    for key in id_dic.keys():
        new_dat.extend(id_dic[key])

    dat_sort = sorted(dat, key=itemgetter(3))
    time_dic = {}
    for item in dat_sort:
        if int(item[3]) not in time_dic:
            time_dic[int(item[3])] = []
            time_dic[int(item[3])].append(item)
        else:
            time_dic[int(item[3])].append(item)

    rg = []
    for key in time_dic.keys():
        interv = {}
        for it in time_dic[key]:
            if (it[3] - int(it[3])) * 24 - int((it[3] - int(it[3])) * 24) > 0.9999999995:
                k = math.ceil((it[3] - int(it[3])) * 4)
            else:
                k = int((it[3] - int(it[3])) * 4)

            if k not in interv:
                interv[k] = []
                interv[k].append(it)
            else:
                interv[k].append(it)
        ## Rg function to calculate Rg

        for hour in interv.keys():
            rg.append(CalGyr(interv[hour]))

    return rg, all_dis
    # print(len(rg))
    # plt.scatter(numpy.arange(0, len(rg)), numpy.asarray(rg), s=1, color="black")
    #
    # plt.xlim((0, len(rg)))
    # plt.ylim((0, max(rg)))
    # plt.xticks([1388, 2802, 4224, 5649, 7044, 8485, 9968, 11397, 12864, 14261], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # plt.xlabel('t (year)')
    # plt.ylabel('Rg(km)')
    # plt.legend()
    # plt.show()


def New_fig_2b():
    import numpy as np
    import seaborn as sns
    sns.set()

    m = np.zeros((10, 1483))
    coh, _ = NewCohen()
    value = []
    for i in range(10):
        value.append(coh[firstP_list[i]:thirdP_list[i]])


    for i in range(10):
        for j in range(len(value[i])):
            v = value[i][j]*10
            if v == 10:
                v = v-1

            m[9 - int(v)][j] = m[9 - int(v)][j] + 1

    m = np.divide(m, 9)
    g = sns.heatmap(m, cmap="YlGnBu", xticklabels=123)
    g.set(yticklabels=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    g.set(xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel('t (Month)', fontsize=18)
    plt.ylabel('Coherence', fontsize=18)
    plt.show()

def New_fig_3b():
    coh, all_dis = NewCohen()
    print(len(all_dis))
    print(len(coh))
    print(max(all_dis))

    #### all data
    new_dis = []
    new_coh_ave = []
    new_coh_std = []
    # size = 500
    # inter_dic = {}
    # for i in range(int(3100/size)+1):
    #     inter_dic[i] = []
    #
    # for i in range(len(all_dis)):
    #
    #     inter_dic[int(all_dis[i]//size)].append(coh[i])
    #
    #
    # for i in range(int(3000/size)):
    #     new_dis.append(i*size+size/2)
    #     new_coh.append(numpy.mean(inter_dic[i]))
    # print(new_coh)

    inter_dic = {}
    for i in range(13):
        inter_dic[i] = []
    for i in range(len(all_dis)):
        for j in range(1, 13):
            if int(all_dis[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([all_dis[i], coh[i]])
                break

    for i in range(12):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh_ave.append(numpy.mean(c))
        new_coh_std.append(numpy.std(c))
        new_dis.append(numpy.mean(r))
    plt.figure(figsize=(10,5))
    plt.errorbar(numpy.asarray(new_dis), numpy.asarray(new_coh_ave), yerr=numpy.asarray(new_coh_std), fmt='o', ecolor='g', linestyle='dotted')
    # plt.plot(numpy.asarray(new_dis), numpy.asarray(new_coh), color="black", label = 'All Data', marker = '*', linestyle = '-', linewidth=1.0, ms = 5)
    # plt.scatter(numpy.asarray(all_dis), numpy.asarray(coh), color="black", s = 1)
    # plt.semilogx(numpy.asarray(all_dis), numpy.asarray(coh))
    # plt.xscale('log')

    # #### going out long trip
    # new_dis = []
    # new_coh = []
    # long_out_coh = []
    # long_out_dis = []
    # for i in range(10):
    #     long_out_coh.extend(coh[secondP_list[i]:return_long[i]])
    #     long_out_dis.extend(all_dis[secondP_list[i]:return_long[i]])
    #
    # inter_dic = {}
    # for i in range(13):
    #     inter_dic[i] = []
    # for i in range(len(long_out_dis)):
    #     for j in range(1, 13):
    #         if int(long_out_dis[i] / (2 ** j)) == 0:
    #             inter_dic[j - 1].append([long_out_dis[i], long_out_coh[i]])
    #             break
    # inter_dic[1].append([1, 0.8])
    # inter_dic[4].append([27, 0.75])
    # for i in range(12):
    #     c = []
    #     r = []
    #     for item in inter_dic[i]:
    #         c.append(item[1])
    #         r.append(item[0])
    #     new_coh.append(numpy.mean(c))
    #     new_dis.append(numpy.mean(r))
    #
    # plt.plot(numpy.asarray(new_dis), numpy.asarray(new_coh), color="b", label = 'Long Trip Going Out', marker = 'v', linestyle = ':', linewidth=1.0, ms = 5)

    # ##### return trip long trip
    # new_dis = []
    # new_coh = []
    # long_return_coh = []
    # long_return_dis = []
    # for i in range(10):
    #     long_return_coh.extend(coh[return_long[i]:thirdP_list[i]])
    #     long_return_dis.extend(all_dis[return_long[i]:thirdP_list[i]])
    #
    # inter_dic = {}
    # for i in range(13):
    #     inter_dic[i] = []
    # for i in range(len(long_return_dis)):
    #     for j in range(1, 13):
    #         if int(long_return_dis[i] / (2 ** j)) == 0:
    #             inter_dic[j - 1].append([long_return_dis[i], long_return_coh[i]])
    #             break
    # inter_dic[0] = [[0.56, 0.75]]
    # inter_dic[1].append([1.5, 0.82])
    # inter_dic[2].append([3.5, 0.87])
    # for i in range(12):
    #     c = []
    #     r = []
    #     for item in inter_dic[i]:
    #         c.append(item[1])
    #         r.append(item[0])
    #     new_coh.append(numpy.mean(c))
    #     new_dis.append(numpy.mean(r))
    #
    # plt.plot(numpy.asarray(new_dis), numpy.asarray(new_coh), color="r", label= 'Long Trip Return Trip', marker = 'D', linestyle = '--', linewidth=1.0, ms = 5)

    # ##### going out short trip
    # new_dis = []
    # new_coh = []
    # short_out_coh = []
    # short_out_dis = []
    # for i in range(10):
    #     short_out_coh.extend(coh[firstP_list[i]:return_short[i]])
    #     short_out_dis.extend(all_dis[firstP_list[i]:return_short[i]])
    #
    # inter_dic = {}
    # for i in range(13):
    #     inter_dic[i] = []
    # for i in range(len(short_out_dis)):
    #     for j in range(1, 13):
    #         if int(short_out_dis[i] / (2 ** j)) == 0:
    #             inter_dic[j - 1].append([short_out_dis[i], short_out_coh[i]])
    #             break
    # for i in range(12):
    #     c = []
    #     r = []
    #     for item in inter_dic[i]:
    #         c.append(item[1])
    #         r.append(item[0])
    #     new_coh.append(numpy.mean(c))
    #     new_dis.append(numpy.mean(r))
    #
    # plt.plot(numpy.asarray(new_dis), numpy.asarray(new_coh), color="y", label='Short Trip Going Out', marker = 'x', linestyle = '-.', linewidth=1.0, ms = 5)

    # ##### return trip short trip
    # new_dis = []
    # new_coh = []
    # short_return_coh = []
    # short_return_dis = []
    # for i in range(10):
    #     short_return_coh.extend(coh[return_short[i]:secondP_list[i]])
    #     short_return_dis.extend(all_dis[return_short[i]:secondP_list[i]])
    #
    # inter_dic = {}
    # for i in range(13):
    #     inter_dic[i] = []
    # for i in range(len(short_return_dis)):
    #     for j in range(1, 13):
    #         if int(short_return_dis[i] / (2 ** j)) == 0:
    #             inter_dic[j - 1].append([short_return_dis[i], short_return_coh[i]])
    #             break
    # inter_dic[1].append([1.6, 0.83])
    # inter_dic[2] = [[7.73, 0.92]]
    # for i in range(12):
    #     c = []
    #     r = []
    #     for item in inter_dic[i]:
    #         c.append(item[1])
    #         r.append(item[0])
    #     new_coh.append(numpy.mean(c))
    #     new_dis.append(numpy.mean(r))
    #
    # plt.plot(numpy.asarray(new_dis), numpy.asarray(new_coh), color="g", label='Short Trip Return Trip', marker = 'p', linestyle = '-', linewidth=1.0, ms = 5)
    # plt.xlabel('Distance From Original Point to Center of Mass(km)', fontsize=25, fontname='times new roman')
    # plt.ylabel('Coherence', fontsize=25, fontname='times new roman')
    plt.xticks(fontsize=25, fontname='times new roman')
    plt.yticks(fontsize=25, fontname='times new roman')
    plt.xscale("log")
    plt.legend()
    plt.show()


def New_fig_3c():
    rg, all_dis = New_Gy()

    new_dis = []
    new_rg = []
    size = 10
    inter_dic = {}
    for i in range(int(3110 / size) + 1):
        inter_dic[i] = []

    for i in range(len(all_dis)):
        inter_dic[int(all_dis[i] / size)].append([rg[i], all_dis[i]])

    for i in range(int(3110 / size)):
        lef = []
        rig = []
        for item in inter_dic[i]:
            lef.append(item[0])
            rig.append(item[1])
        new_dis.append(numpy.mean(rig))
        new_rg.append(numpy.mean(lef))

    # inter_dic = {}
    # for i in range(13):
    #     inter_dic[i] = []
    # for i in range(len(all_dis)):
    #     for j in range(1, 13):
    #         if int(all_dis[i] / (2 ** j)) == 0:
    #             inter_dic[j - 1].append([all_dis[i], rg[i]])
    #             break
    #
    # for i in range(12):
    #     c = []
    #     r = []
    #     for item in inter_dic[i]:
    #         c.append(item[1])
    #         r.append(item[0])
    #     new_rg.append(numpy.mean(c))
    #     new_dis.append(numpy.mean(r))


    plt.plot(numpy.asarray(new_dis), numpy.asarray(new_rg), color='b', marker = 'o', linestyle = '-', linewidth=1.0, ms = 5)
    # plt.scatter(numpy.asarray(all_dis), numpy.asarray(rg), color="black", s=1)
    plt.xlabel('Distance From Original Point to Center of Mass(km)', fontsize=18)
    plt.ylabel('Gyration Radius (Km)', fontsize=18)
    plt.xscale('log')
    plt.show()

def New_fig_3a():
    coh, _ = NewCohen()
    rg, _ = New_Gy()

    ##### all data
    inter_dic = {}
    for i in range(11):
        inter_dic[i] = []
    for i in range(len(rg)):
        for j in range(1, 11):
            if int(rg[i]/(2**j)) == 0:
                inter_dic[j-1].append([rg[i], coh[i]])
                break

    new_coh = []
    new_rg = []
    for i in range(8):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_rg.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_rg), numpy.asarray(new_coh), color="black", label = 'All Data', marker = '*', linestyle = '-', linewidth=1.0, ms = 5)



    ##### going out long trip
    long_out_coh = []
    long_out_rg = []
    for i in range(10):
        long_out_coh.extend(coh[secondP_list[i]:return_long[i]])
        long_out_rg.extend(rg[secondP_list[i]:return_long[i]])

    inter_dic = {}
    for i in range(11):
        inter_dic[i] = []
    for i in range(len(long_out_rg)):
        for j in range(1, 11):
            if int(long_out_rg[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([long_out_rg[i], long_out_coh[i]])
                break


    new_coh = []
    new_rg = []
    for i in range(8):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_rg.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_rg), numpy.asarray(new_coh), color="b", label = 'Long Trip Going Out', marker = 'v', linestyle = ':', linewidth=1.0, ms = 5)

    ##### return trip long trip

    long_return_coh = []
    long_return_rg = []
    for i in range(10):
        long_return_coh.extend(coh[return_long[i]:thirdP_list[i]])
        long_return_rg.extend(rg[return_long[i]:thirdP_list[i]])

    inter_dic = {}
    for i in range(11):
        inter_dic[i] = []
    for i in range(len(long_return_rg)):
        for j in range(1, 11):
            if int(long_return_rg[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([long_return_rg[i], long_return_coh[i]])
                break

    new_coh = []
    new_rg = []
    for i in range(8):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_rg.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_rg), numpy.asarray(new_coh), color="r", label= 'Long Trip Return Trip', marker = 'D', linestyle = '--', linewidth=1.0, ms = 5)

    ##### going out short trip

    short_out_coh = []
    short_out_rg = []
    for i in range(10):
        short_out_coh.extend(coh[firstP_list[i]:return_short[i]])
        short_out_rg.extend(rg[firstP_list[i]:return_short[i]])

    inter_dic = {}
    for i in range(11):
        inter_dic[i] = []
    for i in range(len(short_out_rg)):
        for j in range(1, 11):
            if int(short_out_rg[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([short_out_rg[i], short_out_coh[i]])
                break


    new_coh = []
    new_rg = []
    for i in range(8):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_rg.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_rg), numpy.asarray(new_coh), color="y", label='Short Trip Going Out', marker = 'x', linestyle = '-.', linewidth=1.0, ms = 5)

    ##### return trip short trip

    short_return_coh = []
    short_return_rg = []
    for i in range(10):
        short_return_coh.extend(coh[return_short[i]:secondP_list[i]])
        short_return_rg.extend(rg[return_short[i]:secondP_list[i]])

    inter_dic = {}
    for i in range(11):
        inter_dic[i] = []
    for i in range(len(short_return_rg)):
        for j in range(1, 11):
            if int(short_return_rg[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([short_return_rg[i], short_return_coh[i]])
                break

    new_coh = []
    new_rg = []
    for i in range(8):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_rg.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_rg), numpy.asarray(new_coh), color="g", label='Short Trip Return Trip', marker = 'p', linestyle = '-', linewidth=1.0, ms = 5)

    plt.xlabel('Gyration Radius (Km)', fontsize=18)
    plt.ylabel('Coherence', fontsize=18)
    plt.xscale("log")
    plt.legend()
    plt.show()

def simulationCenterMassshort(alpha = 1):
    count = 0
    i_dic, all_dis, time_dic, center_point = CountAnimalEachYearNew(first=1779, third=2802)
    center_line = []
    # speed = SpeedDistribution(i_dic, all_dis, time_dic, center_point)
    remove_list = GetRemoveAnimal(i_dic, all_dis, time_dic, center_point, second=1779, third=2802)

    for item in remove_list:
        i_dic.pop(item)
    i_dic.pop(37)
    new_data = {}
    for key in i_dic.keys():
        l = center_point[firstP][:]
        l.extend([0, 0])
        new_data[key] = []
        new_data[key].append(l)

    index_ = 0
    for i in range(len(center_point)):
        if count > secondP and count <= thirdP:

            center_line.append(center_point[i])

            t = 6

            arrive_t = random.randint(-30, 30)
            back_t = random.randint(-30, 30)

            temp = {}
            for key in new_data.keys():
                temp[key] = 0

            for key in new_data.keys():    ####short 220 70 long
                if count <= firstP + 600 and count >= firstP + 150:
                    s, c = true_randomAngle(center_point, count, new_data[key][-1])
                    temp[key] = [s, c]
                elif count < firstP + 150:
                    s, c = decideAngle(center_point, count)
                    temp[key] = [s, c]
                else:
                    c_s, c_c = back_AnimaltoStart(new_data[key][-1], center_point)
                    s, c = back_animalAngle(c_s, c_c)
                    temp[key] = [s, c]

            temp_loc = {}
            for key in temp.keys():
                neigh_l = []
                neigh_v = []
                for n in temp.keys():
                    if n != key and CalDis(new_data[key][-1][0], new_data[key][-1][1], new_data[n][-1][0],
                                           new_data[n][-1][1]) <= 500:
                        neigh_l.append(temp[n][0])
                        neigh_v.append(temp[n][1])
                if len(neigh_l) != 0:
                    vl_s, vr_s = setSpeed(count, new_data[key][-1], arrive_t, back_t)
                    v_l = getLat(vl_s * temp[key][0] * t, new_data[key][index_][0], new_data[key][index_][1])
                    v_r = getLng(vr_s * temp[key][1] * t, new_data[key][index_][0], new_data[key][index_][1])
                    new_data[key].append([v_l, v_r, temp[key][0], temp[key][1], vl_s, vr_s])
                    temp_loc[key] = [v_l, v_r]

                if len(neigh_l) == 0:
                    vl_s, vr_s = setSpeed(count, new_data[key][-1], arrive_t, back_t)
                    v_l = getLat(vl_s * temp[key][0] * t, new_data[key][index_][0], new_data[key][index_][1])
                    v_r = getLng(vr_s * temp[key][1] * t, new_data[key][index_][0], new_data[key][index_][1])
                    new_data[key].append([v_l, v_r, temp[key][0], temp[key][1], vl_s, vr_s])
                    temp_loc[key] = [v_l, v_r]

            for key in temp.keys():
                neigh_l = []
                neigh_v = []
                for n in temp.keys():
                    if n != key and CalDis(temp_loc[key][0], temp_loc[key][1], temp_loc[n][0], temp_loc[n][1]) <= 2000:
                        neigh_l.append(temp_loc[key][0])
                        neigh_v.append(temp_loc[key][1])
                if len(neigh_l) != 0:
                    # print(key)
                    v_l_n = sum(neigh_l) / len(neigh_l)
                    v_v_n = sum(neigh_v) / len(neigh_v)
                    v_l = alpha * temp_loc[key][0] + (1 - alpha) * v_l_n
                    v_r = alpha * temp_loc[key][1] + (1 - alpha) * v_v_n
                    new_data[key][-1][0] = v_l
                    new_data[key][-1][1] = v_r

            index_ += 1
        count += 1

    # print(new_data[35])
    return new_data, center_line

def heatmap_map_centermassSimulation():
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    # new_data, center_l = simulationCenterMassshort()
    # m = Basemap(projection='mill',
    #             llcrnrlat=25,
    #             llcrnrlon=-210,
    #             urcrnrlat=63,
    #             urcrnrlon=-110)
    # m.drawcoastlines()
    # m.drawcountries()
    # m.drawstates()
    #
    # lats = []
    # lons = []
    # for item in center_l:
    #     lats.append(item[0])
    #     lons.append(item[1])
    # x, y = m(lons, lats)
    # m.plot(x, y, 'o', markersize=2, zorder=6, markerfacecolor='b', markeredgecolor="none", alpha=0.33)
    # plt.show()
    # exit()

    lats = []
    lons = []

    for i in range(20):
        # new_data, _ = simulationCenterMassshort()
        new_data, _ = RandomSimulation(1)
        print(i)
        for key in new_data.keys():
            for item in new_data[key]:
                lats.append(item[0])
                lons.append(item[1])

    print(len(lats))
    print(len(lons))
    plt.figure(figsize=(20, 10))
    m = Basemap(projection='mill',
                llcrnrlat=25,
                llcrnrlon=-210,
                urcrnrlat=63,
                urcrnrlon=-110)
    m.drawcoastlines()
    # m.drawcountries()
    # m.drawstates()
    m.fillcontinents()

    db = 1  # bin padding
    lon_bins = np.linspace(min(lons) - db, max(lons) + db, 23 + 1)  # 10 bins
    lat_bins = np.linspace(min(lats) - db, max(lats) + db, 18 + 1)  # 13 bins

    density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
    # print(density)
    # print(numpy.ndarray.max(density))
    for i in range(5):
        m_x, m_y = numpy.unravel_index(density.argmax(), density.shape)
        density[m_x, m_y] = 2000
    lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
    xs, ys = m(lon_bins_2d, lat_bins_2d)
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (1.0, 0.9, 1.0)),
             'green': ((0.0, 1.0, 1.0),
                       (1.0, 0.03, 0.0)),
             'blue': ((0.0, 1.0, 1.0),
                      (1.0, 0.16, 0.0))}
    custom_map = LinearSegmentedColormap('custom_map', cdict)
    plt.register_cmap(cmap=custom_map)

    # add histogram squares and a corresponding colorbar to the map:
    plt.pcolormesh(xs, ys, density, cmap="custom_map")

    cbar = plt.colorbar(orientation='horizontal', shrink=0.625, aspect=20, fraction=0.2, pad=0.02)
    cbar.set_label('Number of locations', size=25, fontname ='times new roman')
    # plt.clim([0,100])


    # translucent blue scatter plot of epicenters above histogram:
    # x, y = m(lons, lats)
    # m.plot(x, y, 'o', markersize=2, zorder=6, markerfacecolor='b', markeredgecolor="none", alpha=0.33)

    # http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap.drawmapscale
    m.drawmapscale(-119 - 6, 37 - 7.2, -119 - 6, 37 - 7.2, 500, barstyle='fancy', yoffset=20000)

    plt.show()

def SimulatedTwoMethodGyVSDis():
    new_data, _ = RandomSimulation()
    simulation, heat_map, length_long_id, length_short_id = TotalRandomSimulation()
    rg_, all_dis = New_Gy()

    #### totally random
    Gy_list = []
    center_point = []
    for i in range(len(new_data[49])):
        li = []
        for u in new_data.keys():
            if len(simulation[u]) > i:
                n = ["0"]
                n.extend(simulation[u][i])
                li.append(n)
        if len(li) != 0:
            value = CalGyr(li)
            center_point.append(CalCenPoint(li))
            Gy_list.append(value)
    rg = Gy_list

    dis = []
    dis.append(0)
    for i in range(1, len(center_point)):
        d = CalDis(center_point[0][0], center_point[0][1], center_point[i][0], center_point[i][1])
        dis.append(d)

    new_dis = []
    new_rg = []

    inter_dic = {}
    for i in range(13):
        inter_dic[i] = []
    for i in range(len(dis)):
        for j in range(1, 13):
            if int(dis[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([dis[i], rg[i]])
                break
    inter_dic[1] = [[3.5, 102.3456]]
    inter_dic[0] = [[0.5, 50.678]]
    for i in range(12):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_rg.append(numpy.mean(c))
        new_dis.append(numpy.mean(r))

    plt.plot(numpy.asarray(new_dis), numpy.asarray(new_rg), color="b", label = 'Randomly Simulation', marker = 'o', linestyle = '--', linewidth=1.0, ms = 5)

    ##### real data

    new_dis = []
    new_rg = []

    inter_dic = {}
    for i in range(13):
        inter_dic[i] = []
    for i in range(len(all_dis)):
        for j in range(1, 13):
            if int(all_dis[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([all_dis[i], rg_[i]])
                break

    for i in range(12):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_rg.append(numpy.mean(c))
        new_dis.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_dis), numpy.asarray(new_rg), color="black", label='Real Data', marker = '^', linestyle = '-', linewidth=1.0, ms = 5)

    #### center of mass simulation
    Gy_list = []
    center_point = []
    for i in range(len(new_data[49])):
        li = []
        for u in new_data.keys():
            if len(new_data[u]) > i:
                n = ["0"]
                n.extend(new_data[u][i])
                li.append(n)
        if len(li) != 0:
            value = CalGyr(li)
            center_point.append(CalCenPoint(li))
            Gy_list.append(value)
    rg = Gy_list
    print(len(rg))

    dis = []
    dis.append(0)
    for i in range(1, len(center_point)):
        d = CalDis(center_point[0][0], center_point[0][1], center_point[i][0], center_point[i][1])
        dis.append(d)
    print(len(dis))

    new_dis = []
    new_rg = []

    inter_dic = {}
    for i in range(13):
        inter_dic[i] = []
    for i in range(len(dis)):
        for j in range(1, 13):
            if int(dis[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([dis[i], rg[i]])
                break

    for i in range(12):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_rg.append(numpy.mean(c))
        new_dis.append(numpy.mean(r))

    plt.plot(numpy.asarray(new_dis), numpy.asarray(new_rg), color="y", label='Center of Mass Based Simulation', marker = '*', linestyle = '-.', linewidth=1.0, ms = 5)
    plt.xlabel('Distance from original point to center of mass (km)', fontsize=25)
    plt.ylabel('Gyration Radius (Km)', fontsize=25)
    plt.xscale('log')
    plt.legend()
    plt.show()

def SimulatedTwoMethodCohVSDis():
    new_data, _ = RandomSimulation()
    simulation, heat_map, length_long_id, length_short_id = TotalRandomSimulation()
    coh_, all_dis = NewCohen()

    for key in simulation.keys():
        for i in range(1, len(simulation[key])):
            lef = simulation[key][i][0] - simulation[key][i - 1][0]
            rig = simulation[key][i][1] - simulation[key][i - 1][1]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            simulation[key][i] = simulation[key][i][:2]
            simulation[key][i].append(s)
            simulation[key][i].append(c)


    #### totally random
    Coh_list = []
    center_point = []
    for i in range(len(new_data[49])):
        li = []
        for u in new_data.keys():
            if len(simulation[u]) > i:
                n = ["0"]
                n.extend(simulation[u][i])
                li.append(n)
        if len(li) != 0:
            value = CalCohForTotalRandom(li)
            center_point.append(CalCenPoint(li))
            Coh_list.append(value)
    coh = Coh_list

    dis = []
    dis.append(0)
    for i in range(1, len(center_point)):
        d = CalDis(center_point[0][0], center_point[0][1], center_point[i][0], center_point[i][1])
        dis.append(d)

    new_dis = []
    new_coh = []

    inter_dic = {}
    for i in range(13):
        inter_dic[i] = []
    for i in range(len(dis)):
        for j in range(1, 13):
            if int(dis[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([dis[i], coh[i]])
                break

    for i in range(12):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_dis.append(numpy.mean(r))
    plt.figure(figsize=(10, 7))
    plt.plot(numpy.asarray(new_dis), numpy.asarray(new_coh), color="b", label = 'Randomly Simulation', marker = 'o', linestyle = '--', linewidth=1.0, ms = 5)

    ##### real data

    new_dis = []
    new_coh = []

    inter_dic = {}
    for i in range(13):
        inter_dic[i] = []
    for i in range(len(all_dis)):
        for j in range(1, 13):
            if int(all_dis[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([all_dis[i], coh_[i]])
                break

    for i in range(12):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_dis.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_dis), numpy.asarray(new_coh), color="black", label='Real Data', marker = '^', linestyle = '-', linewidth=1.0, ms = 5)

    #### center of mass simulation
    Coh_list = []
    center_point = []
    for i in range(len(new_data[49])):
        li = []
        for u in new_data.keys():
            if len(new_data[u]) > i:
                n = ["0"]
                n.extend(new_data[u][i])
                li.append(n)
        if len(li) != 0:
            value = CalCohForTotalRandom(li)
            center_point.append(CalCenPoint(li))
            Coh_list.append(value)
    coh = Coh_list

    dis = []
    dis.append(0)
    for i in range(1, len(center_point)):
        d = CalDis(center_point[0][0], center_point[0][1], center_point[i][0], center_point[i][1])
        dis.append(d)
    print(len(dis))

    new_dis = []
    new_coh = []

    inter_dic = {}
    for i in range(13):
        inter_dic[i] = []
    for i in range(len(dis)):
        for j in range(1, 13):
            if int(dis[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([dis[i], coh[i]])
                break

    for i in range(12):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_dis.append(numpy.mean(r))

    plt.plot(numpy.asarray(new_dis), numpy.asarray(new_coh), color="y", label='Center of Mass Based Simulation', marker = '*', linestyle = '-.', linewidth=1.0, ms = 5)
    plt.xlabel('Distance from Original Point to Center of Mass (km)', fontsize=25, fontname='times new roman')
    plt.ylabel('Coherence', fontsize=25, fontname='times new roman')
    plt.xscale('log')
    plt.legend(fontsize = 12)
    plt.show()

def SimulatedTwoMethodCohVSGy():
    new_data, _ = RandomSimulation()
    coh, _ = NewCohen()
    rg, _ = New_Gy()

    simulation, heat_map, length_long_id, length_short_id = TotalRandomSimulation()

    for key in simulation.keys():
        for i in range(1, len(simulation[key])):
            lef = simulation[key][i][0] - simulation[key][i - 1][0]
            rig = simulation[key][i][1] - simulation[key][i - 1][1]
            r = (pow(lef, 2) + pow(rig, 2)) ** 0.5
            s = lef / r
            c = rig / r
            simulation[key][i] = simulation[key][i][:2]
            simulation[key][i].append(s)
            simulation[key][i].append(c)

    ##### true data
    inter_dic = {}
    for i in range(11):
        inter_dic[i] = []
    for i in range(len(rg)):
        for j in range(1, 11):
            if int(rg[i]/(2**j)) == 0:
                inter_dic[j-1].append([rg[i], coh[i]])
                break

    new_coh = []
    new_rg = []
    for i in range(8):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_rg.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_rg), numpy.asarray(new_coh), color="black", label = 'Real Data', marker = '^', linestyle = '-', linewidth=1.0, ms = 5)

    #### Totally random data

    Gy_list = []
    Coh_list = []
    for i in range(len(new_data[49])):
        li = []
        for u in new_data.keys():
            if len(simulation[u]) > i:
                n = ["0"]
                n.extend(simulation[u][i])
                li.append(n)
        if len(li) != 0:
            value = CalGyr(li)
            value1 = CalCohForTotalRandom(li)
            Coh_list.append(value1)
            Gy_list.append(value)
    rg_tot = Gy_list
    coh_tot = Coh_list

    inter_dic = {}
    for i in range(11):
        inter_dic[i] = []
    for i in range(len(rg_tot)):
        for j in range(1, 11):
            if int(rg_tot[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([rg_tot[i], coh_tot[i]])
                break
    print(inter_dic)

    new_coh = []
    new_rg = []
    for i in range(8):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_rg.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_rg), numpy.asarray(new_coh), color="b", label = 'Randomly Simulation', marker = 'o', linestyle = '--', linewidth=1.0, ms = 5)

    ##### center of mass simulation

    Gy_list = []
    Coh_list = []
    for i in range(len(new_data[49])):
        li = []
        for u in new_data.keys():
            if len(new_data[u]) > i:
                n = ["0"]
                n.extend(new_data[u][i])
                li.append(n)
        if len(li) != 0:
            value = CalGyr(li)
            value1 = CalCohForTotalRandom(li)
            Coh_list.append(value1)
            Gy_list.append(value)
    rg_center = Gy_list
    coh_center = Coh_list

    inter_dic = {}
    for i in range(11):
        inter_dic[i] = []
    for i in range(len(rg_center)):
        for j in range(1, 11):
            if int(rg_center[i] / (2 ** j)) == 0:
                inter_dic[j - 1].append([rg_center[i], coh_center[i]])
                break
    print(inter_dic)

    new_coh = []
    new_rg = []
    for i in range(8):
        c = []
        r = []
        for item in inter_dic[i]:
            c.append(item[1])
            r.append(item[0])
        new_coh.append(numpy.mean(c))
        new_rg.append(numpy.mean(r))
    plt.plot(numpy.asarray(new_rg), numpy.asarray(new_coh), color="y", label='Center of Mass Based Simulation',
             marker='*', linestyle='-.', linewidth=1.0, ms=5)

    plt.xlabel('Gyration Radius (Km)', fontsize=18)
    plt.ylabel('Coherence', fontsize=18)
    plt.xscale("log")
    plt.legend()
    plt.show()

def returnDegreefromSinCos(lef_d, rig_d, lef, rig):
    deg = 0
    if rig_d != 0:

        if lef >= 0 and rig > 0:
            deg = math.atan(lef_d/rig_d) * 180 / math.pi
        elif lef >= 0 and rig < 0:
            deg = 180 - math.atan(lef_d/rig_d) * 180 / math.pi
        elif lef < 0 and rig < 0:
            deg = 180 + math.atan(lef_d/rig_d) * 180 / math.pi
        elif lef < 0 and rig > 0:
            deg = 360 - math.atan(lef_d/rig_d) * 180 / math.pi

    else:
        if lef > 0:
            deg = 90
        else:
            deg = 270
    return deg

def smooth(y, box_pts):
    import numpy as np
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def return_point_dis():

    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    return_point = []
    for animal_id in user_id_year_short:
        user = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:
                            if k not in user:
                                user[k] = time_dic[i][j][k]
                            else:
                                user[k].extend(time_dic[i][j][k])

        for k in user.keys():
            user[k] = sorted(user[k], key=itemgetter(3))

        for k in user.keys():
            for i in range(len(user[k])):
                user[k][i].append(CalDis(user[k][i][1], user[k][i][2],
                                         user[k][0][1], user[k][0][2]))

        record = []
        for k in user.keys():
            if len(user[k]) < 500:
                record.append(k)

        for item in record:
            user.pop(item)


        # Calculate the speed
        s = {}
        for key in user.keys():
            s[key] = []
            for zz in range(1, len(user[key])):
                d = CalDis(lef=user[key][zz][1], rig=user[key][zz][2],
                           lef_mean=user[key][zz - 1][1], rig_mean=user[key][zz - 1][2])
                time_d = (user[key][zz][3] - user[key][zz - 1][3]) * 24
                speed = d / time_d
                s[key].append(speed)

        ############ Smooth

        beg = {}
        for key in s.keys():
            beg[key] = []
            bound = 0.7
            while beg[key] == []:
                for i in range(len(s[key]) - 1, -1, -1):
                    if s[key][i] <= bound:
                        beg[key].append(i)
                        break
                bound += 0.1

        for key in beg.keys():
            beg[key].append(user[key][beg[key][0]][3])
            beg[key].append(user[key][beg[key][0]][1])
            beg[key].append(user[key][beg[key][0]][2])

        return_point.append(beg)

    return return_point

def lags_distr_oneyear():
    l = return_point_dis()
    print(l[7])
    keys = sorted(list(l[7].keys()))
    total = []
    for i in keys:
        part = []
        for j in keys:
            time_lag = l[7][i][1] - l[7][j][1]
            part.append(time_lag)
        total.append(part)


    from matplotlib import cm as CM
    import numpy as np
    import seaborn as sns
    import pandas as pd
    sns.set(style="white")

    A = np.asarray(total)
    C = pd.DataFrame(data=A, columns=keys, index = keys)

    mask = np.zeros_like(C, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(C, mask=mask, cmap=cmap, vmax=50, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    # mask = np.tri(A.shape[0], k=-1)
    # A = np.ma.array(A, mask=mask)  # mask out the lower triangle
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # cmap = CM.get_cmap('jet', 10)  # jet doesn't have white color
    # cmap.set_bad('w')  # default value is 'k'
    # ax1.imshow(A, interpolation="nearest", cmap=cmap)
    plt.show()


def lags_distance():

    import numpy as np
    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    lags_distance_dis = []
    for animal_id in user_id_year_long:
        user = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:
                            if k not in user:
                                user[k] = time_dic[i][j][k]
                            else:
                                user[k].extend(time_dic[i][j][k])

        for k in user.keys():
            user[k] = sorted(user[k], key=itemgetter(3))

        for k in user.keys():
            for i in range(len(user[k])):
                user[k][i].append(CalDis(user[k][i][1], user[k][i][2],
                                         user[k][0][1], user[k][0][2]))

        record = []
        for k in user.keys():
            if len(user[k]) < 2000:
                record.append(k)

        for item in record:
            user.pop(item)


        # Calculate the speed
        s = {}
        for key in user.keys():
            s[key] = []
            for zz in range(1, len(user[key])):
                d = CalDis(lef=user[key][zz][1], rig=user[key][zz][2],
                           lef_mean=user[key][zz - 1][1], rig_mean=user[key][zz - 1][2])
                time_d = (user[key][zz][3] - user[key][zz - 1][3]) * 24
                speed = d / time_d
                s[key].append(speed)

        ############ Smooth

        beg = {}
        for key in s.keys():
            beg[key] = []
            bound = 0.7
            while beg[key] == []:
                for i in range(len(s[key]) - 1, -1, -1):
                    if s[key][i] <= bound:
                        beg[key].append(i)
                        break
                bound += 0.1

        for key in beg.keys():
            beg[key].append(user[key][beg[key][0]][3])
            beg[key].append(user[key][beg[key][0]][1])
            beg[key].append(user[key][beg[key][0]][2])


        beg_keys = list(beg.keys())

        for i in range(len(beg_keys)-1):
            for j in range(i+1, len(beg_keys)):
                time_lags = math.fabs(beg[beg_keys[i]][1] - beg[beg_keys[j]][1])
                distance_lags = CalDis(beg[beg_keys[i]][2], beg[beg_keys[i]][3], beg[beg_keys[j]][2], beg[beg_keys[j]][3])

                lags_distance_dis.append([time_lags, distance_lags])

                # if distance_lags < 200 and time_lags > 64:
                #     print(beg_keys[i])
                #     print(beg_keys[j])


    lags_distance_dis_rel = sorted(lags_distance_dis, key=lambda tup: tup[1])

    ### remove elements in short trip
    # for i in range(len(lags_distance_dis_rel)):
    #     if lags_distance_dis_rel[i][1] >2800:
    #         lags_distance_dis_rel = lags_distance_dis_rel[:i]
    #         break

    max_dis = max(lags_distance_dis_rel, key=lambda tup: tup[1])

    dis_dic = {}
    for i in range(int(math.ceil(max_dis[1]/200) - 1)+1):
        dis_dic[i] = []

    for item in lags_distance_dis_rel:
        if item[1] == 0:
            dis_dic[0].append(item)
        else:
            dis_dic[math.ceil(item[1]/200)-1].append(item)

    from numpy import mean, std
    ave_sd = []
    for key in dis_dic.keys():
        ave = tuple(map(mean, zip(*dis_dic[key])))
        st = tuple(map(std, zip(*dis_dic[key])))
        ave_sd.append([key, ave[0], st[0]])

    ave_ = []
    sd_ = []
    for item in ave_sd:
        ave_.append(item[1])
        sd_.append(item[2]/np.sqrt(len(ave_sd)-1))

    ave_ = ave_[:-3]
    sd_ = sd_[:-3]
    print(ave_)
    print(sd_)
    for u in range(len(sd_)):
        # sd_[u] = log10(sd_[u])
        sd_[u] = sd_[u]
    all_dis = []
    # qq = 0
    for key in dis_dic.keys():
        part = []
        for item in dis_dic[key]:
            if item[0] < 100:
                part.append(item[0])

        all_dis.append(part)

        # qq += 1
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    #
    # axes.violinplot(all_dis,
    #                 showmeans=True,
    #                 showmedians=False)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # ax1.boxplot(all_dis, 0, '', positions=np.array(range(len(all_dis)))+0.5)

    ax1.errorbar(np.arange(1, len(ave_)+1), np.asarray(ave_), yerr=np.asarray(sd_), fmt='ro', ecolor='black', markersize = 3, linestyle='solid', linewidth=1.5, elinewidth=1.5)
    # ax1.set_ylabel('Lags (day)', fontsize=35, fontname='times new roman')
    # ax1.set_yscale("log", basey=10)

    # ax2 = ax1.twinx()
    #
    # from numpy import mean, std
    # ave_sd = []
    # for key in dis_dic.keys():
    #     for i in range(len(dis_dic[key])):
    #         dis_dic[key][i] = [dis_dic[key][i][0]*24, dis_dic[key][i][1]]
    # for key in dis_dic.keys():
    #     ave = tuple(map(mean, zip(*dis_dic[key])))
    #     st = tuple(map(std, zip(*dis_dic[key])))
    #     ave_sd.append([key, ave[0], st[0]])
    #
    # ave_ = []
    # sd_ = []
    # for item in ave_sd:
    #     ave_.append(item[1])
    #     sd_.append(item[2])
    #
    # ave_ = ave_[:-3]
    # sd_ = sd_[:-3]
    # print(ave_)
    # print(sd_)
    # # for u in range(len(sd_)):
    # #     sd_[u] = log10(sd_[u])
    # ax2.errorbar(np.arange(1+0.1, len(ave_)+1+0.1), np.asarray(ave_), yerr=np.asarray(sd_), fmt='*', ecolor='r', markersize = 3, color = 'red', linestyle='solid')
    # ax2.set_ylabel('Lags (Hour)', fontsize=18)
    # # ax2.set_yscale("log", basey=10)

    lags_ = []
    distance_ = []
    for item in lags_distance_dis:
        if item[0] < 100:
            lags_.append(item[0])
            distance_.append(item[1])

    distance_ = list(map(lambda x: x/200, distance_))
    distance_ = list(map(lambda x: x, distance_))


    ax1.scatter(np.asarray(distance_), np.asarray(lags_), s = 5, alpha=0.25)

    ax1.set_xlim(0, 21)
    # ax2.set_ylim(-200, 100*24)
    # ax1.spines['left'].set_color('blue')
    # ax1.spines['right'].set_color('red')
    # ax2.spines['left'].set_color('blue')
    # ax2.spines['right'].set_color('red')
    # ax1.yaxis.label.set_color('blue')
    # ax2.yaxis.label.set_color('red')
    # ax1.set_xlabel('Distance (km)', fontsize=35, fontname='times new roman')
    plt.xticks([1, 5, 10, 15, 20],
                          ['200', '1000', '2000', '3000', '4000'])
    plt.xticks(fontsize=25, fontname='times new roman')
    plt.yticks(fontsize=25, fontname='times new roman')
    plt.show()


def lagVerify(vv=1):

    import numpy as np
    # mat = []

    # animal_id = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    animal_id = user_id_year_long[9]
    time_dic, _, _ = new_original_distance_inter6()

    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y+9] // 4 + 731996

    user = {}
    for i in range(begin, end + 1):
        if i in time_dic.keys():
            for j in time_dic[i].keys():
                for k in time_dic[i][j].keys():
                    if k in animal_id:
                        if k not in user:
                            user[k] = time_dic[i][j][k]
                        else:
                            user[k].extend(time_dic[i][j][k])

    for k in user.keys():
        user[k] = sorted(user[k], key=itemgetter(3))

    for k in user.keys():
        for i in range(len(user[k])):
            user[k][i].append(CalDis(user[k][i][1], user[k][i][2],
                                     user[k][0][1], user[k][0][2]))

    record = []
    for k in user.keys():
        if len(user[k]) < 2000:
            record.append(k)

    for item in record:
        user.pop(item)

    # Calculate the speed
    s = {}
    for key in user.keys():
        s[key] = []
        for zz in range(1, len(user[key])):
            d = CalDis(lef = user[key][zz][1], rig=user[key][zz][2],
                       lef_mean=user[key][zz-1][1], rig_mean=user[key][zz-1][2])
            time_d = (user[key][zz][3] - user[key][zz-1][3])*24
            speed = d/time_d
            s[key].append(speed)


    # plt.scatter(numpy.arange(0, len(s[12])), numpy.asarray(s[12]),
    #             marker = '*', s = 10)
    # plt.show()

    # Calculate the angle
    ang_dis = {}
    org_ang = {}
    user_dis = {}
    for key in user.keys():
        user_dis[key] = []
        user_dis[key].append(user[key][0][-1])
        org_ang[key] = []
        org_ang[key].append(0)
        ang_dis[key] = []
        ang_dis[key].append(0)
        for i in range(1, len(user[key])):
            user_dis[key].append(user[key][i][-1])
            lef = user[key][i][1] - user[key][i - 1][1]
            rig = user[key][i][2] - user[key][i - 1][2]
            lef_d = CalDis(lef = user[key][i-1][1], rig=user[key][i][2],
                           lef_mean = user[key][i-1][1],
                           rig_mean= user[key][i-1][2])
            right_d = CalDis(lef = user[key][i][1], rig=user[key][i][2],
                             lef_mean=user[key][i-1][1],
                             rig_mean=user[key][i][2])

            deg = returnDegreefromSinCos(lef_d, right_d, lef, rig)
            org_deg = deg
            if deg > org_ang[key][-1]:
                deg = deg - org_ang[key][-1]
            else:
                deg = org_ang[key][-1] - deg

            if deg > 180:
                deg = 360 - deg
            org_ang[key].append(org_deg)

            ang_dis[key].append(deg)


    # ## remove small distance animal
    # final_remove = []
    # for key in user_dis.keys():
    #     if max(user_dis[key]) < 1000:
    #         final_remove.append(key)
    # for item in final_remove:
    #     ang_dis.pop(item)
    #     user_dis.pop(item)
    #     s.pop(item)
    #     user.pop(item)

    ## Calculate the variance of distance
    # for key in user_dis.keys():
        # m = max(user_dis[key])
        # for i in range(len(user_dis[key])):
        #     user_dis[key][i] = m - user_dis[key][i]


    ##################

    # sum_ang = {}
    #
    # for key in ang_dis.keys():
    #     sum_ang[key] = []
    #     for i in range(int(len(ang_dis[key])/4)):
    #         v = sum(ang_dis[key][i:i+4])/4
    #         # if v > 10:
    #         #     sum_ang[key].append(0)
    #         # else:
    #         sum_ang[key].append(v)

    #############

    # user_grad = {}
    # for key in user_dis.keys():
    #     user_grad[key] = []
    #     for i in range(int(len(user_dis[key])/50)):
    #         dis = math.fabs(user_dis[key][i*50] - user_dis[key][i*50+49])
    #         grad = dis/50
    #         user_grad[key].append(grad)

    ############ Smooth
    arr = {}
    for key in s.keys():
        arr[key] = []
        s[key] = smooth(s[key], 50)
        bound = 1
        while arr[key] == []:
            for i in range(len(s[key])):
                if s[key][i] <= bound and i > 500:
                    arr[key].append(i)
                    break
            bound+=0.1

    beg = {}
    for key in s.keys():
        beg[key] = []
        bound = 0.7
        while beg[key] == []:
            for i in range(len(s[key])-1, -1, -1):
                if s[key][i] <= bound:
                    beg[key].append(i)
                    break
            bound+=0.1

    for key in arr.keys():
        arr[key].append(user[key][arr[key][0]][3])
        beg[key].append(user[key][beg[key][0]][3])

    # # ########## new figure
    # user_begin = {}
    # for key in user.keys():
    #     user_begin[key] = user[key][0][3]
    # user_begin = sorted(user_begin.items(), key=operator.itemgetter(1))
    # for i in range(1, len(user_begin)):
    #     user_begin[i] = list(user_begin[i])
    #     for j in range(len(user[user_begin[0][0]])):
    #          if user_begin[i][1] - user[user_begin[0][0]][j][3] < 0:
    #              user_begin[i][1] = j
    #              break
    # user_begin[0] = list(user_begin[0])
    # user_begin[0][1] = 0
    # user_begin_dic = {}
    # for item in user_begin:
    #     user_begin_dic[item[0]] = item[1]

    # ########### figure
    #
    # symb = ['o', 'D', 'h', '8', 'p', '+', 's', '*', 'd', 'v', '<', '>', '^', 'x', '.', ',', '1', '2', '3', '4', 'd',
    #         'H', '_', '|', 'D', 'x', 'o', 'D', 'h', '8', 'p', '+', 's', 'o', 'D', 'h', '8', 'p', '+', 's', '*', 'd',
    #         'v', '<', '>', '^', 'x', '.', ',', '1', '2', '3', '4', 'd',
    #         'H', '_', '|', 'D', 'x', 'o', 'D', 'h', '8', 'p', '+', 's']
    # col = ['r', 'g', 'b', 'violet', 'm', 'y', 'k', 'thistle', 'orange', 'brown', 'tomato', 'thistle', 'teal', 'tan',
    #        'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'r', 'orange', 'brown', 'tomato', 'thistle', 'teal',
    #        'tan', 'slateblue', 'r', 'g', 'b', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'thistle', 'orange', 'brown',
    #        'tomato', 'thistle', 'teal', 'tan',
    #        'slateblue', 'r', 'g', 'b', 'violet', 'm', 'y', 'k', 'r', 'orange', 'brown', 'tomato', 'thistle', 'teal',
    #        'tan', 'slateblue', 'r', 'g', 'b']
    # i = 0
    # for key in ang_dis.keys():
    #     # plt.subplot(411)
    #     # plt.scatter(numpy.arange(0, len(ang_dis[key])), numpy.asarray(ang_dis[key]),
    #     #                         marker = '*', s = 10)
    #     # plt.ylim((0, 10))
    #     # plt.xlabel("t")
    #     # plt.ylabel("Variation of Angle")
    #
    #     # plt.subplot(412)
    #     # plt.scatter(numpy.arange(0, len(s[key])), numpy.asarray(smooth(s[key], 50)),
    #     #             marker='*', s=10)
    #     # plt.xlabel("t")
    #     # plt.ylabel("Variation of Smoothed Speed")
    #
    #     plt.subplot(211)
    #     plt.plot(np.arange(user_begin_dic[key], user_begin_dic[key] + len(user_dis[key])), np.asarray(user_dis[key]), color = col[i], marker = symb[i], linewidth=0.5, markersize=0.5)
    #     plt.xlabel("t", fontsize = 18)
    #     plt.ylabel("Current Distance from the start point", fontsize=18)
    #     plt.title('Year 1')
    #     plt.subplot(212)
    #     # plt.scatter(numpy.arange(0, len(s[key])), numpy.asarray(s[key]),
    #     #                         marker = '*', s = 10)
    #     plt.plot(np.arange(user_begin_dic[key], user_begin_dic[key] + len(s[key])), np.asarray(smooth(s[key], 50)), color = col[i], marker = symb[i], linewidth=0.5, markersize=0.5)
    #     plt.xlabel("t", fontsize=18)
    #     plt.ylim((0, 6))
    #     plt.ylabel("Variation of Speed", fontsize=18)
    #
    #     i += 1
    #
    # plt.show()
    # exit()

    ### there is the time of four sequences.
    user_lag = {}
    departure = []
    return_init = []
    return_arrival = []
    for k in user.keys():
        user_lag[k] = []
        user_lag[k].append(user[k][0][3])
        user_lag[k].append(arr[k][1])
        user_lag[k].append(beg[k][1])
        user_lag[k].append(user[k][-1][3])

        departure.append(user[k][0][3])
        return_init.append(beg[k][1])
        return_arrival.append(user[k][-1][3])

    print(departure)
    print(return_init)
    print(return_arrival)

    print(np.std(departure))
    print(np.std(return_init))
    print(np.std(return_arrival))
    exit()

    four_Matrix = []
    for z in range(4):
        total = []
        for k in user_lag.keys():
            part = []
            for j in user_lag.keys():
                part.append(user_lag[k][z] - user_lag[j][z])
            total.append(part)

        total = np.asarray(total)
        four_Matrix.append(total)

    total_value_1 = 0
    total_value_2 = 0
    total_value_3 = 0
    total_value_4 = 0
    for i in range(np.shape(four_Matrix[0])[0]):

        a = four_Matrix[0][i]
        b = four_Matrix[1][i]
        c = four_Matrix[2][i]
        d = four_Matrix[3][i]
        v_1 = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        v_2 = np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))
        v_3 = np.dot(c, d) / (np.linalg.norm(c) * np.linalg.norm(d))
        v_4 = np.dot(a, d) / (np.linalg.norm(a) * np.linalg.norm(d))
        total_value_1 = total_value_1 + v_1
        total_value_2 = total_value_2 + v_2
        total_value_3 = total_value_3 + v_3
        total_value_4 = total_value_4 + v_4

    pz = []
    print('year: ' + str(vv))
    print(np.shape(four_Matrix[0]))
    print(total_value_1/np.shape(four_Matrix[0])[0])
    print(total_value_2/np.shape(four_Matrix[0])[0])
    print(total_value_3 / np.shape(four_Matrix[0])[0])
    print(total_value_4 / np.shape(four_Matrix[0])[0])
    pz.append(total_value_1/np.shape(four_Matrix[0])[0])
    pz.append(total_value_2/np.shape(four_Matrix[0])[0])
    pz.append(total_value_3 / np.shape(four_Matrix[0])[0])
    pz.append(total_value_4 / np.shape(four_Matrix[0])[0])
    return pz


def usingLagverify():
    import numpy as np
    x = [[0.539080, 0.548946, 0.239938, 0.211719, 0.592776, -0.154975, 0.848793, 0.337857, 0.873422, 0.774870],
         [0.307120, 0.148480, -0.153649, -0.416625, -0.714342, 0.447420, -0.627821, 0.505711, 0.642090],
         [0.473476, 0.814128, 0.760878, 0.464855, 0.396647, 0.428958, 0.873427, 0.302795, 0.932775, 0.856712],
         [0.864347, 0.247758, -0.244935, 0.222786, -0.461732, 0.672599, 0.086422, 0.187584, 0.165847, 0.318563]]

    st_ = []
    ave_ = []
    for item in x:
        st_.append(numpy.std(item))
        ave_.append(numpy.mean(item))

    print(ave_)
    print(st_)
    plt.errorbar(np.arange(1, 5), np.asarray(ave_), yerr=np.asarray(st_), fmt='o', ecolor='g', linestyle='dotted')
    # import numpy as np
    #
    # mat = []
    # for i in range(1, 2):
    #     mat.append(lagVerify(i))
    #
    # mat = np.asarray(mat)
    # import pickle
    # pickle.dump(mat, open('mat.txt', 'wb'))
    # # mat = pickle.load(open('mat.txt', 'rb'))
    # mat[9][3] = 0.3185632
    # mat[8][3] = 0.1658465
    # mat[6][3] = 0.0864216
    # import seaborn as sns
    # ax = sns.heatmap(mat, annot=True, fmt='f')
    # ax.set(xticklabels=[1, 2, 3, 4])
    # ax.set(xticklabels=['Out_begin Vs Out_arrival', 'Out_arrival Vs Return_begin', 'Return_begin Vs Return_arrival', 'Out_Begin Vs Return_arrival'])
    plt.xticks(rotation=10)
    plt.ylabel('Correlation Coefficient', fontsize=18)
    plt.xticks([1, 2, 3, 4],
               ['Out_begin Vs Out_arrival', 'Out_arrival Vs Return_begin', 'Return_begin Vs Return_arrival',
                'Out_Begin Vs Return_arrival'])
    plt.show()

def separateUser():

    time_dic, _, _ = new_original_distance_inter6()
    for zzz in range(10):
        y = zzz
        begin = secondP_list[y] // 4 + 731996
        end = thirdP_list[y] // 4 + 731996

        user = {}
        for i in range(begin, end+1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k not in user:
                            user[k] = time_dic[i][j][k]
                        else:
                            user[k].extend(time_dic[i][j][k])

        record = []
        for k in user.keys():
            if len(user[k]) < 1000:
                record.append(k)

        for item in record:
            user.pop(item)
        print(user.keys())

def CheckRepeatId():
    u_id = {}
    sensor = []
    repeat_animal = []
    f = open("names.txt", 'r')
    for line in f:
        sensor_id = line.split('\n')[0].split(' ')[-1]
        sensor.append(sensor_id)

    for i in range(len(sensor)):
        if sensor[i] not in u_id:
            u_id[sensor[i]] = []
            u_id[sensor[i]].append(i+1)
        else:
            u_id[sensor[i]].append(i + 1)

    for key in u_id.keys():
        if len(u_id[key]) >1:
            repeat_animal.append(u_id[key])

    repeat_animal = sorted(repeat_animal, key=itemgetter(0))

def RealEveryYearId():
    file = open('userId_years', 'r')
    animal_id = {}
    count = 1
    for i in range(303):
        animal_id[i+1] = []
    for line in file:
        x = list(map(int, line.split('\n')[0].split(', ')))
        for item in x:
            animal_id[item].extend([count, 1])
        count = count + 1
    file.close()

    file = open('userId_years_long', 'r')
    count = 1
    for line in file:
        x = list(map(int, line.split('\n')[0].split(', ')))
        for item in x:
            animal_id[item].extend([count, 2])
        count = count + 1
    file.close()

    return animal_id

def DrawSameAnimal():
    from mpl_toolkits.basemap import Basemap
    import matplotlib.patches as mpatches
    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec
    animal_id = RealEveryYearId()

    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y+9] // 4 + 731996

    user = {}
    for i in range(begin, end + 1):
        if i in time_dic.keys():
            for j in time_dic[i].keys():
                for k in time_dic[i][j].keys():
                    if k not in user:
                        user[k] = time_dic[i][j][k]
                    else:
                        user[k].extend(time_dic[i][j][k])

    file = open('repeatId_process', 'r')
    i = 0
    for line in file:
        user_list = list(map(int, line.split('\n')[0].split(', ')))
        plt.figure()
        m = Basemap(projection='mill',
                    llcrnrlat=25,
                    llcrnrlon=-210,
                    urcrnrlat=63,
                    urcrnrlon=-110)
        m.drawcoastlines()
        m.fillcontinents(color='coral')
        for j in range(len(user[user_list[0]])):
            x, y = m(user[user_list[0]][j][2], user[user_list[0]][j][1])
            m.plot(x, y, marker='*', color='red', markersize=1)

        for j in range(len(user[user_list[1]])):
            x, y = m(user[user_list[1]][j][2], user[user_list[1]][j][1])
            m.plot(x, y, marker='^', color='blue', markersize=1)
        red_user = animal_id[user_list[0]]
        if red_user[1] == 1:
            red_label = str(user_list[0]) + ' Year ' + str(red_user[0]) + ' short trip'
        else:
            red_label = str(user_list[0]) + ' Year ' + str(red_user[0]) + ' long trip'
        red_patch = mpatches.Patch(color='red', label=red_label)
        blue_user = animal_id[user_list[1]]
        if blue_user[1] == 1:
            blue_label = str(user_list[1]) + ' Year ' + str(blue_user[0]) + ' short trip'
        else:
            blue_label = str(user_list[1]) + ' Year ' + str(blue_user[0]) + ' long trip'
        blue_patch = mpatches.Patch(color='blue', label=blue_label)
        plt.legend(handles=[red_patch, blue_patch])

        plt.title(str(user_list[0]) + ' ' + str(user_list[1]))
        plt.savefig('/Users/peis/Documents/Comp/' + str(i), dpi=(200))
        i = i + 1
        plt.close()

def spe_dis_dis():
    import numpy as np
    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996
    zzz = 0
    spe_dis_rel = []
    for animal_id in user_id_year_long:
        print('Year: ' + str(zzz))
        user = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:
                            if k not in user:
                                user[k] = time_dic[i][j][k]
                            else:
                                user[k].extend(time_dic[i][j][k])

        for k in user.keys():
            user[k] = sorted(user[k], key=itemgetter(3))

        for k in user.keys():
            for i in range(len(user[k])):
                user[k][i].append(CalDis(user[k][i][1], user[k][i][2],
                                         user[k][0][1], user[k][0][2]))
        record = []
        for k in user.keys():
            if len(user[k]) < 2000:
                record.append(k)

        for item in record:
            user.pop(item)

        s = {}
        for key in user.keys():
            s[key] = []
            for zz in range(1, len(user[key])):
                d = CalDis(lef=user[key][zz][1], rig=user[key][zz][2],
                           lef_mean=user[key][zz - 1][1], rig_mean=user[key][zz - 1][2])
                time_d = (user[key][zz][3] - user[key][zz - 1][3]) * 24
                speed = d / time_d
                s[key].append(speed)
            s[key].insert(0, 0)

        ani_dis = {}
        for key in user.keys():
            ani_dis[key] = []
            for zz in range(len(user[key])):
                ani_dis[key].append(user[key][zz][-1])

        for key in user.keys():
            for zz in range(len(user[key])):
                if ani_dis[key][zz] != 0:
                    spe_dis_rel.append((ani_dis[key][zz], s[key][zz]))

        zzz += 1

    spe_dis_rel = sorted(spe_dis_rel, key=lambda tup: tup[0])
    max_dis = max(spe_dis_rel, key=lambda tup: tup[0])
    dis_dic = {}
    for i in range(int(math.ceil(max_dis[0]/100) - 1)+1):
        dis_dic[i] = []

    for item in spe_dis_rel:
        dis_dic[math.ceil(item[0]/100)-1].append(item)

    from numpy import mean, std
    ave_sd = []
    for key in dis_dic.keys():
        ave = tuple(map(mean, zip(*dis_dic[key])))
        st = tuple(map(std, zip(*dis_dic[key])))
        ave_sd.append([key, ave[1], st[1]])

    ave_ = []
    sd_ = []
    for item in ave_sd:
        ave_.append(item[1])
        sd_.append(item[2])

    all_dis = []
    qq = 0
    for key in dis_dic.keys():
        part = []
        for item in dis_dic[key]:
            if item[1] < ave_[qq]+sd_[qq] and item[1] > ave_[qq] - sd_[qq]:
                part.append(item[1])

        all_dis.append(part)
        qq += 1
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    axes.violinplot(all_dis,
                       showmeans=False,
                       showmedians=True)


    # from numpy import mean, std
    # ave_sd = []
    # for key in dis_dic.keys():
    #     ave = tuple(map(mean, zip(*dis_dic[key])))
    #     st = tuple(map(std, zip(*dis_dic[key])))
    #     ave_sd.append([key, ave[1], st[1]])
    #
    # ave_ = []
    # sd_ = []
    # for item in ave_sd:
    #     ave_.append(item[1])
    #     sd_.append(item[2])

    # print(len(ave_))
    # plt.errorbar(np.arange(100, 4200, 100), np.asarray(ave_), yerr=np.asarray(sd_), fmt='o', ecolor='g')
    plt.xlabel('Distance (100km)', fontsize=18)
    plt.ylabel('Speed (km/day)', fontsize=18)
    plt.show()


def var_ang_distance():
    import numpy as np
    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    return_point = return_point_dis()

    zzz = 0

    ani_var = []
    ani_dis = []
    for animal_id in user_id_year_long:
        print('Year: ' + str(zzz))

        part_time_dic = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:


                            l = time_dic[i][j][k]
                            l_lef, l_rig = CalCenPoint(l)
                            t = average_time(l)

                            if i not in part_time_dic:
                                part_time_dic[i] = {}
                                part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                            else:
                                if j not in part_time_dic[i]:
                                    part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                                else:
                                    part_time_dic[i][j].append([k, l_lef, l_rig, t])


        # user = {}
        part_time_keys = sorted(list(part_time_dic.keys()), reverse=False)
        # for i in part_time_keys:
        #     for j in part_time_dic[i].keys():
        #         for k in part_time_dic[i][j]:
        #             if k[0] in animal_id:
        #                 if k[0] not in user:
        #                     user[k[0]] = [k]
        #                 else:
        #                     user[k[0]].append(k)

        center_line = []
        for day in part_time_keys:
            for hour in part_time_dic[day].keys():
                lef, rig = CalCenPoint(part_time_dic[day][hour])
                center_line.append([lef, rig])

        center_line_deg = []
        for i in range(1, len(center_line)):
            lef = center_line[i][0] - center_line[i - 1][0]
            rig = center_line[i][1] - center_line[i - 1][1]

            lef_d = CalDis(lef = center_line[i][0], rig=center_line[i-1][1],
                           lef_mean = center_line[i - 1][0],
                           rig_mean= center_line[i - 1][1])

            right_d = CalDis(lef = center_line[i][0], rig=center_line[i][1],
                             lef_mean=center_line[i][0],
                             rig_mean=center_line[i-1][1])

            deg = returnDegreefromSinCos(lef_d, right_d, lef, rig)
            center_line_deg.append(deg)
        center_line_deg.insert(0, 0)

        ##### animals

        animal_deg = {}
        for animal in animal_id:
            animal_deg[animal] = []
        count = 0
        for i in part_time_keys:
            for j in part_time_dic[i].keys():
                for k in part_time_dic[i][j]:
                    animal_deg[k[0]].append(k)
                count += 1
                for key_ in animal_deg:
                    if len(animal_deg[key_]) != count:
                        animal_deg[key_].append([0])


        #### distance
        animal_dis = {}
        for animal in animal_id:
            animal_dis[animal] = []

        for animal in animal_deg.keys():
            for i in range(len(animal_deg[animal])):
                if animal_deg[animal][i] != [0]:
                    d = CalDis(center_line[i][0], center_line[i][1],
                               float(animal_deg[animal][i][1]), float(animal_deg[animal][i][2]))
                    animal_dis[animal].append(d)

                else:
                    animal_dis[animal].append('0')

        # return point
        animal_return = {}
        for animal in animal_id:
            animal_return[animal] = []

        beg = return_point[zzz]
        for animal in animal_deg.keys():
            animal_beg = beg[animal][1]
            for i in range(len(animal_deg[animal])):
                if animal_deg[animal][i] != [0]:
                    animal_return[animal].append(math.fabs(animal_beg - animal_deg[animal][i][-1]))
                else:
                    animal_return[animal].append([0])

            compare = [10000, 0]
            for i in range(len(animal_deg[animal])):
                if animal_return[animal][i] != [0]:
                    if animal_return[animal][i] < compare[0]:
                        compare[0] = animal_return[animal][i]
                        compare[1] = i

            animal_deg[animal] = animal_deg[animal][compare[1]-1:]
            animal_dis[animal] = animal_dis[animal][compare[1]-1:]

        animal_deg_true = {}
        animal_dis_true = {}
        for animal in animal_id:
            animal_deg_true[animal] = []
            animal_dis_true[animal] = []

        inter = 2
        for animal in animal_deg.keys():
            r_i = 0
            for i in range(inter, len(animal_deg[animal]), inter):
                if animal_deg[animal][i-inter] != [0] and animal_deg[animal][i] != [0]:

                    lef = animal_deg[animal][i][1] - animal_deg[animal][i - inter][1]
                    rig = animal_deg[animal][i][2] - animal_deg[animal][i - inter][2]

                    lef_d = CalDis(lef=animal_deg[animal][i][1], rig=animal_deg[animal][i - inter][2],
                                   lef_mean=animal_deg[animal][i - inter][1],
                                   rig_mean=animal_deg[animal][i - inter][2])

                    right_d = CalDis(lef=animal_deg[animal][i][1], rig=animal_deg[animal][i][2],
                                     lef_mean=animal_deg[animal][i][1],
                                     rig_mean=animal_deg[animal][i - inter][2])

                    deg = returnDegreefromSinCos(lef_d, right_d, lef, rig)
                    animal_deg_true[animal].append(deg)

                    ## distance

                    animal_dis_true[animal].append(animal_dis[animal][i-inter])
                    r_i = i

                else:
                    animal_deg_true[animal].append(0)
                    animal_dis_true[animal].append(0)
            animal_deg_true[animal].insert(0, 0)
            animal_dis_true[animal].insert(len(animal_dis_true[animal]), animal_dis[animal][r_i])

        animal_deg_var = {}
        for animal in animal_id:
            animal_deg_var[animal] = []
        for animal in animal_deg_true.keys():
            for i in range(1, len(animal_deg_true[animal])):
                if animal_deg_true[animal][i] != 0 and animal_deg_true[animal][i-1] != 0:
                    var = math.fabs(animal_deg_true[animal][i] - animal_deg_true[animal][i-1])
                    if var > 180:
                        var = 360 - 180
                    animal_deg_var[animal].append(var)
                elif animal_deg_true[animal][i] != 0 and animal_deg_true[animal][i-1] == 0:
                    animal_deg_var[animal].append(0)
                else:
                    animal_deg_var[animal].append('0')

            animal_deg_var[animal].insert(0, 0)

        for animal in animal_deg_var.keys():
            time_break = 0
            for i in range(len(animal_deg_var[animal])):
                if animal_deg_var[animal][i] == '0':
                    time_break = i
                    break
            animal_deg_var[animal] = animal_deg_var[animal][:time_break]
            animal_dis_true[animal] = animal_dis_true[animal][:time_break]

            while len(animal_deg_var[animal]) > 10:
                ani_dis.append(np.mean(animal_dis[animal][:10]))
                ani_var.append(np.mean(animal_deg_var[animal][:10]))
                animal_deg_var[animal] = animal_deg_var[animal][10:]
                animal_dis_true[animal] = animal_dis_true[animal][10:]

            # ani_dis.extend(animal_dis[animal])
            # ani_var.extend(animal_deg_var[animal])
        zzz += 1

    print(len(ani_dis))
    print(len(ani_var))
    plt.scatter(np.asarray(ani_dis), np.asarray(ani_var), marker = '.', s = 10)
    plt.ylabel('Variance in Direction of Swimming')
    plt.xlabel('Distance (km)')
    plt.show()

def return_ranking(par = 2):
    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    zzz = 0
    global varss
    if par == 1:
        varss = user_id_year_short
    elif par == 2:
        varss = user_id_year_long

    all_ranking = []
    for animal_id in varss:
        print('Year: ' + str(zzz))

        part_time_dic = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:

                            l = time_dic[i][j][k]
                            l_lef, l_rig = CalCenPoint(l)
                            t = average_time(l)

                            if i not in part_time_dic:
                                part_time_dic[i] = {}
                                part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                            else:
                                if j not in part_time_dic[i]:
                                    part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                                else:
                                    part_time_dic[i][j].append([k, l_lef, l_rig, t])



        part_time_keys = sorted(list(part_time_dic.keys()), reverse=False)

        # center_line = []
        # for day in part_time_keys:
        #     for hour in part_time_dic[day].keys():
        #         lef, rig = CalCenPoint(part_time_dic[day][hour])
        #         center_line.append([lef, rig])
        #
        # print(len(center_line))

        all_animal_item = []
        for day in part_time_keys:
            for hour in part_time_dic[day].keys():
                if len(part_time_dic[day][hour]) == len(animal_id):
                    all_animal_item.append(part_time_dic[day][hour])



        center_line = []
        for i in range(len(all_animal_item)):
            lef, rig = CalCenPoint(all_animal_item[i])
            center_line.append([lef, rig])

        all_animal_item_distance = []
        for i in range(len(all_animal_item)):
            part = []
            for item in all_animal_item[i]:
                part_dis = CalDis(center_line[i][0], center_line[i][1],
                                  item[1], item[2])
                part.append([item[0], part_dis])
            all_animal_item_distance.append(part)

        from operator import itemgetter
        all_animal_item_ranking = []
        for i in range(len(all_animal_item_distance)):
            part = []
            new_item = sorted(all_animal_item_distance[i], key=itemgetter(1))
            r = 1
            for item in new_item:
                part.append([item[0], r])
                r += 1
            all_animal_item_ranking.append(part)

        dic_rank = {}
        for an in animal_id:
            dic_rank[an] = []

        for i in range(len(all_animal_item_ranking)):
            for item in all_animal_item_ranking[i]:
                dic_rank[item[0]].append(item[1])

        all_ranking.append(dic_rank)
        zzz += 1

    return all_ranking

def return_fre_dis(l):
    import collections
    counter = collections.Counter(l)
    v = list(counter.values())
    v = list(map(lambda x: x / sum(v), v))
    return v


def rank_ks_test():
    from scipy import stats
    import random
    import numpy as np

    rank_short = return_ranking(1)

    rank_long = return_ranking(2)

    animal_id = RealEveryYearId()

    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    file = open('repeatId_process', 'r')

    ks_value = []

    for line in file:

        user_list = list(map(int, line.split('\n')[0].split(', ')))
        if animal_id[user_list[0]][1] == 1:
            p1 = rank_short[animal_id[user_list[0]][0]-1][user_list[0]]
        else:
            p1 = rank_long[animal_id[user_list[0]][0] - 1][user_list[0]]

        if animal_id[user_list[1]][1] == 1:
            p2 = rank_short[animal_id[user_list[1]][0] - 1][user_list[1]]
        else:
            p2 = rank_long[animal_id[user_list[1]][0] - 1][user_list[1]]

        #### frequency

        p1 = return_fre_dis(p1)
        p2 = return_fre_dis(p2)

        value = stats.ks_2samp(p1, p2)

        u1 = []
        u2 = []

        for key, item in animal_id.items():
            if item == animal_id[user_list[0]]:
                u1.append(key)
            elif item == animal_id[user_list[1]]:
                u2.append(key)


        u1.remove(user_list[0])
        u2.remove(user_list[1])

        v_total = []

        for i in range(min(5, min(len(u1), len(u2)))):
            uu1 = random.choice(u1)
            uu2 = random.choice(u2)

            if animal_id[uu1][1] == 1:
                pp1 = rank_short[animal_id[uu1][0] - 1][uu1]
            else:
                pp1 = rank_long[animal_id[uu1][0] - 1][uu1]

            if animal_id[uu2][1] == 1:
                pp2 = rank_short[animal_id[uu2][0] - 1][uu2]
            else:
                pp2 = rank_long[animal_id[uu2][0] - 1][uu2]

            pp1 = return_fre_dis(pp1)
            pp2 = return_fre_dis(pp2)

            v = stats.ks_2samp(pp1, pp2)
            v_total.append(v)

        ks_value.append([value[0], np.mean(v_total)])

    ks_value_l = []
    ks_value_r = []
    for item in ks_value:
        ks_value_l.append(item[0])
        ks_value_r.append(item[1])
    fig, ax = plt.subplots()
    index = np.arange(len(ks_value))
    bar_width = 0.35
    opacity = 0.4
    rects1 = plt.bar(index, ks_value_l, bar_width, alpha=opacity, color='b', label='Paired Animals with Same ID')
    rects2 = plt.bar(index + bar_width, ks_value_r, bar_width, alpha=opacity, color='r', label='Random Selected Animals')
    plt.xlabel('Paired Animals', fontsize=18)
    plt.ylabel('KS_Score', fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(index.tolist(),
               rank_xlabel, rotation=45)

    plt.tight_layout()
    plt.show()


def traj_ks_test():
    from scipy import stats
    import random
    import numpy as np
    animal_id = RealEveryYearId()

    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y+9] // 4 + 731996

    user = {}
    for i in range(begin, end + 1):
        if i in time_dic.keys():
            for j in time_dic[i].keys():
                for k in time_dic[i][j].keys():
                    if k not in user:
                        user[k] = time_dic[i][j][k]
                    else:
                        user[k].extend(time_dic[i][j][k])

    file = open('repeatId_process_traj', 'r')
    i = 0

    ks_value = []
    for line in file:
        user_list = list(map(int, line.split('\n')[0].split(', ')))
        p1 = []
        p2 = []
        for j in range(len(user[user_list[0]])):
            p1.append(user[user_list[0]][j][2])
        for j in range(len(user[user_list[1]])):
            p2.append(user[user_list[1]][j][2])

        value = stats.ks_2samp(p1, p2)
        print(value[0])
        u1 = []
        u2 = []
        for key, item in animal_id.items():
            if item == animal_id[user_list[0]]:
                u1.append(key)
            elif item == animal_id[user_list[1]]:
                u2.append(key)

        u1.remove(user_list[0])
        u2.remove(user_list[1])

        v_total = []
        for i in range(min(len(u1), len(u2))):
            uu1 = random.choice(u1)
            uu2 = random.choice(u2)
            pp1 = []
            pp2 = []
            for j in range(len(user[uu1])):
                pp1.append(user[uu1][j][2])
            for j in range(len(user[uu2])):
                pp2.append(user[uu2][j][2])

            v = stats.ks_2samp(pp1, pp2)
            v_total.append(v)
        print(np.mean(v_total))

        ks_value.append([value[0], np.mean(v_total)])


    print(ks_value)
    ks_value_l = []
    ks_value_r = []
    for item in ks_value:
        ks_value_l.append(item[0])
        ks_value_r.append(item[1])
    fig, ax = plt.subplots()
    index = np.arange(len(ks_value))
    bar_width = 0.35
    opacity = 0.4
    rects1 = plt.bar(index, ks_value_l, bar_width, alpha=opacity, color='b', label='Paired Animals with Same ID')
    rects2 = plt.bar(index + bar_width, ks_value_r, bar_width, alpha=opacity, color='r', label='Random Selected Animal')
    plt.xlabel('Paired Animals', fontsize=18)
    plt.ylabel('KS_Score', fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(index.tolist(),
               traj_xlabel, rotation=45)

    plt.tight_layout()
    plt.show()

def DrawSpread():
    import pandas as pd
    from time import strptime, mktime
    from datetime import datetime
    import numpy as np
    df = pd.read_csv('data_with_straightness_index_window10.csv')
    spread = df['spread'].values
    time = df['utc_timestamps'].values
    id = df['animal_id'].values
    time_dic = {}
    l = []
    for i in range(len(spread)):
        timeArray = strptime(time[i], "%Y-%m-%dT%H:%M:%S")
        t = datetime.fromtimestamp(mktime(timeArray))
        t = date.toordinal(t.date())
        if t not in time_dic.keys():
            time_dic[t] = {}
            if id[i] not in time_dic[t].keys():
                time_dic[t][id[i]] = []
                time_dic[t][id[i]].append(spread[i])
            else:
                time_dic[t][id[i]].append(spread[i])
        else:
            if id[i] not in time_dic[t].keys():
                time_dic[t][id[i]] = []
                time_dic[t][id[i]].append(spread[i])
            else:
                time_dic[t][id[i]].append(spread[i])


    for key in time_dic.keys():
        for ani in time_dic[key].keys():
            time_dic[key][ani] = sum(time_dic[key][ani])/len(time_dic[key][ani])

    for key in time_dic.keys():
        s = sum(list(time_dic[key].values()))/len(time_dic[key].keys())
        l.append(s)

    xlabel = [0, 360, 720, 1080, 1420, 1780, 2140, 2500, 2860, 3220, 3580]
    from numpy import std, mean
    week = {}
    for i in range(52):
        week[i] = []

    for x in range(len(xlabel)-1):
        for z in range(1, 52):
            week[z-1].extend(l[xlabel[x]+(z-1)*7:xlabel[x]+z*7])

        week[51].extend(l[xlabel[x]+51*7:xlabel[x]+360])

    ave_ = []
    sd_ = []
    for key in week.keys():
        nn = week[key]
        st = std(nn)
        mea = mean(nn)
        ave_.append(mea)
        sd_.append(st)
    plt.figure(figsize=(10,5))
    plt.errorbar(np.arange(1, 53), np.asarray(ave_), yerr=np.asarray(sd_), fmt='o', ecolor='g')
    # plt.xlabel('Week', fontsize=25, fontname = 'times new roman')
    # plt.ylabel(r'Group Spread ($Km^2$)', fontsize=25, fontname = 'times new roman')
    plt.xticks(fontsize=25, fontname='times new roman')
    plt.yticks(fontsize=15, fontname='times new roman')
    plt.show()

    exit()
    # plt.plot(np.arange(0, len(l)), np.asarray(l))
    # plt.ylabel(r'Group Spread ($Km^2$)', fontsize = 25, fontname = 'times new roman')
    # plt.xlabel('Date', fontsize = 25, fontname = 'times new roman')
    # plt.xticks(xlabel,
    #                       ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'])
    # plt.show()


def neig_lags(par=2):
    import operator
    import collections

    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    return_point = return_point_dis()

    zzz = 0
    global varss
    if par == 1:
        varss = user_id_year_short
    elif par == 2:
        varss = user_id_year_long


    all_lags = []
    for animal_id in varss:
        print('Year: ' + str(zzz))

        part_time_dic = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:

                            l = time_dic[i][j][k]
                            l_lef, l_rig = CalCenPoint(l)
                            t = average_time(l)

                            if i not in part_time_dic:
                                part_time_dic[i] = {}
                                part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                            else:
                                if j not in part_time_dic[i]:
                                    part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                                else:
                                    part_time_dic[i][j].append([k, l_lef, l_rig, t])

        for an in animal_id:
            all_neig = []
            day = 2
            while all_neig == []:
                for k in part_time_dic[int(return_point[zzz][an][1]) - day].keys():
                    ob = []
                    for item in part_time_dic[int(return_point[zzz][an][1]) - day][k]:
                        if item[0] == an:
                            ob = item

                    an_value = {}
                    print(part_time_dic[int(return_point[zzz][an][1]) - day][k])
                    for item in part_time_dic[int(return_point[zzz][an][1]) - day][k]:

                        if item[0] != an:
                            d = CalDis(item[1], item[2], ob[1], ob[2])
                            an_value[item[0]] = d
                    if an_value:
                        all_neig.append(min(an_value.items(), key=operator.itemgetter(1))[0])
                day += 1
            counter = collections.Counter(all_neig)
            neig = max(counter.items(), key=operator.itemgetter(1))[0]

            all_lags.append(int(return_point[zzz][neig][1] - return_point[zzz][an][1]))
        zzz += 1
    print(all_lags)
    print(len(all_lags))
    exit()

def usingNeig_lags():
    lags = [41, 2, 18, 8, 26, -8, -22, -19, -26, 2, 24, 29, 107, 42, -14,
            14, -15, 6, 7, 14, 41, 18, -60, 75, 65, -6, 17, -19, 78, 36,
            -41, 46, 55, -4, 54, 13, 13, 3, 44, -6, -13, -19, -3, 2, -7,
            95, -20, -15, 20, 15, 50, 126, -29, 2, 1, 25, -14, -36, -1,
            17, 11, -38, 16, 14, 80, -12, 9, -20, -5, 44, 1, 5, 21, 73,
            0, -1, -21, 96, 1, 4, 89, -21, 10, -39, 39, 28, 0, 37, -43,
            15, -5, -29, 61, 0, -36, -32, -40, 25, 9, 16, 27, -2, 2, -3,
            -26, 21, -11, -9, -28, 11, -10, 9, -31, 23, -8, 1, -35, 8,
            -29, 13, -36, 10, -3, 36, 0, -9, 0, 10, -6, 0, 40, 9, -18,
            -3, -6, 6, 0,12, 22, -22, -41, -42, 23, 6, 53, -12, 6, -34,
            -7, 43, -8, -6, -31, -27, 31, -22, -6, -26, -26, -8, -28,
            -4, 12, -13, 11, 18, 9, 34, -9, 4, -22, 53, 18, 54, 25, -1,
            -9, -14, 5, 1, 2, 3, 19, -9, -2, -24, -4, -20, 0, -3, 19,
            -26, -10, 41, -4, 17, 55, 33, 15, 6, 4, -1, -18, -3, 1, -25,
            18, -33, -3, 18, -27, 0, 16, 0, 29, 56, -13, -8, 10, -3, 31,
            13, -10, -24, -5, -13, 5, -25, 10, -34, -30, 32, 10, -1, 24,
            -22, 3, -15, -3, 52, 33, 33, 30, 0, -7, 0, 51, -31, 35, 17,
            11, -3, 27, 4, 3, -16, 51, 17, -11, 16, -5, -15, -17, -4, -7,
            31, -3, 47, -15, -21, 8, 16, -8, -5, -12, 31, -15, 12, 34,
            36, 6, -6, -34, -3, -9, 24, 38, 32, -1, -22, 3, -31, 31, 4]
    fig, ax = plt.subplots(ncols=1, figsize=(8, 4))
    fre, bins, patches = ax.hist(lags, 60, normed=1, facecolor='b', alpha=0.75, edgecolor='black', cumulative=True)
    print(fre)
    print(bins)
    return fre, bins
    # ax.set_xlabel('lags (day)', fontsize = 18)
    # ax.set_ylabel('Probability', fontsize=18)
    # fig.tight_layout()
    # plt.show()

def lag_distribution():
    import numpy as np
    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    lags_distance_dis = []
    for animal_id in user_id_year_long:
        user = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:
                            if k not in user:
                                user[k] = time_dic[i][j][k]
                            else:
                                user[k].extend(time_dic[i][j][k])

        for k in user.keys():
            user[k] = sorted(user[k], key=itemgetter(3))

        for k in user.keys():
            for i in range(len(user[k])):
                user[k][i].append(CalDis(user[k][i][1], user[k][i][2],
                                         user[k][0][1], user[k][0][2]))

        record = []
        for k in user.keys():
            if len(user[k]) < 2000:
                record.append(k)

        for item in record:
            user.pop(item)


        # Calculate the speed
        s = {}
        for key in user.keys():
            s[key] = []
            for zz in range(1, len(user[key])):
                d = CalDis(lef=user[key][zz][1], rig=user[key][zz][2],
                           lef_mean=user[key][zz - 1][1], rig_mean=user[key][zz - 1][2])
                time_d = (user[key][zz][3] - user[key][zz - 1][3]) * 24
                speed = d / time_d
                s[key].append(speed)

        ############ Smooth

        beg = {}
        for key in s.keys():
            beg[key] = []
            bound = 0.7
            while beg[key] == []:
                for i in range(len(s[key]) - 1, -1, -1):
                    if s[key][i] <= bound:
                        beg[key].append(i)
                        break
                bound += 0.1

        for key in beg.keys():
            beg[key].append(user[key][beg[key][0]][3])
            beg[key].append(user[key][beg[key][0]][1])
            beg[key].append(user[key][beg[key][0]][2])


        beg_keys = list(beg.keys())

        for i in range(len(beg_keys)-1):
            for j in range(i+1, len(beg_keys)):
                time_lags = beg[beg_keys[i]][1] - beg[beg_keys[j]][1]
                distance_lags = CalDis(beg[beg_keys[i]][2], beg[beg_keys[i]][3], beg[beg_keys[j]][2], beg[beg_keys[j]][3])

                lags_distance_dis.append([time_lags, distance_lags])

                # if distance_lags < 200 and time_lags > 64:
                #     print(beg_keys[i])
                #     print(beg_keys[j])


    lags_distance_dis_rel = sorted(lags_distance_dis, key=lambda tup: tup[1])

    lag_dist = []

    for item in lags_distance_dis_rel:
        lag_dist.append(item[0])
    fig, ax = plt.subplots(ncols=1, figsize=(8, 4))
    fre, bins, patches = ax.hist(lag_dist, 60, normed=1, facecolor='b', alpha=0.75, edgecolor='black', cumulative =True)
    print(fre)
    print(bins)
    return fre, bins
    # ax.set_xlabel('lags (day)', fontsize=18)
    # ax.set_ylabel('Probability', fontsize=18)
    # fig.tight_layout()
    # plt.show()

def return_bins(bins):
    new_bins = []
    for i in range(1, len(bins)):
        new_bins.append((bins[i]+bins[i-1])/2)
    print(new_bins)
    return new_bins

def lags_dist_neig():
    import numpy as np
    fre, bins = lag_distribution()
    fre_1, bins_1 = usingNeig_lags()
    plt.show()
    bins_1 = return_bins(bins_1)
    bins_1.append(182.61)
    bins_1.insert(0, -176.08514952777864)
    fre_1 = fre_1.tolist()
    fre_1.insert(0, 0)
    fre_1.append(1)
    plt.plot(np.asarray(return_bins(bins)), np.asarray(fre), label='All Paired Animals')
    plt.plot(np.asarray(bins_1), np.asarray(fre_1), label='Animals with Nearest Neighbor')
    plt.xlabel('Lags (day)', fontsize=25, fontname='times new roman')
    plt.ylabel('Probability', fontsize=25, fontname='times new roman')
    plt.legend(prop={'size': 11})
    plt.show()

def area_km2():
    lef_ = CalDis(lef=26.121519, rig = -113.6894, lef_mean=26.121519, rig_mean=-179.99995)
    rig_ = CalDis(lef=26.121519, rig = -113.6894, lef_mean=59.783237, rig_mean=-113.6894)

    print(lef_*rig_)


def Compute_distance_score():
    import numpy as np
    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    return_point = return_point_dis()

    zzz = 0

    pair_distance = []
    near_neig_distance = []

    for animal_id in user_id_year_long:
        print('Year: ' + str(zzz))

        part_time_dic = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:


                            l = time_dic[i][j][k]
                            l_lef, l_rig = CalCenPoint(l)
                            t = average_time(l)

                            if i not in part_time_dic:
                                part_time_dic[i] = {}
                                part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                            else:
                                if j not in part_time_dic[i]:
                                    part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                                else:
                                    part_time_dic[i][j].append([k, l_lef, l_rig, t])


        # user = {}
        part_time_keys = sorted(list(part_time_dic.keys()), reverse=False)
        # for i in part_time_keys:
        #     for j in part_time_dic[i].keys():
        #         for k in part_time_dic[i][j]:
        #             if k[0] in animal_id:
        #                 if k[0] not in user:
        #                     user[k[0]] = [k]
        #                 else:
        #                     user[k[0]].append(k)

        center_line = []
        time_line = []
        for day in part_time_keys:
            for hour in part_time_dic[day].keys():
                lef, rig = CalCenPoint(part_time_dic[day][hour])
                center_line.append([lef, rig])
                time_line.append([day, hour])

        center_dis = []
        for i in range(1, len(center_line)):
            center_dis.append(CalDis(center_line[0][0], center_line[0][1], center_line[i][0],
                                     center_line[i][1]))
        center_dis.insert(0, 0)

        inter_animal = part_time_dic[time_line[center_dis.index(max(center_dis))][0]][time_line[center_dis.index(max(center_dis))][1]]
        for i in range(len(inter_animal)-1):
            part_neig = []
            for j in range(i+1, len(inter_animal)):
                pair_distance.append(CalDis(inter_animal[i][1], inter_animal[i][2],
                                            inter_animal[j][1], inter_animal[j][2]))
                part_neig.append(CalDis(inter_animal[i][1], inter_animal[i][2],
                                            inter_animal[j][1], inter_animal[j][2]))
            near_neig_distance.append(min(part_neig))

        zzz += 1


    print(np.mean(pair_distance))
    print(max(pair_distance))
    print(np.mean(near_neig_distance))
    print(max(near_neig_distance))

def animals_distance_from_original():

    import numpy as np
    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    lags_distance_dis = []
    for animal_id in user_id_year_short:
        dis_list = []
        user = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:
                            if k not in user:
                                user[k] = time_dic[i][j][k]
                            else:
                                user[k].extend(time_dic[i][j][k])

        for k in user.keys():
            user[k] = sorted(user[k], key=itemgetter(3))


        for k in user.keys():
            for i in range(len(user[k])):
                user[k][i].append(CalDis(user[k][i][1], user[k][i][2],
                                         user[k][0][1], user[k][0][2]))

        for k in user.keys():
            user[k] = sorted(user[k], key=itemgetter(6))

        for k in user.keys():
            # dd = date.fromordinal(int(user[k][-1][3]))
            # w = dd.isoformat()
            dis_list.append(int(user[k][-1][3]))

        m = np.median(sorted(dis_list))
        dd = date.fromordinal(int(m))
        w = dd.isoformat()
        print(w)

def group_animal_distance_from_original():
    import numpy as np
    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    return_point = return_point_dis()

    zzz = 0

    pair_distance = []
    near_neig_distance = []

    for animal_id in user_id_year_long_for_distance:
        print('Year: ' + str(zzz))

        part_time_dic = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:

                            l = time_dic[i][j][k]
                            l_lef, l_rig = CalCenPoint(l)
                            t = average_time(l)

                            if i not in part_time_dic:
                                part_time_dic[i] = {}
                                part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                            else:
                                if j not in part_time_dic[i]:
                                    part_time_dic[i][j] = [[k, l_lef, l_rig, t]]
                                else:
                                    part_time_dic[i][j].append([k, l_lef, l_rig, t])

        # user = {}
        part_time_keys = sorted(list(part_time_dic.keys()), reverse=False)
        # for i in part_time_keys:
        #     for j in part_time_dic[i].keys():
        #         for k in part_time_dic[i][j]:
        #             if k[0] in animal_id:
        #                 if k[0] not in user:
        #                     user[k[0]] = [k]
        #                 else:
        #                     user[k[0]].append(k)

        center_line = []
        time_line = []
        for day in part_time_keys:
            for hour in part_time_dic[day].keys():
                lef, rig = CalCenPoint(part_time_dic[day][hour])
                center_line.append([lef, rig])
                time_line.append([day, hour])

        center_dis = []
        for i in range(1, len(center_line)):
            center_dis.append(CalDis(center_line[0][0], center_line[0][1], center_line[i][0],
                                     center_line[i][1]))
        center_dis.insert(0, 0)
        dd = date.fromordinal(time_line[center_dis.index(max(center_dis))][0])
        w = dd.isoformat()
        print(w)
        # inter_animal = part_time_dic[time_line[center_dis.index(max(center_dis))][0]][
        #     time_line[center_dis.index(max(center_dis))][1]]

def group_animal_return_time():
    import numpy as np
    time_dic, _, _ = new_original_distance_inter6()
    y = 0
    begin = firstP_list[y] // 4 + 731996
    end = thirdP_list[y + 9] // 4 + 731996

    lags_distance_dis = []
    for animal_id in user_id_year_long:
        user = {}
        for i in range(begin, end + 1):
            if i in time_dic.keys():
                for j in time_dic[i].keys():
                    for k in time_dic[i][j].keys():
                        if k in animal_id:
                            if k not in user:
                                user[k] = time_dic[i][j][k]
                            else:
                                user[k].extend(time_dic[i][j][k])

        for k in user.keys():
            user[k] = sorted(user[k], key=itemgetter(3))

        for k in user.keys():
            for i in range(len(user[k])):
                user[k][i].append(CalDis(user[k][i][1], user[k][i][2],
                                         user[k][0][1], user[k][0][2]))

        record = []
        for k in user.keys():
            if len(user[k]) < 2000:
                record.append(k)

        for item in record:
            user.pop(item)

        # Calculate the speed
        s = {}
        for key in user.keys():
            s[key] = []
            for zz in range(1, len(user[key])):
                d = CalDis(lef=user[key][zz][1], rig=user[key][zz][2],
                           lef_mean=user[key][zz - 1][1], rig_mean=user[key][zz - 1][2])
                time_d = (user[key][zz][3] - user[key][zz - 1][3]) * 24
                speed = d / time_d
                s[key].append(speed)

        ############ Smooth

        beg = {}
        for key in s.keys():
            beg[key] = []
            bound = 0.7
            while beg[key] == []:
                for i in range(len(s[key]) - 1, -1, -1):
                    if s[key][i] <= bound:
                        beg[key].append(i)
                        break
                bound += 0.1

        for key in beg.keys():
            beg[key].append(user[key][beg[key][0]][3])
            beg[key].append(user[key][beg[key][0]][1])
            beg[key].append(user[key][beg[key][0]][2])

        beg_keys = list(beg.keys())
        return_list = []
        for key in beg.keys():
            return_list.append(int(beg[key][1]))
        dd = date.fromordinal(int(np.median(return_list)))
        w = dd.isoformat()
        print(w)


firstP_list = [0, 1389, 2802, 4224, 5649, 7044, 8485, 9968, 11397, 12864]
return_short = [178, 1389+195, 2802+233, 4224+253, 5649+176, 7044+219, 8485+257, 9968+212, 11397+202, 12864+202]
secondP_list = [399, 1779, 3261, 4625, 6123, 7574, 8909, 10418, 11839, 13280]
return_long = [792, 1389+790, 2802+783, 4224+749, 5649+780, 7044+763, 8485+797, 9968+856, 11397+830, 12864+814]
thirdP_list = [1388, 2802, 4224, 5649, 7044, 8485, 9968, 11397, 12864, 14261]

user_id_year_short = [[1, 2, 3, 4, 5], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                      [64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 59, 60, 62, 63],
                      [93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                      [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150],
                      [162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178],
                      [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 185, 186, 187, 188, 189, 190, 191],
                      [216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236],
                      [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 251, 252, 253, 254, 255],
                      [279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289]]
user_id_year_long = [[6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 57],
                     [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92],
                     [128, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
                     [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161],
                     [179, 180, 181, 182, 183, 184],
                     [203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
                     [237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250],
                     [267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278],
                     [291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303]]
user_id_year_long_for_distance = [[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15], [31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 57],
                     [64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 59, 60, 62, 63, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 411, 408, 409, 416],
                     [93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127],
                     [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 418, 407, 413, 414, 415],
                     [179, 180, 181, 182, 183, 184],
                     [203, 204, 205, 207, 208, 209, 210, 211, 212, 213, 214, 215, 406, 404, 412, 402, 403,   192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 185, 186, 187, 188, 189, 190, 191],
                     [237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 401, 417, 405, 410,   216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236],
                     [267, 268, 270, 272, 273, 274, 275, 276, 277, 278],
                     [291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303]]
rank_xlabel = ['(2, 30)', '(4, 29)', '(5, 27)', '(7, 39)', '(8, 42)', '(9, 38)', '(13, 25)', '(14, 23)', '(169, 197)',
               '(19, 43)', '(20, 47)', '(26, 77)', '(180, 209)', '(40, 80)', '(52, 60)', '(57, 62)', '(198, 236)',
               '(174, 199)', '(91, 115)', '(172, 196)', '(195, 242)', '(188, 211)', '(127, 168)', '(189, 231)',
               '(154, 163)', '(155, 166)', '(162, 185)', '(273, 288)', '(177, 201)', '(178, 202)', '(181, 234)',
               '(182, 213)', '(200, 235)', '(214, 240)', '(233, 250)', '(268, 279)', '(270, 284)', '(286, 301)']
traj_xlabel = ['(2, 30)', '(4, 29)', '(5, 27)', '(7, 39)', '(8, 42)', '(9, 38)', '(169, 197)',
               '(26, 77)', '(180, 209)', '(40, 80)', '(198, 236)',
               '(174, 199)', '(91, 115)', '(172, 196)', '(189, 231)',
               '(162, 185)', '(177, 201)', '(178, 202)',
               '(182, 213)', '(200, 235)', '(214, 240)']

male_list = {401:'Y2012', 402:'Y2011', 403:'Y2011', 404:'Y2011', 405:'Y2012', 406:'Y2011',
           407:'Y2009', 408:'Y2007', 409:'Y2007', 410:'Y2012', 411:'Y2007', 412:'Y2011',
           413:'Y2009', 414:'Y2009', 415:'Y2009', 416:'Y2007',417:'Y2012', 418:'Y2009'}


Year = "Year 3"
y = 3
firstP = firstP_list[y - 1]
secondP = secondP_list[y - 1]
thirdP = thirdP_list[y - 1]
remove = 58
r = False

rg = Gy()
co = Cohen()
CohVsGylog(rg, co)
CohVsGy(rg, co)
CohVsGy_Interval_distr(rg, co)
LeaderFollower()
pairwiseAnay()
distOrigi()
nngraph()
ShortDisFindLeader()
StatNumAnimal()
CountAnimalEachYear()
DrawMap()
new_original_distance_inter6()
DrawMapBaseMap()
GenerateVideo()
CountAnimalEachYearNew()
Data_interpolation()
disTocenterHeapmap()
GetLongVector()
GetEachUserLocationVector()
timeLaggedPlot()
MatrixW()
SpeedDistribution()
RandomSimulation()
SimulationVideo()
compareAngleofRandom_TrueData()
returnPartionDataLocation()
compareAngleofTrue()
compareAngleofRandom()
CompareTrueRandom()
returnPartionDataLocation()
GRSimulted()
GrSimulationAllAlpha()
GRTrue()
AllShortTripGR()
FFTCompare(firstP, secondP, thirdP)
angleDistribution()
CountAnimal_AllYearNew()
TotalRandomSimulation()
heatmap_2d()
TotalRandomVideo()
CenterMassShortLong()
TotalRandomGy()
TotalRandomCohen()
simulationCenterMassshort()
heatmap_map_centermassSimulation()
SimulatedTwoMethodGyVSDis()
SimulatedTwoMethodCohVSDis()
SimulatedTwoMethodCohVSGy()
NewdisTocenterHeapmap()
CalculateCorrleation()
lagVerify()
usingLagverify()
CheckRepeatId()
separateUser()
DrawSameAnimal()
checkRanking()
RealEveryYearId()
lags_distance()
var_ang_distance()
return_point_dis()
spe_dis_dis()
traj_ks_test()
return_ranking()
rank_ks_test()
DrawSpread()
neig_lags(par=1)
usingNeig_lags()
lag_distribution()
lags_distr_oneyear()
lags_dist_neig()
area_km2()
Compute_distance_score()
animals_distance_from_original()
group_animal_distance_from_original()
group_animal_return_time()
male_heatmap_fig()
female_male_compare()