"""
Created on Thu Dec 14 10:12:13 2023
@author: Shulei Ji
"""
import muspy
import numpy as np
from muspy.metrics.metrics import pitch_range,n_pitch_classes_used,polyphony
import os
import pickle
import matplotlib.pyplot as plt


def compute_object_metrics(path):
    midi_files=os.listdir(path)
    PR = [];NPC = [];POLY = []
    cnt = 0
    for midi in midi_files:
        midi_file_path = os.path.join(path, midi)
        music = muspy.read_midi(midi_file_path)
        PR.append(pitch_range(music))
        NPC.append(n_pitch_classes_used(music))
        POLY.append(polyphony(music))
        cnt += 1
    return PR, NPC, POLY


def metrics_per_song(data):
    data=data.tolist()
    note_density_list=[]
    note_pitch_list = []
    bar_pitch = []
    bar_pitch_range=[]
    bar_NPC = []
    bar_POLY=[]
    bar_beat=[]
    bar_beat_real = []
    note_length_list=[]
    note_velocity_list=[]
    tempo_list=[]
    chord_list=[]
    beat_list=[]
    type_list=[]
    for i in range(len(data)):
        type_list.append(data[i][3])
        if data[i][3]==3:
            if Duration_dict[data[i][5]]!=0 and Velocity_dict[data[i][6]]!=0:
                duration_index=Duration_dict[data[i][5]].rindex('_')
                note_length=int(Duration_dict[data[i][5]][duration_index+1:])/480
                velocity_index=Velocity_dict[data[i][6]].rindex('_')
                note_velocity=int(Velocity_dict[data[i][6]][velocity_index+1:])
                pitch_index = Pitch_dict[data[i][4]].rindex('_')
                note_pitch = int(Pitch_dict[data[i][4]][pitch_index + 1:])
                note_length_list.append(note_length)
                note_velocity_list.append(note_velocity)
                note_pitch_list.append(note_pitch)
                bar_pitch.append(note_pitch)
                if len(bar_beat)==0:
                    bar_beat_real.append(0)
                else:
                    bar_beat_real.append(bar_beat[-1])
        if data[i][3]==2:
            if data[i][2]!=1:
                if Tempo_dict[data[i][0]]!=0:
                    if Tempo_dict[data[i][0]]=='CONTI':
                        if len(tempo_list)==0:
                            continue
                        tempo_list.append(tempo_list[-1])
                    else:
                        tempo_index=Tempo_dict[data[i][0]].rindex('_')
                        tempo=int(Tempo_dict[data[i][0]][tempo_index+1:])
                        tempo_list.append(tempo)
                if chord_dict[data[i][1]] != 0:
                    if chord_dict[data[i][1]]=='CONTI':
                        if len(chord_list)==0:
                            continue
                        chord_list.append(chord_list[-1])
                    else:
                        chord_index = chord_dict[data[i][1]].rindex('_')
                        chord = chord_dict[data[i][1]][chord_index + 1:]
                        chord_list.append(chord)
                if data[i][2] > 1:
                    beat_index = beat_dict[data[i][2]].rindex('_')
                    beat = int(beat_dict[data[i][2]][beat_index + 1:])
                    beat_list.append(beat)
                    bar_beat.append(beat)
            else:
                if len(bar_pitch)>0:
                    bar_pitch_range.append(max(bar_pitch)-min(bar_pitch))
                    bar_pitch_temp = set([i % 12 for i in bar_pitch])
                    bar_NPC.append(len(bar_pitch_temp))
                    if len(bar_beat)>0:
                        bar_POLY.append(len(bar_pitch)/len(set(bar_beat_real)))
                bar_pitch=[]
                bar_beat=[]
                bar_beat_real=[]
    if len(bar_pitch) > 0:
        bar_pitch_range.append(max(bar_pitch) - min(bar_pitch))
        bar_pitch_temp = set([i % 12 for i in bar_pitch])
        bar_NPC.append(len(bar_pitch_temp))
    assert len(note_length_list)==len(note_velocity_list)
    return type_list,note_pitch_list,note_density_list,note_length_list,note_velocity_list,tempo_list,chord_list,beat_list,\
           np.mean(bar_pitch_range),np.mean(bar_NPC),np.mean(bar_POLY)


def metrics_all_song(data,index):
    data_stat = {"note_density":{1:[],2:[],3:[],4:[]},
                 "note_length":{1:[],2:[],3:[],4:[]},
                 "note_velocity":{1:[],2:[],3:[],4:[]},
                 "note_pitch":{1:[],2:[],3:[],4:[]},
                 "tempo":{1:[],2:[],3:[],4:[]},
                 "chord":{1:[],2:[],3:[],4:[]},
                 "beat":{1:[],2:[],3:[],4:[]},
                 "type":{1:[],2:[],3:[],4:[]}}
    bar_pitch_range=[]
    bar_NPC=[]
    bar_POLY=[]
    for i in range(len(data)):
        type_list,note_pitch_list,note_density_list,note_length_list,note_velocity_list,tempo_list,chord_list,beat_list,\
        BPR,BNPC,BPOLY=metrics_per_song(data[i])
        bar_pitch_range.append(BPR)
        bar_NPC.append(BNPC)
        bar_POLY.append(BPOLY)
        if i in index['cls_1_idx']:
            data_stat["note_pitch"][1].extend(note_pitch_list)
            data_stat["note_density"][1].extend(note_density_list)
            data_stat["note_length"][1].extend(note_length_list)
            data_stat["note_velocity"][1].extend(note_velocity_list)
            data_stat["tempo"][1].extend(tempo_list)
            data_stat["chord"][1].extend(chord_list)
            data_stat["beat"][1].extend(beat_list)
            data_stat["type"][1].extend(type_list)
        elif i in index['cls_2_idx']:
            data_stat["note_pitch"][2].extend(note_pitch_list)
            data_stat["note_density"][2].extend(note_density_list)
            data_stat["note_length"][2].extend(note_length_list)
            data_stat["note_velocity"][2].extend(note_velocity_list)
            data_stat["tempo"][2].extend(tempo_list)
            data_stat["chord"][2].extend(chord_list)
            data_stat["beat"][2].extend(beat_list)
            data_stat["type"][2].extend(type_list)
        elif i in index['cls_3_idx']:
            data_stat["note_pitch"][3].extend(note_pitch_list)
            data_stat["note_density"][3].extend(note_density_list)
            data_stat["note_length"][3].extend(note_length_list)
            data_stat["note_velocity"][3].extend(note_velocity_list)
            data_stat["tempo"][3].extend(tempo_list)
            data_stat["chord"][3].extend(chord_list)
            data_stat["beat"][3].extend(beat_list)
            data_stat["type"][3].extend(type_list)
        elif i in index['cls_4_idx']:
            data_stat["note_pitch"][4].extend(note_pitch_list)
            data_stat["note_density"][4].extend(note_density_list)
            data_stat["note_length"][4].extend(note_length_list)
            data_stat["note_velocity"][4].extend(note_velocity_list)
            data_stat["tempo"][4].extend(tempo_list)
            data_stat["chord"][4].extend(chord_list)
            data_stat["beat"][4].extend(beat_list)
            data_stat["type"][4].extend(type_list)
    return data_stat,bar_pitch_range,bar_NPC,bar_POLY


np.set_printoptions(threshold=np.inf)


def violin_plot(data,index):

    data_stat,_,_,_=metrics_all_song(data,index)
    chord_type_dict = {'N': 0, 'M7': 1, 'M': 2, 'm': 3, 'm7': 4, 'o': 5, 'sus2': 6, '7': 7, 'sus4': 8, '+': 9,
                       '/o7': 10, 'o7': 11}
    plt.figure(figsize=(8,4.5))

    plt.subplot(2,3,5)
    plt.violinplot([data_stat["note_length"][1], data_stat["note_length"][2],
                    data_stat["note_length"][3], data_stat["note_length"][4]],
                   showextrema=True, showmedians=True,points=1000,widths=0.8)
    for i, d in enumerate([data_stat["note_length"][1], data_stat["note_length"][2],
                    data_stat["note_length"][3], data_stat["note_length"][4]]):
        min_value, quantile1, median, quantile3, max_value = np.percentile(d, [0, 25, 50, 75, 100])
        plt.scatter(i + 1, median, color='white', zorder=8,s=6)
        plt.vlines(i + 1, quantile1, quantile3, lw=6, zorder=3)

    plt.xticks(ticks=[1,2,3,4],labels=["Q1","Q2","Q3","Q4"])
    plt.ylabel("Duration")

    plt.subplot(2,3,6)

    plt.violinplot([data_stat["note_velocity"][1], data_stat["note_velocity"][2],
                    data_stat["note_velocity"][3], data_stat["note_velocity"][4]],
                   showextrema=True, showmedians=True,points=1000,widths=0.8)
    for i, d in enumerate([data_stat["note_velocity"][1], data_stat["note_velocity"][2],
                    data_stat["note_velocity"][3], data_stat["note_velocity"][4]]):
        min_value, quantile1, median, quantile3, max_value = np.percentile(d, [0, 25, 50, 75, 100])
        # print(median)
        plt.scatter(i + 1, median, color='white', zorder=8,s=6)
        plt.vlines(i + 1, quantile1, quantile3, lw=6, zorder=3)
    plt.xticks(ticks=[1, 2, 3, 4], labels=["Q1", "Q2", "Q3", "Q4"])
    plt.ylabel("Velocity")

    plt.subplot(2, 3, 1)

    plt.violinplot([data_stat["tempo"][1], data_stat["tempo"][2],
                    data_stat["tempo"][3], data_stat["tempo"][4]],
                   showextrema=True, showmedians=True, points=1000,widths=0.8)
    for i, d in enumerate([data_stat["tempo"][1], data_stat["tempo"][2],
                    data_stat["tempo"][3], data_stat["tempo"][4]]):
        min_value, quantile1, median, quantile3, max_value = np.percentile(d, [0, 25, 50, 75, 100])
        plt.scatter(i + 1, median, color='white', zorder=8,s=6)
        plt.vlines(i + 1, quantile1, quantile3, lw=6, zorder=3)
    plt.xticks(ticks=[1, 2, 3, 4], labels=["Q1", "Q2", "Q3", "Q4"])
    plt.ylabel("Tempo")

    plt.subplot(2, 3, 2)

    def chordType2int(data):
        new_data = []
        for j in data:
            new_data.append(chord_type_dict[j])
        return new_data

    plt.violinplot([chordType2int(data_stat["chord"][1]), chordType2int(data_stat["chord"][2]),
                    chordType2int(data_stat["chord"][3]), chordType2int(data_stat["chord"][4])],
                   showextrema=True, showmedians=True, points=1000, widths=0.8)
    for i, d in enumerate([chordType2int(data_stat["chord"][1]), chordType2int(data_stat["chord"][2]),
                           chordType2int(data_stat["chord"][3]), chordType2int(data_stat["chord"][4])]):
        min_value, quantile1, median, quantile3, max_value = np.percentile(d, [0, 25, 50, 75, 100])
        plt.scatter(i + 1, median, color='white', zorder=8, s=6)
        plt.vlines(i + 1, quantile1, quantile3, lw=6, zorder=3)
    plt.xticks(ticks=[1, 2, 3, 4], labels=["Q1", "Q2", "Q3", "Q4"])
    plt.yticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11],
               labels=['N','M7', 'M', 'm', 'm7', 'o', 'sus2', '7', 'sus4', '+', '/o7', 'o7'])
    plt.ylabel("Chord")

    plt.subplot(2, 3, 4)
    #
    plt.violinplot([data_stat["note_pitch"][1], data_stat["note_pitch"][2],
                    data_stat["note_pitch"][3], data_stat["note_pitch"][4]],
                   showextrema=True, showmedians=True, points=1000, widths=0.8)
    for i, d in enumerate([data_stat["note_pitch"][1], data_stat["note_pitch"][2],
                           data_stat["note_pitch"][3], data_stat["note_pitch"][4]]):
        min_value, quantile1, median, quantile3, max_value = np.percentile(d, [0, 25, 50, 75, 100])
        plt.scatter(i + 1, median, color='white', zorder=8, s=6)
        plt.vlines(i + 1, quantile1, quantile3, lw=6, zorder=3)
    plt.xticks(ticks=[1, 2, 3, 4], labels=["Q1", "Q2", "Q3", "Q4"])
    plt.ylabel("Pitch")

    plt.subplot(2, 3, 3)

    plt.violinplot([data_stat["beat"][1], data_stat["beat"][2],
                    data_stat["beat"][3], data_stat["beat"][4]],
                   showextrema=True, showmedians=True, points=1000, widths=0.8)
    for i, d in enumerate([data_stat["beat"][1], data_stat["beat"][2],
                           data_stat["beat"][3], data_stat["beat"][4]]):
        min_value, quantile1, median, quantile3, max_value = np.percentile(d, [0, 25, 50, 75, 100])
        plt.scatter(i + 1, median, color='white', zorder=8, s=6)
        plt.vlines(i + 1, quantile1, quantile3, lw=6, zorder=3)
    plt.xticks(ticks=[1, 2, 3, 4], labels=["Q1", "Q2", "Q3", "Q4"])
    plt.ylabel("Beat")

    plt.tight_layout()


if __name__=="__main__":
    fr = open('./data/co-representation/dictionary.pkl', 'rb')
    data = pickle.load(fr)

    Pitch_dict = data[1]['pitch']
    Duration_dict = data[1]['duration']
    Velocity_dict = data[1]['velocity']
    Tempo_dict = data[1]['tempo']
    chord_dict = data[1]['chord']
    beat_dict = data[1]['bar-beat']

    data_file=open("generate_midi/MusER_TRANS_CA_ER_emopia_data.data",'rb')
    data=pickle.load(data_file)
    index_file=open("generate_midi/MusER_TRANS_CA_ER_emopia_index.data",'rb')
    index=pickle.load(index_file)
    violin_plot(data,index)
    eps = plt.gcf()
    eps.savefig('plot/MusER_TRANS_CA_ER_emopia.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.show()

    _,BPR,BNPC,BPOLY=metrics_all_song(data,index)
    PR, NPC, POLY=compute_object_metrics("generate_midi/MusER_TRANS_CA_ER_emopia/")
    print("PR=",PR)
    print("BPR=",BPR)
    print("NPC=",NPC)
    print("BNPC=",BNPC)
    print("POLY=",POLY)
    print("BPOLY=",BPOLY)

    precision=2
    print("PR: ",round(np.mean(PR),precision),"±",round(np.std(PR),precision),\
          ", BPR: ",round(np.mean(BPR),precision),"±",round(np.std(BPR),precision),\
          ", NPC: ",round(np.mean(NPC),precision),"±",round(np.std(NPC),precision),\
          ", BNPC: ",round(np.mean(BNPC),precision),"±",round(np.std(BNPC),precision),\
          ", POLY: ",round(np.mean(POLY),precision),"±",round(np.std(POLY),precision),\
          ", BPOLY: ",round(np.mean(BPOLY),precision),"±",round(np.std(BPOLY),precision))