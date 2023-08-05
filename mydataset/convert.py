import os
import json
import guitarpro # Note: works only with PyGuitarPro version 0.6
import guitarpro as gp
from collections import Counter
import re
import json
import math
from fractions import Fraction
import sys

effect_list = {
    # "dead" : "ded",
    # "ghost_note" : "gst",
    "harmonic" : "har",
    "vibrato" : "vib",
    "bend" : "bn",
    # "slide" : "sl",
    # "hammer" : "h_p",
    "trill" : "trl"
    # "palm_mute" : "brm",
    # "staccato" : "st",
    # "slap_effect" : "slp"
}

def translate_trackbeats(track, inst):
    measure_note = []
    # select instrument
    inst_track = track[inst]
    # process at each moment
    for clock in inst_track:
        clock_notes = {}
        position = {"1": "-", "2": "-", "3": "-", "4": "-", "5": "-", "6": "-"}
        effects = {"1": "-", "2": "-", "3": "-", "4": "-", "5": "-", "6": "-"}
        ties = {"1": "-", "2": "-", "3": "-", "4": "-", "5": "-", "6": "-"}
        # process of ach note
        for string in track[clock]["notes"]:
            # only for notes
            if string["token"].split(":")[1] == "note":
                # position
                string_number = string["token"].split(":")[2][1:]
                fret_number = int(string["token"].split(":")[3][1:])
                position[string_number] = fret_number
                # effects
                nfx = string["nfx"]
                for l in nfx:
                    if l.split(":")[1] not in effect_list.keys():
                        pass
                    #tie
                    elif l.split(":")[1] == "tie":
                        ties[string_number] = "'"
                    #bending
                    elif l.split(":")[1] == "bend":
                        #searching max of bend degree
                        max_bend_degree = 0
                        for m in l.split(":"):
                            if m[:3] == "val" and int(m[3:]) > max_bend_degree:
                                max_bend_degree = int(m[3:])
                        #rounding degree
                        # half-tone = 4, whole-tone = 8, max-tone = 12
                        if max_bend_degree <= 6:
                            effects[string_number] = effect_list["bend"] + "1"
                        elif 6 < max_bend_degree <= 10:
                            effects[string_number] = effect_list["bend"] + "2"
                        else:
                            effects[string_number] = effect_list["bend"] + "3"
                    #trill
                    elif l.split(":")[1] == "trill":
                        fret_distance = l.split(":")[2][4:]
                        effects[string_number] = effect_list["trill"] + fret_distance
                    #other effects
                    else:
                        effects[string_number] = effect_list[l.split(":")[1]]
        duration = track[clock]["duration"]
        clock_notes["position"] = position
        clock_notes["effects"] = effects
        clock_notes["duration"] = duration
        measure_note.append(clock_notes)
    return measure_note

def translate_all_measure(content, inst):
    time = 960
    repeat_clock = 0
    all_notes = []
    #process each measure
    for measure in content:
        # {
        #     "trackbeats":{
        #         "clean0": {
        #             "960": {
        #                 "bfx": [],
        #                 "notes": [
        #                     {
        #                         "token": "clean0:note:s1:f4",
        #                         "nfx":[
        #                               "nfx:staccato",
        #                               "nfx:bend:type1:pos0:val0:vib0:pos6:val4:vib0:pos12:val4:vib0"
        #                             ]
        #                         },
        #                     {
        #                         "token": "clean0:note:s5:f4",
        #                         "nfx": [
        #                             "nfx:slide:1"
        #                         ]
        #                     }
        #                     ],
        #                 "duration": 960
        #                 },
        #             "1920": {...}
        #             }
        #         },
        #     "measure_tokens":[],
        #     "clock": 960
        # }
        
        # measure_clock
        measure_clock = measure["clock"]
        
        # measure_tokens
        measure_tokens = measure["measure_tokens"]
        repeat_open_clock = 0
        repeat_close = False
        repeat_alternative = 0
        for measure_token in measure_tokens:
            if measure_token.split(":")[1] == "repeat_open":
                repeat_open_clock = measure_clock
            elif measure_token.split(":")[1] == "repeat_close":
                repeat_close = True
            elif measure_token.split(":")[1] == "repeat_alternative":
                
        
        # trackbeats
        notes_moment = translate_trackbeats(measure["trackbeats"], inst)
        all_notes.append(notes_moment)
                
        
def main():
    #load json file
    file = "labels/test2.json"
    data = open(file, "r")
    load = json.load(data)
    #seperate head and content
    head = load[0]
    content = load[1:]
    all_data = []
    print(head)
    #translate content
    for measure in content:
        # {
        #     "trackbeats":{
        #         "clean0": {
        #             "960": {
        #                 "bfx": [],
        #                 "notes": [
        #                     {
        #                         "token": "clean0:note:s1:f4",
        #                         "nfx":[
        #                               "nfx:staccato",
        #                               "nfx:bend:type1:pos0:val0:vib0:pos6:val4:vib0:pos12:val4:vib0"
        #                             ]
        #                         },
        #                     {
        #                         "token": "clean0:note:s5:f4",
        #                         "nfx": [
        #                             "nfx:slide:1"
        #                         ]
        #                     }
        #                     ],
        #                 "duration": 960
        #                 },
        #             "1920": {...}
        #             }
        #         },
        #     "measure_tokens":[],
        #     "clock": 960
        # }
        # notes of clean0
        track = measure["trackbeats"]["clean0"]
        # process at each moment
        for clock in track:
            position = {"1": "-", "2": "-", "3": "-", "4": "-", "5": "-", "6": "-"}
            effects = {"1": "-", "2": "-", "3": "-", "4": "-", "5": "-", "6": "-"}
            ties = {"1": "-", "2": "-", "3": "-", "4": "-", "5": "-", "6": "-"}
            # process of ach note
            for k in track[clock]["notes"]:
                # only for notes
                if k["token"].split(":")[1] == "note":
                    # position
                    string = k["token"].split(":")[2][1:]
                    fret = int(k["token"].split(":")[3][1:])
                    position[string] = fret
                    # effects
                    nfx = k["nfx"]
                    for l in nfx:
                        if l.split(":")[1] not in effect_list.keys():
                            pass
                        #tie
                        elif l.split(":")[1] == "tie":
                            ties[string] = "'"
                        #bending
                        elif l.split(":")[1] == "bend":
                            #searching max of bend degree
                            max_bend_degree = 0
                            for m in l.split(":"):
                                if m[:3] == "val" and int(m[3:]) > max_bend_degree:
                                    max_bend_degree = int(m[3:])
                            #rounding degree
                            # half-tone = 4, whole-tone = 8, max-tone = 12
                            if max_bend_degree <= 6:
                                effects[string] = effect_list["bend"] + "1"
                            elif 6 < max_bend_degree <= 10:
                                effects[string] = effect_list["bend"] + "2"
                            else:
                                effects[string] = effect_list["bend"] + "3"
                        #trill
                        elif l.split(":")[1] == "trill":
                            fret_distance = l.split(":")[2][4:]
                            effects[string] = effect_list["trill"] + fret_distance
                        #other effects
                        else:
                            effects[string] = effect_list[l.split(":")[1]]
            duration = track[clock]["duration"]
            print("clock", clock)
            print("position", position)
            print("effects", effects)            
            print("duration", duration)
            
                
            

if __name__ == "__main__":
    main()