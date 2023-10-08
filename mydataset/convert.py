import os
import json
import pprint
import sys

track = {
    "clean0": "clean0",
}

effect_list = {
    # "dead" : "ded",
    # "ghost_note" : "gst",
    "harmonic" : "har",
    "vibrato" : "vib",
    "bend" : "bn",
    # "slide" : "sld",
    # "hammer" : "h_p",
    "trill" : "trl",
    # "palm_mute" : "brm",
    # "staccato" : "stc",
    # "slap_effect" : "slp"
    "tie" : "tie"
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
        for string in inst_track[clock]["notes"]:
            # only for notes
            if string["token"].split(":")[1] == "note":
                # position
                string_number = string["token"].split(":")[2][1:]
                fret_number = int(string["token"].split(":")[3][1:])
                position[string_number] = str(fret_number)
                # effects
                nfx = string["nfx"]
                for l in nfx:
                    if l.split(":")[1] not in effect_list.keys():
                        pass
                    #tie
                    elif l.split(":")[1] == "tie":
                        ties[string_number] = 't'
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
                        fret_distance = int(l.split(":")[2][4:]) - fret_number
                        effects[string_number] = effect_list["trill"] + str(fret_distance)
                    #other effects
                    else:
                        effects[string_number] = effect_list[l.split(":")[1]]
        duration = inst_track[clock]["duration"]
        clock_notes["position"] = position
        clock_notes["effects"] = effects
        clock_notes["ties"] = ties
        clock_notes["duration"] = duration
        measure_note.append(clock_notes)
    return measure_note

def translate_all_measure(content, inst):
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
    all_notes = []
    measure_index = 0
    repeat_close_clocks = []
    print(content[0])
    #process each measure
    while measure_index < len(content):
        measure = content[measure_index]
        
        # measure_clock
        measure_clock = measure["clock"]
        
        # trackbeats
        measure_notes = translate_trackbeats(measure["trackbeats"], inst)
        for measure_note in measure_notes:
            all_notes.append(measure_note)
        
        # measure_tokens
        measure_tokens = measure["measure_tokens"]
        repeat_close_clock = False
        for measure_token in measure_tokens:
            if measure_token.split(":")[1] == "repeat_open":
                repeat_open_clock = measure_clock
            elif measure_token.split(":")[1] == "repeat_close":
                repeat_close_clock = True
        
        # update measure_index
        if repeat_close_clock and measure_clock not in repeat_close_clocks:
            for i in range(measure_index, -1, -1):
                if content[i]["clock"] == repeat_open_clock:
                    measure_index = i
                    repeat_close_clock = False
                    repeat_close_clocks.append(measure_clock)
                    break
        else:
            measure_index += 1
    
    return all_notes

def encode(file, inst, output):
    #load json file
    data = open(file, "r")
    load = json.load(data)
    #seperate head and content
    head = load[0]
    content = load[1:]
    #translate content
    all_notes = translate_all_measure(content, inst)
    #concatenate head and content
    concatenate = [head] + all_notes
    #write to json file
    with open(output, "w") as f:
        json.dump(concatenate, f, indent=4)
        
def main():
    usage = """
ENCODE
If you want to convert json file, please use this command:
    python convert.py encode [json file] [instrument] [output file]
    ex) python convert.py encode labels/test2.json clean0 test2.json
    """
    assert len(sys.argv) >= 4, usage + "\nError: Not enough arguments."
    if sys.argv[1] == "encode":
        assert os.path.exists(sys.argv[2]), usage + "\nError: File not found."
        assert sys.argv[3] in track.keys(), usage + "\nError: Invalid instrument."
        encode(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(usage)
        print("Error: Invalid command.")
        sys.exit(1)
        
if __name__ == "__main__":
    main()