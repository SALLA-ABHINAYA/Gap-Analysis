import requests
import traceback
import os


def make_api_call(prompt, api_key, api_model) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": api_model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    content = r.json()["choices"][0]["message"]["content"]
    return content


def convert_to_json(ocel):
    # Convert timestamps to ISO format strings for JSON serialization
    events_df = ocel.events.copy()
    events_df['ocel:timestamp'] = events_df['ocel:timestamp'].dt.isoformat()

    relations_df = ocel.relations.copy()
    relations_df['ocel:timestamp'] = relations_df['ocel:timestamp'].dt.isoformat()

    ocel_json = {
        "ocel:events": events_df.to_dict(orient='records'),
        "ocel:objects": ocel.objects.to_dict(orient='records'),
        "ocel:relations": relations_df.to_dict(orient='records'),
        "ocel:o2o": ocel.o2o.to_dict(orient='records') if hasattr(ocel, 'o2o') else [],
        "ocel:global-log": {
            "ocel:version": "2.0",
            "ocel:ordering": "timestamp",
            "ocel:attribute-names": [col for col in events_df.columns if col.startswith('ocel:')],
            "ocel:object-types": ocel.objects['ocel:type'].unique().tolist()
        }
    }
    return ocel_json

def simulate_process(target_process, api_key, description_generation_model, simulation_generation_model,
                     output_file="output.json", simulation_script="simscript.py"):
    with open("target_process.txt", "w") as f:
        f.write(target_process)

    process_description_request = "Generate a description of real-life process with many different object types and high degree of variability, batching, synchronization, and workarounds. describe everything in detail."

    process_simulation_generation_request = """
    The OCEL object in pm4py is stored in the class OCEL contained in pm4py.objects.ocel.obj

    In the pm4py process mining library, the OCEL class is a collection of different dataframes, containing at least the following columns:
    The constructor of OCEL objects is __init__(self, events=None, objects=None, relations=None, globals=None, parameters=None, o2o=None, e2e=None,
                 object_changes=None)


    ocel.events
     #   Column          Non-Null Count  Dtype
    ---  ------          --------------  -----
     0   ocel:eid        23 non-null     string
     1   ocel:timestamp  23 non-null     datetime64[ns, UTC]
     2   ocel:activity   23 non-null     string

    ocel.objects
    #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   ocel:oid   15 non-null     string
     1   ocel:type  15 non-null     string

    ocel.relations 
     #   Column          Non-Null Count  Dtype
    ---  ------          --------------  -----
     0   ocel:eid        39 non-null     string
     1   ocel:activity   39 non-null     string
     2   ocel:timestamp  39 non-null     datetime64[ns, UTC]
     3   ocel:oid        39 non-null     string
     4   ocel:type       39 non-null     string
     5   ocel:qualifier  0 non-null      object


    The 'ocel.relations' dataframe contains the many-to-many relationships between events and objects (E2O). An event can be related to different objects of different object types. The relationship can be qualified (i.e., described by a qualifier).

    Moreover, there is an ocel.o2o dataframe containing the object to object (O2O) relationships. Also these relationships can be qualified.


     #   Column          Non-Null Count  Dtype
    ---  ------          --------------  -----
     0   ocel:oid        0 non-null      string
     1   ocel:oid_2      0 non-null      string
     2   ocel:qualifier  0 non-null      string


    Could you create a Python script to simulate an object-centric event log?
    Please include at least 30 different activities in the object-centric event log and at least 6 different object types.
    Please include at least 5000 events and 5000 objects.
    Do not randomly choose activities! Make sure that the process flow is consistent with the process!

    The result should be stored in the "ocel" variable.

    Include attributes at the events and objects level.
    Include different types of behavior, including batching, synchronization, and workarounds.
    Include a high degree of variability in the object-centric event log.

    <ProcessDescription>
    !!REPLACE HERE!!
    </ProcessDescription>

    The object-centric event log should resemble a real-life process. The activities and object types should have realistic names.
    Please also include the following lines at the end of the script to clean it up and export it as a JSON file:

    import pm4py
    from pm4py.objects.ocel.util import ocel_consistency
    from pm4py.objects.ocel.util import filtering_utils

    ocel_events = ocel.events[["ocel:eid", "ocel:activity", "ocel:timestamp"]].to_dict("records")
    ocel_objects = ocel.objects[["ocel:oid", "ocel:type"]].to_dict("records")
    ocel_id_act = {x["ocel:eid"]: x["ocel:activity"] for x in ocel_events}
    ocel_id_time = {x["ocel:eid"]: x["ocel:timestamp"] for x in ocel_events}
    ocel_objects = {x["ocel:oid"]: x["ocel:type"] for x in ocel_objects}
    ocel.relations["ocel:activity"] = ocel.relations["ocel:eid"].map(ocel_id_act)
    ocel.relations["ocel:timestamp"] = ocel.relations["ocel:eid"].map(ocel_id_time)
    ocel.relations["ocel:type"] = ocel.relations["ocel:oid"].map(ocel_objects)
    ocel.relations.dropna(subset=["ocel:activity", "ocel:timestamp", "ocel:type"], inplace=True)
    if "ocel:qualifier" not in ocel.relations.columns:
        ocel.relations["ocel:qualifier"] = [None] * len(ocel.relations)
    ocel = ocel_consistency.apply(ocel)
    ocel = filtering_utils.propagate_relations_filtering(ocel)

    # Convert OCEL to JSON and save
    ocel_json = {
        "ocel:events": ocel.events.to_dict(orient='records'),
        "ocel:objects": ocel.objects.to_dict(orient='records'),
        "ocel:relations": ocel.relations.to_dict(orient='records'),
        "ocel:o2o": ocel.o2o.to_dict(orient='records') if hasattr(ocel, 'o2o') else [],
        "ocel:global-log": {
            "ocel:version": "2.0",
            "ocel:ordering": "timestamp",
            "ocel:attribute-names": [col for col in ocel.events.columns if col.startswith('ocel:')],
            "ocel:object-types": ocel.objects['ocel:type'].unique().tolist()
        }
    }
    
    # Convert timestamps to ISO format strings
    for event in ocel_json["ocel:events"]:
        event["ocel:timestamp"] = event["ocel:timestamp"].isoformat()
    for relation in ocel_json["ocel:relations"]:
        relation["ocel:timestamp"] = relation["ocel:timestamp"].isoformat()
    
    with open("output.json", "w") as f:
        json.dump(ocel_json, f, indent=2)

    After the script, include a <description> XML tag containing the textual description of the simulated process model
    (for non-technical process analysts).
    """

    if target_process:
        process_description_request += "\n\nPlease focus on the following process: " + target_process

    ite = 0
    while not os.path.exists(output_file):
        ite += 1
        with open("iterations.txt", "w") as f:
            f.write(str(ite))

        print("\n\n== Generating Process Description ==\n\n")
        process_description = make_api_call(process_description_request, api_key=api_key,
                                            api_model=description_generation_model)

        process_description = process_simulation_generation_request.replace("!!REPLACE HERE!!", process_description)
        process_description = process_description.replace("output.json", output_file)

        with open("process_description.txt", "w") as f:
            f.write(process_description)

        print(process_description)
        print("\n\n== Generating Process Simulation ==\n\n")

        try:
            process_simulation = make_api_call(process_description, api_key=api_key,
                                               api_model=simulation_generation_model)
            with open("process_simulation.txt", "w") as f:
                f.write(process_simulation)

            print(process_simulation)
            process_simulation = process_simulation.split("```python")[1].split("```")[0]

            with open(simulation_script, "w") as f:
                f.write(process_simulation)

            os.system("python " + simulation_script)
        except Exception as e:
            traceback.print_exc()
