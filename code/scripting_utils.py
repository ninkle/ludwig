
def summarize_hps(hps):
    """
    hps: a list of dictionaries containing the relevant hps
    """
    line = "----------------------------------------"
    print(line)
    print("experiment hps")
    print(line)
    for d in hps:
        for k, v in d.items():
            print(k+": "+str(v))
    print(line) 