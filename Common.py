import pickle

def OpenFile(file):
    with open('Pickle_files/' + file, 'rb') as fi:
        data = pickle.load(fi)
    return data 
    
def WriteFile(file, data):
    with open('Pickle_files/' + file, 'wb') as fi:
        pickle.dump(data, fi)