import pickle

def save_dict(filename, datadict):
    binary_format = pickle.dumps(datadict)
    f = open(filename, 'wb')
    f.write(binary_format)
    f.close()

def load_dict(filename):
    with open( filename, mode='rb') as file:
        binary_format = file.read()
    datadict = pickle.loads(binary_format)
    return datadict

